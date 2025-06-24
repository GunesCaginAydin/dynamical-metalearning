"""
Implementation of the Diffuser Model for dynamical systems, adapted from 
https://github.com/jannerm/diffuser/blob/main/diffuser/models/diffusion.py
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from collections import namedtuple

try:
    from architectures.diffuser.diffuser_utils import *
    from architectures.diffuser.diffusion_models import *
except:
    from Sys_Identification.architectures.diffuser.diffuser_utils import * # type: ignore
    from Sys_Identification.architectures.diffuser.diffusion_models import * # type: ignore


@dataclass
class Configdf:
    seq_len_ctx: int = 200
    seq_len_new: int = 800
    n_embd: int = 192
    n_u: int = 7
    n_y: int = 14
    dropout: float = 0.0
    bias: bool = True

    timesteps: int = 1000
    beta1: float = 1e-4
    beta2: float = 2e-2

    hidden: int = 64
    multiplier: int = 3
    attention: bool = False

    ucond: bool = False
    ycond: bool = False
    epsilon: bool = True
    loss_weight: int = 3    
    diffusion_model: str = 'Unet'
    diftype: int = 1

class TSDiffuser(nn.Module):
    """
    Diffuser class for training on trajectory dataset and sampling on pure noise
    trajectories. Training dataset should be conditioned to adhere with the action/
    observation/transition modalities of the init function st.

    Parameters
    ---
        model (model object): nn model to be diffusion scheduled from diffusion_models.py
        horizon_dimension (int): horizon dimension, 1000 iterations
        action_dimension (int): action/u dimension, 7 joints
        observation_dimension (int): output/y dimension 14/16 coordinates
        transition_dimension (int): u+y = 21/23
        number_of_timesteps (int): noising steps, 1000
        clip_denoised (bool): if True clip objects between (-1,1)
        predict_epsilon (bool): if True predict noise in trajectories
        loss_type (string): selection of noise implementation
        loss_weight (dict): loss multiplier for weighted loss propagation
        loss_discount (int): discount
        action_weight (int): weight
    """
    def __init__(self,
                 config,
                 clip_denoised=False,
                 loss_type='mse',
                 loss_weights=None,
                 loss_discount=1.0,
                 action_weight=1.0,
                 observation_weight=1.0
                 ):
        super().__init__()
        self.horizon_dim = config.seq_len_ctx + config.seq_len_new # horizon of timesteps := iter | 1000/5000
        self.context_dim = config.seq_len_ctx # context available for predictions - clip/condition? := ctx | 200
        self.observation_dim = config.n_y # observation dimension := y | 14/16
        self.action_dim = config.n_u if config.diftype==1 else config.n_y # action dimension := u | 7
        self.transition_dim = self.observation_dim + self.action_dim # y + u = 21/23

        self.nt = int(config.timesteps) # denoising timesteps := 1000
        self.clip_denoised = clip_denoised # clip inouts
        self.predict_epsilon = config.epsilon # predict errors with diffusion

        self.loss_type = loss_type # mse, mae, rmse, huber, logcosh
        self.observation_weight = config.loss_weight

        self.condition_type = [config.ucond, config.ycond] # conditioning of context and forcing, 0,1 := False,True 

        multipliers = [2**x for x in range(config.multiplier+1)]
        self.model = TemporalUnet(horizon=self.horizon_dim,
                                      transition_dim=self.transition_dim,
                                      cond_dim=0,
                                      dim=config.hidden,
                                      dim_mults=multipliers,
                                      attention=config.attention)

        self.register_buffer('betas', # betas used in noise scheduling
                            cosine_beta_schedule(config.timesteps))
        self.register_buffer('alphas', # alpha = √( 1 - beta )
                            1 - self.betas) 
        self.register_buffer('alphas_cumprod', # prod**alpha
                            torch.cumprod(self.alphas, axis=0))
        self.register_buffer('alphas_cumprod_old', # prod**alpha_posterior 
                            torch.cat([torch.ones(1), self.alphas_cumprod[:-1]]))
        self.register_buffer('sqrt_alphas_cumprod',  # √( prod**alpha_posterior )
                            torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', # √( 1 - prod**alpha )
                            torch.sqrt(1.-self.alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', # log( 1 - prod**alpha )
                            torch.log(1.-self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',  # √( 1 / prod**alpha )
                            torch.sqrt(1./self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', # √( (1 - prod**alpha) / prod**alpha )
                            torch.sqrt(1./self.alphas_cumprod-1))
        self.register_buffer('posterior_mean_coef1', # coeff for posterior mean | x0
                            self.betas * np.sqrt(self.alphas_cumprod_old)/(1.-self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', # coeff for posterior mean | xt
                            (1.-self.alphas_cumprod_old)*np.sqrt(self.alphas)/(1.-self.alphas_cumprod))
        self.register_buffer('posterior_variance', # (1 - prod**alpha_posterior) / (1 - prod**alpha)
                            self.betas*(1.-self.alphas_cumprod_old) /(1.-self.alphas_cumprod))
        self.register_buffer('posterior_log_variance_clipped', # log( (1 - prod**alpha_posterior) / (1 - prod**alpha) ) + clip(-1,1)
                            torch.log(torch.clamp(self.posterior_variance, min=1e-20)))
        
        self.loss_weights = self.get_loss_weights(action_weight, self.observation_weight, loss_discount, loss_weights)

    def get_loss_weights(self, action_weight, observation_weight, discount, weights_dict):
        '''
        sets loss coefficients for trajectory
        
        Parameters
        ---
        action_weight (float) : 
            coefficient on first action loss
        discount (float) :
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict (dict) : 
            { i: c } multiplies dimension i of observation loss by c
        
        Returns
        ---
        loss weights (dict)
            { i: c } 
                                    
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        discounts = discount ** torch.arange(self.horizon_dim, dtype=torch.float)
        discounts = discounts / discounts.mean()
        self.loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        self.loss_weights[:, :self.action_dim] = action_weight
        self.loss_weights[:self.context_dim, self.action_dim:] = observation_weight
        return self.loss_weights.to('cuda')
        
    def x0_from_noised_trajectory(self, xt, t, noise):
        """
        Obtains x0 from a noised image without any conditions

        Different parametrization schemes:
            direct parametrization of mean
            parametrization of x0 - start
            parametrization of eps - noise
        """
        if self.predict_epsilon: # predicts eps in sampling, return x0
            return (
                extract_params(self.sqrt_recip_alphas_cumprod, t, xt.shape) * xt -
                extract_params(self.sqrt_recipm1_alphas_cumprod, t, xt.shape) * noise
            )
        else:
            return noise # predicts x0 in sampling, return eps

    def posterior_mean_and_variance(self, xstart, xt, t):
        """
        Posterior probability distribution of q at xt-1
        """
        posterior_mean =  (
            extract_params(self.posterior_mean_coef1, t, xt.shape) * xstart +
            extract_params(self.posterior_mean_coef2, t, xt.shape) * xt
        )

        posterior_var = extract_params(self.posterior_variance, t, xt.shape)

        posterior_log = extract_params(self.posterior_log_variance_clipped, t, xt.shape)

        return posterior_mean, posterior_var, posterior_log

    def get_posterior_mean_and_variance(self, x, cond, force, t):
        """
        Posterior probability distribution of q at xt-1 from x0, xt

            --> q(xt-1) = p(xt-1 | xt,x0)
        """
        xr = self.x0_from_noised_trajectory(xt=x, t=t, noise=self.model(x, cond, force, t))

        if self.clip_denoised:
            xr.clamp_(-1., 1.)
        else:
            pass

        model_mean, posterior_variance, posterior_log_variance = self.posterior_mean_and_variance(
                xstart=xr, xt=x, t=t)
        
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()    
    def sample_function(self, xt, cond, force, t):
        """
        Default sampling function to sample from a noised trajectory, 
        """
        b, *_, device = *xt.shape, xt.device

        model_mean, _, model_log_variance = self.get_posterior_mean_and_variance(x=xt, cond=cond, force=force, t=t)
        model_std = torch.exp(0.5 * model_log_variance)

        noise = torch.randn_like(xt)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(xt.shape) - 1)))

        values = torch.zeros(len(xt), device=xt.device)
        return model_mean + nonzero_mask * (model_std * noise), values

    @torch.inference_mode()
    def sample_from_noisy_distribution(self, shape, cond, force, xwarmstart=False, twarmstart=False, verbose=False, return_chain=False, sample_fn=sample_function, **sample_kwargs):
        """
        Sample a new distribution from a prior noised trajectory, used for testing
        """
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if not xwarmstart else xwarmstart
        x = conditioning(x, cond, force, self.context_dim, self.action_dim, self.condition_type,to='infer')

        chain = [x] if return_chain else None

        progress = Progress(self.nt) if verbose else Silent()
        for i in reversed(range(0, self.nt if not twarmstart else twarmstart)):
            t = create_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, force, t, **sample_kwargs)
            x = conditioning(x, cond, force, self.context_dim, self.action_dim, self.condition_type,to='infer')

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)
        progress.stamp()

        x, values = descending_sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        
        return x, values, chain

    @torch.inference_mode()
    def conditional_sample_from_noisy_distiribution(self, cond, force, horizon=None, **sample_kwargs):
        """
        Sample a new distribution with conditionals from a prior noised trajectory, used for testing
        """
        device = self.betas.device
        batch_size = cond.shape[0]
        horizon = horizon or self.horizon_dim
        shape = (batch_size, horizon, self.transition_dim)

        return self.sample_from_noisy_distribution(shape, cond, force, **sample_kwargs)

    def get_noisy_distribution(self, xstart, t, cond=None, force=None, noise=None):
        """
        Prior distribution of q at xt, user for training
        """
        if noise is None:
            noise = torch.randn_like(xstart)

        sample = (
            extract_params(self.sqrt_alphas_cumprod, t, xstart.shape) * xstart +
            extract_params(self.sqrt_one_minus_alphas_cumprod, t, xstart.shape) * noise            
        )

        return conditioning(sample, cond, force, self.context_dim, self.action_dim, self.condition_type,to='train'), noise
    
    def estimate_loss_on_noisy_distribution(self, xstart, t, noise=None, cond=None, force=None):
        """
        Estimate noise eps or starting distribution x0 from a sampled noisy distribution: to be
        used in the training loop according to selected loss functions, used for training
        """
        xnoisy, noise = self.get_noisy_distribution(xstart=xstart, t=t, noise=noise, cond=cond, force=force)
        est = self.model(xnoisy, cond, force, t)
        est = conditioning(est, cond, force, self.context_dim, self.action_dim, self.condition_type,to='train')
        
        return est, noise, xstart
    
    def get_loss_params(self, x, **kwargs):
        batch_size = len(x)
        t = torch.randint(0, self.nt, (batch_size,), device=x.device).long()
        return self.estimate_loss_on_noisy_distribution(xstart=x, t=t, **kwargs)

    def forward(self, cond, force, *args, **kwargs):
        return self.conditional_sample_from_noisy_distiribution(cond, force, *args, **kwargs)