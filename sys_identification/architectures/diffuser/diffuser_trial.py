import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from collections import namedtuple

from diffuser_utils import *
from diffusion_models import *

Sample = namedtuple('Sample', 'Trajectory Values')

@dataclass
class Config:
    seq_len_ctx: int = 200
    seq_len_new: int = 800
    n_embd: int = 192
    n_u: int = 7
    n_y: int = 14
    dropout: float = 0.0
    bias: bool = True

    timesteps: int = 500
    beta1: float = 1e-4
    beta2: float = 2e-2

    hidden: int = 64
    channels: int = 3

class TSDiffuser(nn.Module):
    def __init__(self,
                 model,
                 horizon_dimension,
                 context_dimension,
                 action_dimension,
                 iterations,
                 clip=True,
                 predict_epsilon=True
                 ):
        super().__init__()
        self.model = model
        self.horizon_dim = horizon_dimension
        self.context_dim = context_dimension
        self.action_dim = action_dimension
        self.transition_dim = context_dimension + action_dimension

        self.iter = iterations
        self.clip_denoised = clip
        self.predict = predict_epsilon

        self.register_buffer('betas', 
                            cosine_beta_schedule(iterations))
        self.register_buffer('alphas', 
                            1 - self.betas)
        self.register_buffer('alphas_cumprod', 
                            torch.cumprod(self.alphas, axis=0))
        self.register_buffer('alphas_cumprod_old', 
                            torch.cat([torch.ones(1), self.alphas_cumprod[:-1]]))
        self.register_buffer('sqrt_alphas_cumprod', 
                            torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.-self.alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', 
                            torch.log(1.-self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', 
                            torch.sqrt(1./self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', 
                            torch.sqrt(1./self.alphas_cumprod-1))
        self.register_buffer('posterior_mean_coef1',
                            self.betas * np.sqrt(self.alphas_cumprod_old)/(1.-self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                            (1.-self.alphas_cumprod_old)*np.sqrt(self.alphas)/(1.-self.alphas_cumprod))
        self.register_buffer('posterior_variance', self.betas*(1.-self.alphas_cumprod_old)
                             /(1.-self.alphas_cumprod))
        self.register_buffer('posterior_log_variance_clipped', 
                            torch.log(torch.clamp(self.posterior_variance, min=1e-20)))
        
    def x0_from_noised_trajectory(self, t, xt, noise):
        """
        Parametrizes prior mean on data distribution xt on timestep t

        Different parametrization schemes:
            direct parametrization of mean
            parametrization of x0 - start
            parametrization of eps - noise
        """
        if self.predict:
            return (
                extract_params(self.sqrt_recip_alphas_cumprod, t, xt.shape) * xt -
                extract_params(self.sqrt_recipm1_alphas_cumprod, t, xt.shape) * noise
            )
        else:
            return noise

    def posterior_mean_and_variance(self, t, xt, xstart):
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

    def get_mean_and_variance(self, x, cond, t):
        """
        Posterior probability distribution of p at xt-1
        """
        xr = self.x0_from_noised_trajectory(xt=x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            xr.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.posterior_mean_and_variance(
                x_start=xr, x_t=x, t=t)
        
        return model_mean, posterior_variance, posterior_log_variance

    def sample_function(self, xt, cond, t):
        """
        Default sampling function to sample from a noised trajectory
        """
        model_mean, _, model_log_variance = self.get_mean_and_variance(x=xt, cond=cond, t=t)
        model_std = torch.exp(0.5 * model_log_variance)

        noise = torch.randn_like(xt)
        noise[t == 0] = 0

        values = torch.zeros(len(xt), device=xt.device)
        return model_mean + model_std * noise, values

    @torch.inference_mode()
    def sample_from_noisy_distribution(self, shape, cond, verbose=True, return_chain=False, sample_fn=sample_function, **sample_kwargs):
        """
        Sample a new distribution from a prior noised trajectory, used for testing
        """
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = create_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)
        progress.stamp()

        x, values = descending_sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.inference_mode()
    def conditional_sample_from_noisy_distiribution(self, cond, horizon=None, **sample_kwargs):
        """
        Sample a new distribution with conditionals from a prior noised trajectory, used for testing
        """
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.sample_from_noisy_distribution(shape, cond, **sample_kwargs)

    def sample_noisy_distribution(self, t, xstart, cond, noise=None):
        """
        Prior distribution of q at xt, user for training
        """
        if not noise:
            noise = torch.randn_like(xstart)

        sample = (
            extract_params(self.sqrt_alphas_cumprod, t, xstart.shape) * xstart +
            extract_params(self.sqrt_one_minus_alphas_cumprod, t, xstart.shape) * noise            
        )

        return conditioning(sample, cond, self.action_dim)
    
    def estimate_loss(self, t, xstart, noise=None, cond=None):
        """
        Estimate noise eps or starting distribution x0 from a sampled noisy distribution: to be
        used in the training loop according to selected loss functions, used for training
        """
        xnoisy = self.sample_noisy_distribution(t, xstart, noise=noise, cond=cond)
        est = self.model(xnoisy, cond, t)
        
        if self.predict_epsilon:
            loss, info = self.loss_fn(est, noise)
        else:
            loss, info = self.loss_fn(est, xstart)

        return loss, info
    
    def configure_optimizer(self, weight_decay, learning_rate, betasoptim, device_type):
        """
        Adjust optimizer parameters to be imported into main, optimizer and model are acquired in tandem
        to for user configuration during training and testing
        """
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
       
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
       
        fused_available = True 
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betasoptim, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def forward(self, cond, *args, **kwargs):
        return self.p_sample_conditional(cond, *args, **kwargs)