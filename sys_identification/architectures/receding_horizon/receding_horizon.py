import torch
from torch import nn
from typing import Dict
from dataclasses import dataclass
from einops import rearrange, reduce
try:    
    from architectures.diffusers.src.diffusers.schedulers.scheduling_ddpm import DDPMScheduler
except:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler # type:ignore
try: 
    from architectures.receding_horizon.rechor_models import *
    from architectures.receding_horizon.rechor_utils import *
except:
    from Sys_Identification.architectures.receding_horizon.rechor_models import * # type:ignore
    from Sys_Identification.architectures.receding_horizon.rechor_utils import * # type:ignore

@dataclass
class configrhdf:
    seq_len_ctx: int = 200
    seq_len_new: int = 800
    n_u: int = 7
    n_y: int = 14

    timesteps: int = 100
    beta1: float = 1e-4
    beta2: float = 2e-2

    prediction_type: str = 'epsilon'
    hidden: int = 256
    multiplier: int = 3
    lobscond: bool = False
    gobscond: bool = True
    rechortype: int = 1
    controlgain_horizon : int = 1


@dataclass
class configrhtf:
    seq_len_ctx: int = 200
    seq_len_new: int = 800
    n_u: int = 7
    n_y: int = 14

    timesteps: int = 100
    beta1: float = 1e-4
    beta2: float = 2e-2

    prediction_type: str = 'epsilon'
    layers: int = 12
    heads: int = 4
    embeddings: int = 384
    causality: bool = True
    timecond: bool = True
    obscond: bool = True
    rechortype: int = 1
    controlgain_horizon : int = 1


class BaseLowdimPolicy(ModuleAttrMixin):  
    """
    LowDimensional Policy parent class, implements predict_action and set_normalizer.
    """
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            obs: B,To,Do
        return: 
            action: B,Ta,Da
        To = 3
        Ta = 4
        T = 6
        |o|o|o|
        | | |a|a|a|a|
        |o|o|
        | |a|a|a|a|a|
        | | | | |a|a|
        """
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()

class RHDiffusionUNet(BaseLowdimPolicy):
    """
    LowDimensional Unet Policy for training and inference.
    ---
    Parameters:

    """
    def __init__(self, 
            config,
            model=RecedingHorizonUnet,
            noise_scheduler=DDPMScheduler,
            pred_action_steps_only=False,
            oa_step_convention=False,
            **kwargs):
        super().__init__()

        assert not (config.lobscond and config.gobscond) # either local or global conditioning
        if pred_action_steps_only: # predict action steps -> global conditioning
            assert config.gobscond

        self.normalizer = LinearNormalizer()
        self.horizon = config.seq_len_ctx + config.seq_len_new # horizon length, 1000
        self.ctx = config.seq_len_ctx
        if config.rechortype==1 and (config.lobscond or config.gobscond):
            self.obs_dim = config.n_u + config.n_y
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = config.seq_len_ctx
        elif config.rechortype==2 and (config.lobscond or config.gobscond):
            self.obs_dim = config.n_u
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = self.horizon 
        elif config.rechortype==3 and (config.lobscond or config.gobscond):
            self.obs_dim = config.n_y
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = config.seq_len_ctx
        elif config.rechortype==4 and not(config.lobscond or config.gobscond):
            self.obs_dim = config.n_u
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = config.horizon
        elif config.rechortype==5 and (config.lobscond or config.gobscond):
            self.obs_dim = config.n_y # target dimension --- 7 | 9
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = self.horizon
        elif config.rechortype==6 and (config.lobscond or config.gobscond):
            self.obs_dim = config.n_u # kp + kv + ki dimension --- 3 | 18 | 27
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = config.controlgain_horizon
        self.action_dim = config.n_y # action dim 7
        self.obs_as_local_cond = config.lobscond # local conditioning
        self.obs_as_global_cond = config.gobscond # glocal conditioning
        self.pred_action_steps_only = pred_action_steps_only # action step prediction only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        multipliers = [2**x * config.hidden for x in range(config.multiplier)]
        self.model = model( # input (B, 1000, 14|16), output (B, 1000, 14|16)
            input_dim=self.action_dim if (self.obs_as_global_cond or self.obs_as_local_cond) else config.n_u + config.n_y,
            local_cond_dim=self.obs_dim if self.obs_as_local_cond else None,
            global_cond_dim=self.obs_dim*self.n_obs_steps if self.obs_as_global_cond else None,
            diffusion_step_embed_dim=config.hidden,
            down_dims=multipliers
        )
        self.noise_scheduler = noise_scheduler(
            num_train_timesteps=config.timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',
            clip_sample=True,
            prediction_type=config.prediction_type
        )

        self.mask_generator = LowdimMaskGenerator( # mask ovs_dim different!!!
            action_dim=self.action_dim,
            obs_dim=0 if (self.obs_as_local_cond or self.obs_as_global_cond) else self.obs_dim,
            max_n_obs_steps=self.n_obs_steps,
            max_n_act_steps=config.seq_len_ctx,
            fix_obs_steps=True,
            action_visible=False if (config.rechortype==4 or config.rechortype==3) else True
        )

        self.num_inference_steps = config.timesteps
        self.kwargs = kwargs
        
        self.normparams = {
            "range_eps" : 5e-2
        }
    
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
            ):
        """
        Receding Horizon Unet conditional sample method.
        """
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        trajectory[condition_mask] = condition_data[condition_mask]        
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predicts action/observation through receding horizon Unet pipeline.

        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = obs_dict['obs']
        nact = obs_dict['action']

        B, _, Do = nobs.shape 
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_local_cond: # local conditioning
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            cond_data[:,:self.ctx,:] = nact[:,:self.ctx,:] 
            cond_mask[:,:self.ctx,:] = True

        elif self.obs_as_global_cond: # global conditioning
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            cond_data[:,:self.ctx,:] = nact[:,:self.ctx,:]
            cond_mask[:,:self.ctx,:] = True

        else: # no conditioning
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        naction_pred = nsample[...,:Da] # action prediction
        
        action_pred = naction_pred

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    def get_normalizer(self, batch: torch.Tensor, ndims=1, mode='limits'):
        data = batch if batch is not None else {}
        self.normalizer.fit(data=data, last_n_dims=ndims, mode=mode, **self.normparams)
        return self.normalizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer)

    def set_external_normalizer(self, normalizer):
        self.normalizer = normalizer

    def compute_loss(self, batch):
        """
        Computes loss through a receding horizon Unet pipeline.
        """
        assert 'valid_mask' not in batch
        obs = batch['obs']
        action = batch['action']

        local_cond = None # init local cond
        global_cond = None # init global cond
        trajectory = action
        if self.obs_as_local_cond: # local cond, (B, T, D)
            local_cond = obs # cond = (B, Tobs, Dobs), traj | action = (B, Tact, Dact)
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond: # global cond (B, TxD)
            global_cond = obs[:,:self.n_obs_steps,:].reshape( 
                obs.shape[0], -1) # cond = (B, Tobs*Dobs), traj | action = (B , Tact, Dact)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else: # no cond, inpainting
            trajectory = torch.cat([action, obs], dim=-1) # traj = (B, T, Dobs+Dact), action = (B, Tact, Dact)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        return pred, target, loss_mask
    
class RHDiffusionTrf(BaseLowdimPolicy):
    """
    LowDimensional Transformer Policy for training and inference.
    ---
    Parameters:
    
    """
    def __init__(self, 
            config,
            model=RecedingHorizonTrf,
            noise_scheduler=DDPMScheduler,
            pred_action_steps_only=False,
            **kwargs):
        super().__init__()

        if pred_action_steps_only:
            assert config.obscond

        self.normalizer = LinearNormalizer()
        self.horizon = config.seq_len_ctx + config.seq_len_new
        self.ctx = config.seq_len_ctx
        if config.rechortype==1 and (config.obscond or config.timecond):
            self.obs_dim = config.n_u + config.n_y
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = config.seq_len_ctx
        elif config.rechortype==2 and (config.obscond or config.timecond):
            self.obs_dim = config.n_u
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = self.horizon 
        elif config.rechortype==3 and (config.obscond or config.timecond):
            self.obs_dim = config.n_y
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = config.seq_len_ctx
        elif config.rechortype==4 and not(config.obscond or config.timecond):
            self.obs_dim = config.n_u
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = config.horizon
        elif config.rechortype==5 and (config.obscond or config.timecond):
            self.obs_dim = config.n_y # target dimension --- 7 | 9
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = config.horizon
        elif config.rechortype==6 and (config.obscond or config.timecond):
            self.obs_dim = config.n_u # kp + kv + ki dimension --- 3 | 18 | 27
            self.n_action_steps = config.seq_len_new
            self.n_obs_steps = config.controlgain_horizon
        self.action_dim = config.n_y # action dim 7
        self.obs_as_cond = config.obscond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        self.model = model(
            input_dim=(self.action_dim+self.obs_dim) if config.rechortype==4 else self.action_dim, 
            output_dim=(self.action_dim+self.obs_dim) if config.rechortype==4 else self.action_dim, 
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            cond_dim=self.obs_dim if (config.obscond) else 0,
            n_layer=config.layers,
            n_head=config.heads,
            n_emb=config.embeddings,
            causal_attn=config.causality,
            time_as_cond=config.timecond,
            obs_as_cond=config.obscond,
            n_cond_layers=4
        )

        self.noise_scheduler = noise_scheduler(
            num_train_timesteps=config.timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',
            clip_sample=True,
            prediction_type=config.prediction_type
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if (config.obscond) else self.obs_dim,
            max_n_obs_steps=self.n_obs_steps,
            max_n_act_steps=config.seq_len_ctx,
            fix_obs_steps=True,
            action_visible=False if (config.rechortype==4 or config.rechortype==3) else True
        )

        self.num_inference_steps = config.timesteps

        self.normparams = {
            "range_eps" : 5e-2
        }

    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            **kwargs
            ):
        """
        Receding Horizon Unet conditional sample method.
        """
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, cond)
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]        
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predicts action/observation through receding horizon Unet pipeline.

        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict

        nobs = obs_dict['obs']
        nact = obs_dict['action']

        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            cond = nobs[:,:To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            cond_data[:,:self.ctx,:] = nact[:,:self.ctx,:]
            cond_mask[:,:self.ctx,:] = True

        else:
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
    
        naction_pred = nsample[...,:Da]
        
        action_pred = naction_pred

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    def get_normalizer(self, batch: torch.Tensor, ndims=1, mode='limits'):
        data = batch if batch is not None else {}
        self.normalizer.fit(data=data, last_n_dims=ndims, mode=mode, **self.normparams)
        return self.normalizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer)

    def set_external_normalizer(self, normalizer):
        self.normalizer = normalizer

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        action = batch['action']
        obs = batch['obs']

        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:,:self.n_obs_steps,:]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)
        
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        return pred, target, loss_mask