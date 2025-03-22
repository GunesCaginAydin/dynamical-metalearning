import math
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops.layers.torch import Rearrange

import pdb

class Progress:

	def __init__(self, total, name = 'Progress', ncol=3, max_length=20, indent=0, line_width=100, speed_update_freq=100):
		self.total = total
		self.name = name
		self.ncol = ncol
		self.max_length = max_length
		self.indent = indent
		self.line_width = line_width
		self._speed_update_freq = speed_update_freq

		self._step = 0
		self._prev_line = '\033[F'
		self._clear_line = ' ' * self.line_width

		self._pbar_size = self.ncol * self.max_length
		self._complete_pbar = '#' * self._pbar_size
		self._incomplete_pbar = ' ' * self._pbar_size

		self.lines = ['']
		self.fraction = '{} / {}'.format(0, self.total)

		self.resume()

		
	def update(self, description, n=1):
		self._step += n
		if self._step % self._speed_update_freq == 0:
			self._time0 = time.time()
			self._step0 = self._step
		self.set_description(description)

	def resume(self):
		self._skip_lines = 1
		print('\n', end='')
		self._time0 = time.time()
		self._step0 = self._step

	def pause(self):
		self._clear()
		self._skip_lines = 1

	def set_description(self, params=[]):

		if type(params) == dict:
			params = sorted([
					(key, val)
					for key, val in params.items()
				])
		self._clear()

		percent, fraction = self._format_percent(self._step, self.total)
		self.fraction = fraction

		speed = self._format_speed(self._step)

		num_params = len(params)
		nrow = math.ceil(num_params / self.ncol)
		params_split = self._chunk(params, self.ncol)
		params_string, lines = self._format(params_split)
		self.lines = lines

		description = '{} | {}{}'.format(percent, speed, params_string)
		print(description)
		self._skip_lines = nrow + 1

	def append_description(self, descr):
		self.lines.append(descr)

	def _clear(self):
		position = self._prev_line * self._skip_lines
		empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
		print(position, end='')
		print(empty)
		print(position, end='')
		
	def _format_percent(self, n, total):
		if total:
			percent = n / float(total)

			complete_entries = int(percent * self._pbar_size)
			incomplete_entries = self._pbar_size - complete_entries

			pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
			fraction = '{} / {}'.format(n, total)
			string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent*100))
		else:
			fraction = '{}'.format(n)
			string = '{} iterations'.format(n)
		return string, fraction

	def _format_speed(self, n):
		num_steps = n - self._step0
		t = time.time() - self._time0
		speed = num_steps / t
		string = '{:.1f} Hz'.format(speed)
		if num_steps > 0:
			self._speed = string
		return string

	def _chunk(self, l, n):
		return [l[i:i+n] for i in range(0, len(l), n)]

	def _format(self, chunks):
		lines = [self._format_chunk(chunk) for chunk in chunks]
		lines.insert(0,'')
		padding = '\n' + ' '*self.indent
		string = padding.join(lines)
		return string, lines

	def _format_chunk(self, chunk):
		line = ' | '.join([self._format_param(param) for param in chunk])
		return line

	def _format_param(self, param):
		k, v = param
		return '{} : {}'.format(k, v)[:self.max_length]

	def stamp(self):
		if self.lines != ['']:
			params = ' | '.join(self.lines)
			string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
			self._clear()
			print(string, end='\n')
			self._skip_lines = 1
		else:
			self._clear()
			self._skip_lines = 0

	def close(self):
		self.pause()

class Silent:

	def __init__(self, *args, **kwargs):
		pass

	def __getattr__(self, attr):
		return lambda *args: None
		
def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    Cosine Beta Scheduling for improved DDPM convergence and loss reduction
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)

    return torch.tensor(betas_clipped, dtype=dtype)

def conditioning(x, condition, action_dim):
    """
    Conditioning of sample/sampled data
    """
    for t, val in condition.items():
        x[:, t, action_dim:] = val.clone()
    return x

def extract_params(a, t, xshape):
    """
    Reshape parameter a into the desired xshape format with temporal variable t
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(xshape) - 1)))

def descending_sort_by_values(x, values):
    """
    Descending double sort of values applied to values and x
    """
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values

def create_timesteps(batch_size, i, device):
    """
    Timesteps for diffusion process
    """
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

