import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from collections import namedtuple
from functools import partial

# from einops import reduce
from einops.layers.torch import Rearrange

# from model.diffusion.model import *
# from diff_utils.helpers import * 
from diff_utils.helpers import linear_beta_schedule, default, extract, cosine_beta_schedule, perturb_point_cloud, \
    sample_pc

import numpy as np
import os
from statistics import mean
from tqdm.auto import tqdm
import open3d as o3d

# constants
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, input, target):
        # 计算余弦相似度
        cosine_loss = self.cosine_similarity(input, target)
        # 将余弦相似度转换为损失值
        # 由于余弦相似度范围是[-1, 1]，我们通过减去1并取绝对值来得到损失
        # 这样相似度为1时损失为0，相似度为-1时损失为2
        return torch.mean(1 - cosine_loss)
class SinL1Loss(nn.Module):
    def __init__(self):
        super(SinL1Loss, self).__init__()

    def forward(self, input, target, reduction):
        # 计算 L1 损失
        l1_loss = F.l1_loss(input, target, reduction=reduction)

        # 计算 sin 项
        print("--------------sinl1 loss")
        print("l1_loss", l1_loss.shape)
        sin_term = torch.sin(l1_loss)
        print("sin_term", sin_term.shape)

        # 将 L1 损失和 sin 项结合

        # 计算最终的平均损失
        return sin_term

class CombinedCosineL1Loss(nn.Module):
    def __init__(self):
        super(CombinedCosineL1Loss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, input, target, reduction=None, ret_cos=False):
        print("loss_function validation=======")
        print("input shape: {}".format(input.shape))
        print("target shape: {}".format(target.shape))
        # 计算余弦相似度损失
        cosine_loss = self.cosine_similarity(input, target)
        cosine_loss = 1 - cosine_loss
        print("cosine_loss shape: {}".format(cosine_loss.shape))

        # 计算L1损失
        l1_loss = F.l1_loss(input, target, reduction=reduction)
        print("l1_loss: {}".format(l1_loss.shape))
        print("l1_loss")

        # 结合两种损失
        combined_loss = cosine_loss.unsqueeze(1).expand_as(l1_loss) + l1_loss
        print("combined_loss: {}".format(combined_loss.shape))
        if ret_cos:
            return combined_loss, cosine_loss.clone().detach()
        else:
            return combined_loss

class DiffusionModel(nn.Module):
    def __init__(
            self,
            model,
            timesteps=1000, sampling_timesteps=None, beta_schedule='cosine',
            sample_pc_size=682, perturb_pc=None, crop_percent=0.25,
            loss_type='l2', objective='pred_x0',
            data_scale=1.0, data_shift=0.0,
            p2_loss_weight_gamma=0.,
            # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k=1,
            ddim_sampling_eta=1.
    ):
        super().__init__()

        self.model = model
        self.objective = objective

        betas = linear_beta_schedule(timesteps) if beta_schedule == 'linear' else cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.pc_size = sample_pc_size
        self.perturb_pc = perturb_pc
        self.crop_percent = crop_percent
        assert self.perturb_pc in [None, "partial", "noisy"]

        self.loss_type = loss_type
        # self.loss_fn2 = None
        if loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif loss_type == "l2": 
            self.loss_fn = F.mse_loss
        elif loss_type == "cos+l1":
            self.loss_fn = CombinedCosineL1Loss()
            # self.loss_fn = F.l1_loss
            # self.loss_fn2 = CosineSimilarityLoss()
        elif loss_type=="sinl1":
            self.loss_fn = SinL1Loss()
        else:
            raise NotImplementedError


        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # self.register_buffer('data_scale', torch.tensor(data_scale))
        # self.register_buffer('data_shift', torch.tensor(data_shift))

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (x0 - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def ddim_sample(self, dim, batch_size, noise=None, clip_denoised=True, traj=False, cond=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(0., total_timesteps, steps=sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        traj = []

        x_T = default(noise, torch.randn(batch, dim, device=device))

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            model_input = (x_T, cond) if cond is not None else x_T
            pred_noise, x_start, *_ = self.model_predictions(model_input, time_cond)

            if clip_denoised:
                x_start.clamp_(-1., 1.)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = torch.randn_like(x_T) if time_next > 0 else 0.

            x_T = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            traj.append(x_T.clone())

        if traj:
            return x_T, traj
        else:
            return x_T

    @torch.no_grad()
    def sample(self, dim, batch_size, noise=None, clip_denoised=True, traj=False, cond=None):

        batch, device, objective = batch_size, self.betas.device, self.objective

        traj = []

        x_T = default(noise, torch.randn(batch, dim, device=device)) # N(0,1)

        for t in reversed(range(0, self.num_timesteps)):

            time_cond = torch.full((batch,), t, device=device, dtype=torch.long)

            model_input = (x_T, cond) if cond is not None else x_T
            pred_noise, x_start, *_ = self.model_predictions(model_input, time_cond) # model_input # ([1 768],[1,128,3])
            if clip_denoised:
                x_start.clamp_(-1., 1.)

            model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x_T, t=time_cond)

            noise = torch.randn_like(x_T) if t > 0 else 0.  # no noise if t == 0

            x_T = model_mean + (0.5 * model_log_variance).exp() * noise

            traj.append(x_T.clone())

        if traj:
            return x_T, traj
        else:
            return x_T

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # "nice property": return x_t given x_0, noise, and timestep
    def q_sample(self, x_start, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        # noise = torch.clamp(noise, min=-6.0, max=6.0)

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # main function for calculating loss
    def forward(self, x_start, t, ret_pred_x=False, noise=None, cond=None):
        '''
        x_start: [B, D]
        t: [B]
        cond: pc
        '''

        noise = default(noise, lambda: torch.randn_like(x_start))  # B, latent code dim:256

        print("x_start", x_start)
        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # B, latent code dim, 加噪过程
        print("x_noised", x)
        # import pdb; pdb.set_trace()

        model_in = (x, cond) if cond is not None else x  # 128, 256
        model_out = self.model.forward(model_in, t)  # 128 256 # DiffusionNet forward
        print("x_denoised", model_out)

        if x.shape[0] == 1:
            print("at {}".format(t))
            print("x_start mean{} max{} min{}".format(x_start.mean(), x_start.max(), x_start.min()))
            print("x_noised mean{} max{} min{}".format(x.mean(), x.max(), x.min()))
            print("x_denoised mean{} max{} min{}".format(model_out.mean(), model_out.max(), model_out.min()))
            with torch.no_grad():
                e1 = torch.abs(x - x_start)
                e2 = torch.abs(model_out - x_start)
                print("{} of digits reduced error".format((e1>e2).float().mean()))
            # import pdb; pdb.set_trace()

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # pdb.set_trace()
        if self.loss_type == "cos+l1":
            loss, loss_cos = self.loss_fn(model_out, target, reduction='none') # loss_cos [B,
        else:
            loss = self.loss_fn(model_out, target, reduction='none')  # F1 loss [B, 256]
        # self.loss_fn(x, target, reduction='mean') # tensor(0.2621, device='cuda:0')

        # loss = reduce(loss, 'b ... -> b (...)', 'mean', b = x_start.shape[0]) # only one dim of latent so don't need this line
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        unreduced_loss = loss.detach().clone().mean(dim=1) # t steps [B,]

        # pdb.set_trace()
        if ret_pred_x:
            return loss.mean(), x, target, model_out, unreduced_loss # loss.mean() -> [1,]
        else:
            return loss.mean(), unreduced_loss

    def model_predictions(self, model_input, t, no_cond=False):

        # model_output1 = self.model(model_input, t, pass_cond=0)
        # model_output2 = self.model(model_input, t, pass_cond=1)
        # model_output = model_output2*5 - model_output1*4
        model_output = self.model.forward(model_input, t, pass_cond=1) # model_out B 256

        x = model_input[0] if type(model_input) is tuple else model_input

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return ModelPrediction(pred_noise, x_start)

    # a wrapper function that only takes x_start (clean modulation vector) and condition
    # does everything including sampling timestep and returns loss, loss_100, loss_1000, prediction
    def diffusion_model_from_latent(self, x_start, cond=None):
        # if self.perturb_pc is None and cond is not None:
        #    print("check whether to pass condition!!!")

        # STEP 1: sample timestep # 随机采样B个timesteps, 此处为之前得到的groundtruth latent
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()

        # STEP 2: perturb condition

        pc = perturb_point_cloud(cond, self.perturb_pc, self.pc_size, self.crop_percent) if cond is not None else None
        print("before perturb: ", cond.shape)
        print("after perturb: ", pc.shape)
        # import pdb; pdb.set_trace()

        # STEP 3: pass to forward function
        loss, x, target, model_out, unreduced_loss = self.forward(x_start, t, cond=pc, ret_pred_x=True)
        loss_100 = unreduced_loss[t < 100].mean().detach()
        loss_1000 = unreduced_loss[t > 100].mean().detach()
        import pdb
        # pdb.set_trace()
        return loss, loss_100, loss_1000, model_out, pc

    def generate_from_pc(self, pc, load_pc=False, batch=5, save_pc=False, return_pc=False, ddim=False, perturb_pc=True, no_cond=False):
        self.eval()

        with torch.no_grad():
            if load_pc:
                pc = sample_pc(pc, self.pc_size).cuda().unsqueeze(0)

            if pc is None:
                input_pc = None
                save_pc = False
                full_perturbed_pc = None

            else:
                if perturb_pc:
                    full_perturbed_pc = perturb_point_cloud(pc, self.perturb_pc)
                    perturbed_pc = full_perturbed_pc[:, torch.randperm(full_perturbed_pc.shape[1])[:self.pc_size]]
                    input_pc = perturbed_pc.repeat(batch, 1, 1)
                else:
                    full_perturbed_pc = pc
                    perturbed_pc = pc
                    input_pc = pc.repeat(batch, 1, 1)

            # print("shapes: ", pc.shape, self.pc_size, self.perturb_pc, perturbed_pc.shape, full_perturbed_pc.shape)
            # print("pc path: ", pc_path)

            # print("pc shape: ", perturbed_pc.shape, input_pc.shape)
            if save_pc:  # save perturbed pc ply file for visualization
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(perturbed_pc.cpu().numpy().squeeze())
                o3d.io.write_point_cloud("{}/input_pc.ply".format(save_pc), pcd)

            sample_fn = self.ddim_sample if ddim else self.sample
            if no_cond:
                intput_pc = torch.zeros_like(input_pc)
                samp, traj = sample_fn(dim=self.model.dim_in_out, batch_size=batch, traj=False, cond=intput_pc)
            else:
                samp, traj = sample_fn(dim=self.model.dim_in_out, batch_size=batch, traj=True, cond=input_pc)

        if return_pc:
            return samp, perturbed_pc, traj
        return samp, traj

    def generate_unconditional(self, num_samples):
        self.eval()
        with torch.no_grad():
            samp, _ = self.sample(dim=self.model.dim_in_out, batch_size=num_samples, traj=False, cond=None)

        return samp
