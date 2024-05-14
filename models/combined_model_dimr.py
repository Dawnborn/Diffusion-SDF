import pdb

import torch
import torch.utils.data
from torch.nn import functional as F
import pytorch_lightning as pl

# add paths in model/__init__.py for new models
from models import *
from models.archs.DDITmodel import DDIT_model
from models.archs.deep_implicit_template_decoder import Decoder, load_SDF_model_from_specs, load_SDF_specs, remove_weight_norm, calc_and_fix_weights

import time

import deep_sdf
import deep_sdf.workspace as ws

from lib.pointgroup_ops.functions import pointgroup_ops
import spconv
from spconv.modules import SparseModule
from models.archs.dimr_model import DIMR_model

loss_l1 = torch.nn.L1Loss(reduction="mean")

def huber_loss(error, delta=1.0):
    """
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

class CombinedModel(pl.LightningModule):
    def __init__(self, specs, dataloader=None):
        super().__init__()
        self.specs = specs
        self.dataloader = dataloader

        self.task = specs['training_task']  # 'combined' or 'modulation' or 'diffusion'

        if self.task in ('combined', 'modulation'):
            self.sdf_model = SdfModel(specs=specs)

            feature_dim = specs["SdfModelSpecs"]["latent_dim"]  # latent dim of pointnet
            modulation_dim = feature_dim * 3  # latent dim of modulation
            latent_std = specs.get("latent_std", 0.25)  # std of target gaussian distribution of latent space
            hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
            self.vae_model = BetaVAE(in_channels=feature_dim * 3, latent_dim=modulation_dim, hidden_dims=hidden_dims,
                                     kl_std=latent_std)

        if self.task in ('combined', 'diffusion', 'combined2'):
            self.diffusion_model = DiffusionModel(model=DiffusionNet(**specs["diffusion_model_specs"]),
                                                  **specs["diffusion_specs"])
        
        if self.task in ('combined_ddit'):
            self.ddit_model = DDIT_model(specs["ddit_specs"])
            
            decoder_specs, experiment_directory = load_SDF_specs(specs["ddit_specs"]["wanted_category"][0], specs["sdf_model"])
            self.decoder = load_SDF_model_from_specs(decoder_specs, experiment_directory)
            # self.decoder = Decoder(decoder_specs["CodeLength"], **decoder_specs["NetworkSpecs"])

            test_input = torch.rand(256,3)
            with torch.no_grad():
                test_output = self.decoder.sdf_decoder(test_input)


            remove_weight_norm(self.decoder.sdf_decoder)
            calc_and_fix_weights(self.decoder.sdf_decoder)

            with torch.no_grad():
                test_output2 = self.decoder.sdf_decoder(test_input)
                r = test_output-test_output2

            # self.decoder.sdf_decoder.eval()
            for param in self.decoder.sdf_decoder.parameters():
                param.requires_grad = False
        
        if self.task in ('combined_dimr'):
            self.dimr_model = DIMR_model()


    def training_step(self, x, idx):

        if self.task == 'combined':
            return self.train_combined(x)
        elif self.task == 'modulation':
            return self.train_modulation(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
        elif self.task == 'combined_ddit':
            return self.train_combined_ddit(x)
        elif self.task == 'combined_dimr':
            return self.train_combined_dimr(x)

    def configure_optimizers(self):

        if self.task == 'combined':
            params_list = [
                {'params': list(self.sdf_model.parameters()) + list(self.vae_model.parameters()),
                 'lr': self.specs['sdf_lr']},
                {'params': self.diffusion_model.parameters(), 'lr': self.specs['diff_lr']}
            ]
        elif self.task == 'modulation':
            params_list = [
                {'params': self.parameters(), 'lr': self.specs['sdf_lr']}
            ]
        elif self.task == 'diffusion':
            params_list = [
                {'params': self.parameters(), 'lr': self.specs['diff_lr']}
            ]
        elif self.task == 'combined_ddit':
            params_list = [
                {'params': list(self.ddit_model.parameters()) + list(self.decoder.warper.parameters()),
                 'lr': self.specs['sdf_lr']}
            ]
            # params_list = [
            #     {'params': self.ddit_model.parameters(),
            #      'lr': self.specs['sdf_lr']}
            # ]
        elif self.task == "combined_dimr":
            params_list = [
                {'params':self.parameters(), 'lr':self.specs['lr']}
            ]
        optimizer = torch.optim.Adam(params_list)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50000, threshold=0.0002, min_lr=1e-6, verbose=False),
            # "monitor": "total"
            # }
        }

    # -----------different training steps for sdf modulation, diffusion, combined----------

    def train_combined_dimr(self, x):
        self.train

        inner_ptc = x['point_cloud'].detach()
        
        B,N,C = inner_ptc.shape

        locs = []
        locs_float = []

        for idx_batch, xyz_middle in enumerate(inner_ptc):
            xyz = xyz_middle*50
            xyz -= xyz.min(0)
            batch_idxs = torch.LongTensor(xyz.shape[0], 1).fill_(idx_batch)
            locs.append(torch.cat([batch_idxs, xyz.long()]))
            locs_float.append(xyz_middle)

        locs = torch.cat(locs,0)
        locs_float = torch.cat(locs_float, 0)

        self.full_scale = [128, 512]
        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)

        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, B, mode=4)

        feats = locs_float
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, mode=4)  # (M, C), float, cuda # cfg.mode 4=mean

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_locs.int(), spatial_shape, B)

        model_inp = {
            "input": input_,
            "input_map": p2v_map,
            "coords": locs_float,
            "batch_idxs": locs[:,0].int(), # id mark of concatenated pointcloud in the batch
            "B": B
        }

        ret  = self.dimr_model.forward(model_inp)
        pred_zs = ret['proposal_zs']

        gt_latent = x["latent"].detach()

        loss = huber_loss(pred_zs - gt_latent).mean(dim=1)

        loss_dict = {"loss": loss.detach()}

        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss

    def train_combined_ddit(self, x):
        self.train()

        pcd = x['point_cloud'].detach()  # (B, 1024, 3) or False if unconditional

        neighbor_pcd = x['neighbor_pcds'].detach()
        B, N_neighbor, N_pcd, _ = neighbor_pcd.shape
        neighbor_pcd = neighbor_pcd.view(B, N_neighbor*N_pcd, -1)
        
        latent = x['latent'].detach()  # (B, D)

        gt_sdf_xyzv = x['gt_sdf_xyzv'].detach()  # (B, N，4)

        xyz = gt_sdf_xyzv[:,:,:3]
        gt_sdf = gt_sdf_xyzv[:,:,3]

        deform_code, categ_prediction = self.ddit_model.forward(pcd, neighbor_pcd) # B * 256

        # latent_loss = loss_l1(deform_code, latent)
        # loss = latent_loss

        deform_code = deform_code.repeat_interleave(self.specs["sdf_samples"], dim=0) # B*sdf_samples, 256
        xyz = xyz.view(xyz.shape[0]*xyz.shape[1], 3)# B*sdf_samples, 3
        
        decoder_input = torch.cat([deform_code, xyz], dim = 1)
        pred_sdf = self.decoder.forward(decoder_input, output_warped_points=True, output_warping_param=True) # p_final, x[40000,1], warping_param
        gt_sdf = gt_sdf.view(gt_sdf.shape[0]*gt_sdf.shape[1], -1).detach()
        
        train_loss_l1 = loss_l1(pred_sdf, gt_sdf)
        loss = train_loss_l1

        # sdf_loss = F.l1_loss(pred_sdf_list[-1].squeeze(), gt_sdf, reduction='none') # 40000 40000
        # sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()
        # print("sdf_loss shape:{}".format(sdf_loss.shape))
        # loss = sdf_loss

        loss_dict = {"loss": loss.detach()}

        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss

    def train_modulation(self, x):

        xyz = x['xyz']  # (B, N:16000, 3)
        gt = x['gt_sdf']  # (B, N:16000)
        pc = x['point_cloud']  # (B, 1024, 3)

        # STEP 1: obtain reconstructed plane feature and latent code 
        plane_features = self.sdf_model.pointnet.get_plane_features(pc)   # ([16, 256, 64, 64],[16, 256, 64, 64],[16, 256, 64, 64])
        original_features = torch.cat(plane_features, dim=1)  # [16,256*3,64,64]
        out = self.vae_model(original_features)  # ([16, 768, 64, 64],[16, 768, 64, 64],[16, 768],[16, 768],[16, 768]) out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]

        # STEP 2: pass recon back to GenSDF pipeline
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        if torch.isnan(pred_sdf).any():
            print("NaN detected! Launching debugger.")
            # pdb.set_trace()
        # pdb.set_trace()
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"])
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None  # skips this batch

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')  # ([16,16000], [16,16000]) -> [16,16000]
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()  # [16,16000] -> [1]

        loss = sdf_loss + vae_loss

        loss_dict = {"sdf": sdf_loss, "vae": vae_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss

    def train_diffusion(self, x):

        self.train()

        pc = x['point_cloud']  # (B, 1024, 3) or False if unconditional
        latent = x['latent']  # (B, D)

        # unconditional training if cond is None 
        if self.specs['diffusion_model_specs']['cond']:
            cond = pc
            if self.specs['use_neighbor']:
                neighbor_pcds = x['neighbor_pcds']
                cond = (pc, neighbor_pcds) # ([10, 6000, 3], [10, 3, 6000, 3])
        else:
            cond = None            

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)  # latent: 64,256 B,D

        loss_dict = {
            "total": diff_loss,
            "diff100": diff_100_loss,
            # note that this can appear as nan when the training batch does not have sampled timesteps < 100
            "diff1000": diff_1000_loss
        }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss

    # the first half is the same as "train_sdf_modulation" the reconstructed latent is used as input to the diffusion
    # model, rather than loading latents from the dataloader as in "train_diffusion"
    def train_combined(self, x):
        xyz = x['gt_sdf_xyzv'][:,:3]  # (B, N, 3)
        gt = x['gt_sdf_xyzv'][:,3]  # (B, N)
        pc = x['point_cloud']  # (B, 1024, 3)

        # STEP 1: obtain reconstructed plane feature for SDF and latent code for diffusion
        plane_features = self.sdf_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        # print("plane feat shape: ", feat.shape)
        out = self.vae_model(original_features)  # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]  # [B, D*3, resolution, resolution], [B, D*3]

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)

        # STEP 3: losses for VAE and SDF 
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"])
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None  # skips this batch
        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        # STEP 4: use latent as input to diffusion model
        cond = pc if self.specs['diffusion_model_specs']['cond'] else None
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(
            latent, cond=cond)

        # STEP 5: use predicted / reconstructed latent to run SDF loss 
        generated_plane_feature = self.vae_model.decode(pred_latent)
        generated_sdf_pred = self.sdf_model.forward_with_plane_features(generated_plane_feature, xyz)
        generated_sdf_loss = F.l1_loss(generated_sdf_pred.squeeze(), gt.squeeze())

        # surface weight could prioritize points closer to surface but we did not notice better results when using it 
        # surface_weight = torch.exp(-50 * torch.abs(gt))
        # generated_sdf_loss = torch.mean( F.l1_loss(generated_sdf_pred, gt, reduction='none') * surface_weight )

        # we did not experiment with using constants/weights for each loss (VAE loss is weighted using value in specs
        # file) results could potentially improve with a grid search
        loss = sdf_loss + vae_loss + diff_loss + generated_sdf_loss

        loss_dict = {
            "total": loss,
            "sdf": sdf_loss,
            "vae": vae_loss,
            "diff": diff_loss,
            # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively
            # typically diff_100 approaches 0 while diff_1000 can still be relatively high
            # visualizing loss curves can help with debugging if training is unstable
            # "diff100": diff_100_loss, # note that this can sometimes appear as nan when the training batch does not have sampled timesteps < 100
            # "diff1000": diff_1000_loss,
            "gensdf": generated_sdf_loss,
        }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss

    def generate_lat_from_pc_ddit(self, pcd, neighbor_pcd):
        self.eval()
        B, N_neighbor, N_pcd, _ = neighbor_pcd.shape
        neighbor_pcd = neighbor_pcd.view(B, N_neighbor * N_pcd, -1)

        deform_code, categ_prediction = self.ddit_model.forward(pcd, neighbor_pcd)  # B * 256

        # latent_loss = loss_l1(deform_code, latent)
        # loss = latent_loss

        # deform_code = deform_code.repeat_interleave(self.specs["sdf_samples"], dim=0)  # B*sdf_samples, 256

        return deform_code

