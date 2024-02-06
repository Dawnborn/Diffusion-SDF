import math
import torch
import torch.nn.functional as F
from torch import nn, einsum 

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape

from rotary_embedding_torch import RotaryEmbedding

from diff_utils.model_utils import * 

from random import sample

class CausalTransformer(nn.Module): #junpeng: CausalTransformer
    def __init__(
        self,
        dim, 
        depth,
        dim_in_out=None,
        cross_attn=False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True, 
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True, 
        normformer = False,
        rotary_emb = True, 
        **kwargs
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None
        rotary_emb_cross = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])

        dim_in_out = default(dim_in_out, dim)
        self.use_same_dims = (dim_in_out is None) or (dim_in_out==dim)
        point_feature_dim = kwargs.get('point_feature_dim', dim)

        if cross_attn:
            #print("using CROSS ATTN, with dropout {}".format(attn_dropout))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, out_dim=dim_in_out, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
        else:
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, out_dim=dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim, out_dim=dim_in_out, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))

        self.norm = LayerNorm(dim_in_out, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim_in_out, dim_in_out, bias = False) if final_proj else nn.Identity()

        self.cross_attn = cross_attn

    def forward(self, x, time_emb=None, context=None):
        """

        :param x: B [time_embed, data, learned_queries]
        :param time_emb:
        :param context:
        :return:
        """
        # import pdb; pdb.set_trace()
        n, device = x.shape[1], x.device

        x = self.init_norm(x) # B,3,latent_code_dim ->

        attn_bias = self.rel_pos_bias(n, n + 1, device = device) # [8, 3, 4]

        if self.cross_attn:
            #assert context is not None 
            for idx, (self_attn, cross_attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = self_attn(x, attn_bias = attn_bias)
                    x = cross_attn(x, context=context) # removing attn_bias for now 
                else:
                    x = self_attn(x, attn_bias = attn_bias) + x 
                    x = cross_attn(x, context=context) + x  # removing attn_bias for now 
                #print("x2 shape, context shape: ", x.shape, context.shape)
                
                #print("x3 shape, context shape: ", x.shape, context.shape)
                x = ff(x) + x
        
        else:
            for idx, (attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = attn(x, attn_bias = attn_bias) # B,3,latent code dim
                else:
                    x = attn(x, attn_bias = attn_bias) + x
                #print("x2 shape: ", x.shape)
                x = ff(x) + x
                #print("x3 shape: ", x.shape)

        out = self.norm(x) # B,3,dim_latent_code
        out = self.project_out(out)
        return out

class DiffusionNet(nn.Module): #junpeng: Diffusion 模型

    def __init__(
        self,
        dim,
        dim_in_out=None,
        num_timesteps = None,
        num_time_embeds = 1,
        cond = None,
        **kwargs
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.dim = dim
        self.cond = cond
        self.cross_attn = kwargs.get('cross_attn', False)
        self.cond_dropout = kwargs.get('cond_dropout', False)
        self.point_feature_dim = kwargs.get('point_feature_dim', dim)

        self.dim_in_out = default(dim_in_out, dim)
        #print("dim, in out, point feature dim: ", dim, dim_in_out, self.point_feature_dim)
        #print("cond dropout: ", self.cond_dropout)

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out), MLP(self.dim_in_out, self.dim_in_out * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # last input to the transformer: "a final embedding whose output from the Transformer is used to predicted the unnoised CLIP image embedding"
        self.learned_query = nn.Parameter(torch.randn(self.dim_in_out))
        self.causal_transformer = CausalTransformer(dim = dim, dim_in_out=self.dim_in_out, **kwargs)

        if cond:
            # output dim of pointnet needs to match model dim; unless add additional linear layer
            self.pointnet = ConvPointnet(c_dim=self.point_feature_dim) 


    def forward(
        self,
        data, # B,latent code dim
        diffusion_timesteps, # [B]
        pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling
        no_cond=False
    ):

        if self.cond:
            if no_cond:
                data, cond = data
            else:
                assert type(data) is tuple
                data, cond = data # adding noise to cond_feature so doing this in diffusion.py

            print("data, cond shape: ", data.shape, cond.shape) # B, dim_in_out; B, N, 3
            print("pass cond: ", pass_cond)
            if self.cond_dropout:
                # classifier-free guidance: 20% unconditional # 80% unconditional
                prob = torch.randint(low=0, high=10, size=(1,))
                percentage = 8
                if prob < percentage or pass_cond==0:
                    cond_feature = torch.zeros((cond.shape[0], cond.shape[1], self.point_feature_dim), device=data.device)
                    #print("zeros shape: ", cond_feature.shape)
                    print("cond feature shape: ", cond_feature.shape)
                elif prob >= percentage or pass_cond==1:
                    cond_feature = self.pointnet(cond, cond) # [B, N_ptc, 128]
                    # np.save("/storage/user/lhao/hjp/ws_dditnach/Diffusion-SDF/repro_scanarcw.npy",cond_feature.cpu().numpy())
                    print("cond feature shape: ", cond_feature.shape)
                    # import pdb; pdb.set_trace()

            else:
                cond_feature = self.pointnet(cond, cond)
                print("cond feature shape: ", cond_feature.shape) #


        # import pdb; pdb.set_trace()
        batch, dim, device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds(diffusion_timesteps) # [B, num_time_embeds, latent code dim]，由embeddings表示，学习优化得到

        data = data.unsqueeze(1) # [B, 1, latent code dim]

        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch) # [B, 1, latent code dim], self.learned_query 256 需要学习优化得到 [-3,3] var 0.9

        model_inputs = [time_embed, data, learned_queries] # 三部分: 可学习的time embeddings, 原始数据，可学习的queries ([1, 1, 256]) ([1, 1, 256]) ([1, 1, 256])

        if self.cond and not self.cross_attn:
            model_inputs.insert(0, cond_feature) # cond_feature defined in first loop above
        
        tokens = torch.cat(model_inputs, dim = 1) # (b, 3 or 4, d); batch and d=512 same across the model_inputs 
        print("tokens shape: ", tokens.shape) # [1 3 256]

        if self.cross_attn:
            cond_feature = None if not self.cond else cond_feature
            #print("tokens shape: ", tokens.shape, cond_feature.shape)  # [1 3 256], [1,N_ptc,128]
            tokens = self.causal_transformer(tokens, context=cond_feature)
        else:
            tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        pred = tokens[..., -1, :]

        return pred

