import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
from .Transformer import PCTransformer, PCTransformerLocal

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc

class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2

class PoinTr(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim # 384
        self.knn_layer = config.knn_layer # 1

        # self.fold_step = int(pow(self.num_pred//self.num_query, 0.5) + 0.5)
        self.base_model = PCTransformer(in_chans = 4, 
                                        embed_dim = self.trans_dim,
                                        depth = config.PCTranformer_depth, 
                                        drop_rate = 0., 
                                        knn_layer = self.knn_layer,
                                        categ_num = len(config.wanted_category))


    def forward(self, xyz):
        global_feature, categ_prediction = self.base_model(xyz) # B M C and B M 3
    
        return global_feature, categ_prediction


class PoinTrLocal(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        if type(config) is dict:
            self.trans_dim = config["trans_dim"]  # 384
            self.knn_layer = config["knn_layer"]  # 1
        else:
            self.trans_dim = config.trans_dim  # 384
            self.knn_layer = config.knn_layer  # 1

        # self.fold_step = int(pow(self.num_pred//self.num_query, 0.5) + 0.5)
        self.base_model = PCTransformerLocal(in_chans=3,
                                        embed_dim=self.trans_dim,
                                        depth=config["PCTranformer_depth"],
                                        drop_rate=0.,
                                        knn_layer=self.knn_layer,
                                        categ_num=len(config["wanted_category"]),
                                        output_channel=256)

    def forward(self, xyz):
        local_feature, categ_prediction = self.base_model.forward(xyz)  # B 128 C and B M 3 # B C:1024 N_group:128 写死的1024:output_channel

        return local_feature, categ_prediction

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, input_dim, output_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.query_projections = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
        self.key_projections = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
        self.value_projections = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
        self.output_projection = nn.Linear(num_heads * output_dim, output_dim)

    def forward(self, query, key, value):
        outputs = []
        for i in range(self.num_heads):
            query_projection = self.query_projections[i](query) # 2 * 256 -> 2 * 256
            key_projection = self.key_projections[i](key) # 2 * 5 * 256 ->  2 * 5 * 256
            value_projection = self.value_projections[i](value) # 2 * 5 * 256 ->  2 * 5 * 256
            dot_product = torch.matmul(query_projection.unsqueeze(1), key_projection.transpose(1,2)) # 2 * 256 * (2 * 256 * 4) -> 2 * 4
            attention_weights = torch.softmax(dot_product, dim=-1)
            output = torch.matmul(attention_weights, value_projection)
            outputs.append(output)
        concatenated_outputs = torch.cat(outputs, dim=-1)
        final_output = self.output_projection(concatenated_outputs)
        return final_output

class DDIT_model(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.PoinTr_ = PoinTr(config)
        self.inter_attention = MultiHeadAttention(config.head_num, 256, 256)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.selcetion_Fc = nn.Linear(512, 256)


    def forward(self, inner_ptc, inter_ptc, need_ori_output = False):
        """
        Input:
            inter point cloud
            inner point cloud

        Feature extraction and latent code generation
            with inter attention of inner global features,
            and self attention
        """
        B, N, C= inner_ptc.shape
        global_feature_inner, categ_prediction = self.PoinTr_(inner_ptc) # torch.Size([2, 6000, 4]) -> 1 * 256 
        global_feature_inter = [global_feature_inner.unsqueeze(2)] 
        
        for i in range(self.config.max_neighbour):
            global_feature_inter.append(self.PoinTr_(inter_ptc[:, i * N : (i + 1) * N, :])[0].unsqueeze(2))

        global_feature_inter = torch.cat(global_feature_inter, dim=2).transpose(1,2) # 1 * 4 * 256
 
        latent_code_mixed = self.inter_attention(global_feature_inner, global_feature_inter, global_feature_inter) # 1 * 256 
        latent_code_with_mixed = torch.cat([latent_code_mixed.squeeze(1), global_feature_inner], dim=1) # 1 * 512 
        latent_code_with_mixed = self.LeakyReLU(latent_code_with_mixed)
        latent_code = self.selcetion_Fc(latent_code_with_mixed) # 256 
        if need_ori_output:
            return latent_code, global_feature_inner
        return latent_code, categ_prediction


class DDIT_modelLocal(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.PoinTrLocal_ = PoinTrLocal(config)
        self.inter_attention = MultiHeadAttention(config["head_num"], 256, 256)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.selcetion_Fc = nn.Linear(512, 256)

    def forward(self, inner_ptc, inter_ptcs=None, need_ori_output=False):
        """
        Input:
            inter point cloud
            inner point cloud

        Feature extraction and latent code generation
            with inter attention of inner local features,
            and self attention
        """
        B, N, C = inner_ptc.shape
        local_feature_inner, categ_prediction = self.PoinTrLocal_(inner_ptc)  # torch.Size([2, 6000, 4]) -> B C:1024 N_group:128

        # if not (inter_ptcs is None):
        #     local_feature_inters = []
        #     for inter_ptc in inter_ptcs:
        #         local_feature_inter, categ_prediction_inter = self.PoinTrLocal_(inter_ptc)
        #         local_feature_inters.append(local_feature_inter)
        #     local_feature_inters
        #     local_feature_inters = torch.stack(local_feature_inters, dim=2)


        # global_feature_inter = [global_feature_inner.unsqueeze(2)]

        # for i in range(self.config.max_neighbour):
        #     global_feature_inter.append(self.PoinTr_(inter_ptc[:, i * N: (i + 1) * N, :])[0].unsqueeze(2))

        # global_feature_inter = torch.cat(global_feature_inter, dim=2).transpose(1, 2)  # 1 * 4 * 256
        local_feature_inner = local_feature_inner.permute(0,2,1) # B, N_group, l_dim
        return local_feature_inner



def main():
   model = DDIT_model()
   return None

if __name__ == "__main__":
    main()