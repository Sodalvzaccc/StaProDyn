import torch
from models.Encoder_KIAdapter import UnimodalEncoder
from models.DAF import DynamicAttentionAllocation, SentiCLS
from core.utils import calculate_ratio_senti
from models.SAF import SAFModule
from models.WME import PromptLearner

import torch.nn.functional as F
from torch import nn
from core.transformer import TransformerEncoder


class StaProDyn(nn.Module):
    def __init__(self, opt, dataset, bert_pretrained='bert-base-uncased'):
        super(KMSA, self).__init__()
        # Unimodal Encoder & Knowledge Inject Adapter
        self.UniEncKI = UnimodalEncoder(opt, bert_pretrained)

        # Multimodal Fusion
        self.DAF = DynamicAttentionAllocation(opt)
        self.d_t = 256
        self.d_a = 256
        self.d_v = 256

        # Output Classification for Sentiment Analysis
        self.CLS = SentiCLS(opt)
        self.orig_d_t = 1536
        self.orig_d_a = 256
        self.orig_d_v = 256
        self.layers = opt.nlevels
        self.prompt_dim = opt.prompt_dim
        self.prompt_length = opt.prompt_length
        self.proj_dim = opt.proj_dim

        # SAF module
        self.saf = SAFModule(opt)

        self.num_heads = opt.num_heads
        self.attn_dropout = opt.attn_dropout
        self.attn_dropout_a = opt.attn_dropout_a
        self.attn_dropout_v = opt.attn_dropout_v
        self.relu_dropout = opt.relu_dropout
        self.res_dropout = opt.res_dropout
        self.out_dropout = opt.out_dropout
        self.embed_dropout = opt.embed_dropout
        self.attn_mask = opt.attn_mask

        self.llen, self.vlen, self.alen = opt.seq_lens

        combined_dim = self.d_l + self.d_a + self.d_v
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        # Prompt learner module
        self.prompt = PromptLearner(
            prompt_dim=self.prompt_dim,
            prompt_length=self.prompt_length,
            llen=self.llen,
            alen=self.alen,
            vlen=self.vlen,
            orig_d_t=self.orig_d_t,
            orig_d_a=self.orig_d_a,
            orig_d_v=self.orig_d_v,
            d_l=self.d_l,
            d_a=self.d_a,
            d_v=self.d_v,
        )

        # 2. Crossmodal Attentions
        self.trans_t_with_a = self.get_network(self_type="la")
        self.trans_t_with_v = self.get_network(self_type="lv")

        self.trans_a_with_t = self.get_network(self_type="al")
        self.trans_a_with_v = self.get_network(self_type="av")

        self.trans_v_with_t = self.get_network(self_type="vl")
        self.trans_v_with_a = self.get_network(self_type="va")

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_t_mem = self.get_network(self_type="l_mem", layers=3)
        self.trans_a_mem = self.get_network(self_type="a_mem", layers=3)
        self.trans_v_mem = self.get_network(self_type="v_mem", layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)

        self.xx_t_linear = nn.Linear(256, 1536)
        self.xx_t_norm = nn.LayerNorm(1536)

    def get_network(self, self_type="t", layers=-1):
        config_map = {
            "t": (self.d_t, self.attn_dropout),
            "a": (self.d_a, self.attn_dropout_a),
            "v": (self.d_v, self.attn_dropout_v),
            "t_mem": (2 * self.d_t, self.attn_dropout),
            "a_mem": (2 * self.d_a, self.attn_dropout),
            "v_mem": (2 * self.d_v, self.attn_dropout),
        }


        config = config_map.get(self_type)

        if config is None:
            # 使用 f-string 改进了错误消息
            raise ValueError(f"Unknown network type: {self_type}")

        embed_dim, attn_dropout = config

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def get_complete_data(self, x_t, x_a, x_v, weak_mode):
        return self.prompt.get_complete_data(x_t, x_a, x_v, weak_mode)


    def forward(self, inputs_data_mask, multi_senti, weak_mode, epoch: int = None, opt: object = None):
        uni_fea, uni_senti = self.UniEncKI(inputs_data_mask)  # [T, V, A]
        uni_mask = inputs_data_mask['mask']

        if (epoch is not None) and (opt is not None) and self.training:
           weak_mode = self.saf.select_weak(uni_senti, weak_mode, epoch, opt, training=True)

        xx_t, xx_a, xx_v = None, None, None
        for idx in range(len(x_l)):
            x_l_temp, x_a_temp, x_v_temp = self.get_complete_data(
                x_l[idx], x_a[idx], x_v[idx], weak_mode[idx]
            )
            if xx_l is None:
                xx_l = x_l_temp
                xx_a = x_a_temp
                xx_v = x_v_temp
            else:
                xx_l = torch.cat([xx_l, x_l_temp], dim=0)
                xx_a = torch.cat([xx_a, x_a_temp], dim=0)
                xx_v = torch.cat([xx_v, x_v_temp], dim=0)

        proj_x_a = xx_a.permute(0, 2, 1)
        proj_x_v = xx_v.permute(0, 2, 1)
        proj_x_t = xx_t.permute(0, 2, 1)

        proj_x_t = self.xx_l_linear(proj_x_t)
        proj_x_t = self.xx_l_norm(proj_x_t)

        mask_t = (weak == 0)
        mask_a = (weak == 1)
        mask_v = (weak == 2)
        if mask_t.any():
            uni_fea['T'][mask_t] = proj_x_t[mask_t]
        if mask_a.any():
            uni_fea['A'][mask_a] = proj_x_a[mask_a]
        if mask_v.any():
            uni_fea['V'][mask_v] = proj_x_v[mask_v]


        if multi_senti is not None:
            senti_ratio = calculate_ratio_senti(uni_senti, multi_senti, k=0.1)
        else:
            senti_ratio = None

        multimodal_features, nce_loss = self.DAF(uni_fea, uni_mask, senti_ratio)

        # Sentiment Classification
        prediction = self.CLS(multimodal_features)

        return prediction, nce_loss

    def preprocess_model(self, pretrain_path):
        # 加载预训练模型
        ckpt_t = torch.load(pretrain_path['T'])
        self.UniEncKI.enc_t.load_state_dict(ckpt_t)
        ckpt_v = torch.load(pretrain_path['V'])
        self.UniEncKI.enc_v.load_state_dict(ckpt_v)
        ckpt_a = torch.load(pretrain_path['A'])
        self.UniEncKI.enc_a.load_state_dict(ckpt_a)
        # 冻结外部知识注入参数
        for name, parameter in self.UniEncKI.named_parameters():
            if 'adapter' in name or 'decoder' in name:
                parameter.requires_grad = False


def build_model(opt):
    if 'sims' in opt.datasetName:
        l_pretrained = './pretrainedModel/BERT/bert-base-chinese'
    else:
        l_pretrained = './pretrainedModel/BERT/bert-base-uncased'

    model = StaProDyn(opt, dataset=opt.datasetName, bert_pretrained=l_pretrained)

    return model


class MLPLayer(nn.Module):
    def __init__(self, dim, embed_dim, is_Fusion=False):
        super().__init__()
        if is_Fusion:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        else:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))
