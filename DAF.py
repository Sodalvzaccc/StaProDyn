import copy
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange


class DynamicLayer(nn.Module):
    def __init__(self, input_dim, output_dim, max_depth):
        super(DynamicLayer, self).__init__()
        self.max_depth = max_depth
        self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else output_dim, output_dim) for i in range(max_depth)])
        self.gates = nn.ModuleList([nn.Linear(output_dim, 1) for _ in range(max_depth)])

    def forward(self, x, depth=0):
        if depth >= self.max_depth:
            return x
        x = F.relu(self.layers[depth](x))
        gate_status = torch.sigmoid(self.gates[depth](x)).mean()
        if gate_status > 0.5:
            return self.forward(x, depth + 1)
        else:
            return x


class BiLSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first=True, dropout_rate=0.5):
        super(BiLSTMModule, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout_input = nn.Dropout(p=0.0)
        self.bilstm = nn.LSTM(input_size=input_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              bidirectional=True)
        self.dropout_output = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        output, (hn, cn) = self.bilstm(x)
        output = self.dropout_output(output)
        return output


class CPC(nn.Module):
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)

    def forward(self, x, y):
        x = torch.mean(x, dim=-2)
        y = torch.mean(y, dim=-2)
        x_pred = y

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        nce = -(pos - neg).mean()
        return nce


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)


class MultiHAtten(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(MultiHAtten, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class CrossTransformer(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super(CrossTransformer, self).__init__()
        self.cross_attn = MultiHAtten(dim, heads=8, dim_head=64, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, source_x, target_x):
        target_x_tmp = self.cross_attn(target_x, source_x, source_x)
        target_x = self.layernorm1(target_x_tmp + target_x)
        target_x = self.layernorm2(self.ffn(target_x) + target_x)
        return target_x


class DynamicAttentionBlock(nn.Module):
    def __init__(self, opt, dropout):
        super(DynamicAttentionBlock, self).__init__()
        self.f_t = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)
        self.f_v = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)
        self.f_a = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)

        self.layernorm_t = nn.LayerNorm(256)
        self.layernorm_v = nn.LayerNorm(256)
        self.layernorm_a = nn.LayerNorm(256)
        self.layernorm = nn.LayerNorm(256)

    def forward(self, source, t, v, a, senti):
        cross_f_t = self.f_t(target_x=source, source_x=t)
        cross_f_v = self.f_v(target_x=source, source_x=v)
        cross_f_a = self.f_a(target_x=source, source_x=a)

        if senti is not None:
            output = self.layernorm(self.layernorm_t(cross_f_t + senti['T'] * cross_f_t) + self.layernorm_v(cross_f_v + senti['V'] * cross_f_v) + self.layernorm_a(cross_f_a + senti['A'] * cross_f_a))
        else:
            output = self.layernorm(cross_f_t + cross_f_v + cross_f_a)
        return output


class DynamicAttentionAllocationBlock(nn.Module):
    def __init__(self, opt):
        super(DynamicAttentionAllocationBlock, self).__init__()
        self.mhatt1 = DynamicAttentionBlock(opt, dropout=0.3)
        self.mhatt2 = MultiHAtten(opt.hidden_size, dropout=0.)
        self.ffn = FeedForward(opt.hidden_size, opt.ffn_size, dropout=0.)

        self.norm1 = nn.LayerNorm(opt.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(opt.hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(opt.hidden_size, eps=1e-6)

    def forward(self, source, t, v, a, mask, senti):
        source = self.norm1(source + self.mhatt1(source, t, v, a, senti=senti))
        source = self.norm2(source + self.mhatt2(q=source, k=source, v=source))
        source = self.norm3(source + self.ffn(source))
        return source


class DynamicAttentionAllocation(nn.Module):
    def __init__(self, opt):
        super(DynamicAttentionAllocation, self).__init__()
        self.opt = opt

        # Length Align
        self.len_t = nn.Linear(opt.seq_lens[0], opt.seq_lens[0])
        self.len_v = nn.Linear(opt.seq_lens[1], opt.seq_lens[0])
        self.len_a = nn.Linear(opt.seq_lens[2], opt.seq_lens[0])

        # Dimension Align
        self.dim_t = nn.Linear(768*2, 256)
        self.dim_v = nn.Linear(256, 256)
        self.dim_a = nn.Linear(256, 256)
        text_feat_dim = 256
        audio_feat_dim = 256
        video_feat_dim = 256



        fusion_block = DynamicAttentionAllocationBlock(opt)
        self.dec_list = self._get_clones(fusion_block, 3)

        self.visual_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=video_feat_dim + text_feat_dim, nhead=8, dim_feedforward=256),
            num_layers=6
        )
        self.acoustic_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=audio_feat_dim + text_feat_dim, nhead=8, dim_feedforward=256),
            num_layers=6
        )

        self.visual_dyn = DynamicLayer(video_feat_dim + text_feat_dim, text_feat_dim, max_depth=opt.max_depth)
        self.acoustic_dyn = DynamicLayer(audio_feat_dim + text_feat_dim, text_feat_dim, max_depth=opt.max_depth)

        self.visual_reshape = nn.Linear(video_feat_dim, text_feat_dim)
        self.acoustic_reshape = nn.Linear(audio_feat_dim, text_feat_dim)

        self.attn_v = nn.Sequential(
            BiLSTMModule(input_dim=text_feat_dim, hidden_dim=video_feat_dim // 2, num_layers=1, dropout_rate=0.5),
            nn.Linear(video_feat_dim, 1)
        )
        self.attn_a = nn.Sequential(
            BiLSTMModule(input_dim=text_feat_dim, hidden_dim=audio_feat_dim // 2, num_layers=1, dropout_rate=0.5),
            nn.Linear(audio_feat_dim, 1)
        )

        self.LayerNorm = nn.LayerNorm(opt.hidden_size)
        self.dropout = nn.Dropout(opt.output_droupout_prob)

        self.prelu_weight_v = nn.Parameter(torch.tensor(0.25))
        self.prelu_weight_a = nn.Parameter(torch.tensor(0.25))


        self.cpc_ft = CPC(x_size=256, y_size=256)
        self.cpc_fv = CPC(x_size=256, y_size=256)
        self.cpc_fa = CPC(x_size=256, y_size=256)

    def forward(self, uni_fea, uni_mask, senti_ratio):
        eps = 1e-6

        hidden_t = self.len_t(self.dim_t(uni_fea['T']).permute(0, 2, 1)).permute(0, 2, 1)
        hidden_v = self.len_v(self.dim_v(uni_fea['V']).permute(0, 2, 1)).permute(0, 2, 1)
        hidden_a = self.len_a(self.dim_a(uni_fea['A']).permute(0, 2, 1)).permute(0, 2, 1)

        visual_text_pair = self.visual_attn(torch.cat((hidden_v, hidden_t), dim=-1))
        acoustic_text_pair = self.acoustic_attn(torch.cat((hidden_a, hidden_t), dim=-1))

        weight_v = F.prelu(self.visual_dyn(visual_text_pair), self.prelu_weight_v)
        weight_a = F.prelu(self.acoustic_dyn(acoustic_text_pair), self.prelu_weight_a)

        visual_transformed = self.visual_reshape(hidden_v)
        acoustic_transformed = self.acoustic_reshape(hidden_a)

        # Compute intermediate modality-specific features
        weighted_v = weight_v * visual_transformed
        weighted_a = weight_a * acoustic_transformed

        attn_scores_v = torch.sigmoid(self.attn_v(weighted_v))
        attn_scores_a = torch.sigmoid(self.attn_a(weighted_a))

        # Normalize attention scores across modalities
        total_attn = attn_scores_v + attn_scores_a + eps
        attn_scores_v = attn_scores_v / total_attn
        attn_scores_a = attn_scores_a / total_attn

        weighted_v = attn_scores_v * weighted_v
        weighted_a = attn_scores_a * weighted_a

        fusion = weighted_v + weighted_a + hidden_t

        # Normalize and apply dropout
        source = self.dropout(self.LayerNorm(fusion))

        for i, dec in enumerate(self.dec_list):
            source = dec(source, hidden_t, hidden_v, hidden_a, uni_mask, senti_ratio)

        nce_t = self.cpc_ft(hidden_t, source)
        nce_v = self.cpc_fv(hidden_v, source)
        nce_a = self.cpc_fa(hidden_a, source)
        nce_loss = nce_t + nce_v + nce_a

        return source, nce_loss

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SentiCLS(nn.Module):
    def __init__(self, opt):
        super(SentiCLS, self).__init__()
        self.cls_layer = nn.Sequential(
            nn.Linear(256, 64, bias=True),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32, bias=True),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, fusion_features):
        fusion_features = torch.mean(fusion_features, dim=-2)
        output = self.cls_layer(fusion_features)
        return output
