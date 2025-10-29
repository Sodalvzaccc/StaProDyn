import torch
from torch import nn


class SAFModule(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.alpha = getattr(opt, 'alpha', 1.0)
        self.beta = getattr(opt, 'beta', 1.0)

        self.register_buffer('modality_means_t', torch.tensor(0.0))
        self.register_buffer('modality_means_a', torch.tensor(0.0))
        self.register_buffer('modality_means_v', torch.tensor(0.0))

        self.register_buffer('modality_vars_t', torch.tensor(1.0))
        self.register_buffer('modality_vars_a', torch.tensor(1.0))
        self.register_buffer('modality_vars_v', torch.tensor(1.0))

    def calculate_sample_stats(self, uni_senti):
        means = {}
        stds = {}
        for m in ['T', 'V', 'A']:
            means[m] = torch.mean(uni_senti[m], dim=0)
            stds[m] = torch.std(uni_senti[m], dim=0)
        return means, stds

    def update_running_stats(self, uni_senti):
        current_mean_t = torch.mean(uni_senti['T'])
        current_var_t = torch.var(uni_senti['T'])
        self.modality_means_t = 0.9 * self.modality_means_t + 0.1 * current_mean_t
        self.modality_vars_t = 0.9 * self.modality_vars_t + 0.1 * current_var_t

        current_mean_v = torch.mean(uni_senti['V'])
        current_var_v = torch.var(uni_senti['V'])
        self.modality_means_v = 0.9 * self.modality_means_v + 0.1 * current_mean_v
        self.modality_vars_v = 0.9 * self.modality_vars_v + 0.1 * current_var_v

        current_mean_a = torch.mean(uni_senti['A'])
        current_var_a = torch.var(uni_senti['A'])
        self.modality_means_a = 0.9 * self.modality_means_a + 0.1 * current_mean_a
        self.modality_vars_a = 0.9 * self.modality_vars_a + 0.1 * current_var_a

    def select_weak(self, uni_senti, weak, epoch: int = None, opt: object = None, training: bool = True):
        if (not training) or (epoch is None) or (opt is None):
            return weak

        with torch.no_grad():
            means, stds = self.calculate_sample_stats({k: v.detach() for k, v in uni_senti.items()})
            for m in ['T', 'V', 'A']:
                stds[m] = torch.clamp(stds[m], max=getattr(opt, 'saf_std_clip', 2.0))

            score_t = means['T'] - self.alpha * stds['T'] * (1 + self.beta * self.modality_vars_t)
            score_v = means['V'] - self.alpha * stds['V'] * (1 + self.beta * self.modality_vars_v)
            score_a = means['A'] - self.alpha * stds['A'] * (1 + self.beta * self.modality_vars_a)

            scores = torch.stack([score_t, score_a, score_v], dim=0)
            saf_choice = torch.argmin(scores, dim=0)  # [B]

            sorted_vals, _ = torch.sort(scores, dim=0)
            margin = sorted_vals[1] - sorted_vals[0]
            conf_mask = (margin >= getattr(opt, 'saf_margin', 0.05))

            p_max = getattr(opt, 'saf_mix_prob_max', 0.5)
            warmup = getattr(opt, 'saf_warmup_epochs', 5)
            p_use = p_max * min(1.0, float(epoch) / max(1, warmup))
            rand_mask = (torch.rand_like(saf_choice.float()) < p_use)
            use_mask = conf_mask & rand_mask.bool()

            final_weak = torch.where(use_mask, saf_choice.to(weak.device), weak)

            if getattr(opt, 'saf_balance', False):
                bsz = final_weak.size(0)
                target_each = bsz // 3
                slack = getattr(opt, 'saf_balance_slack', 2)
                counts = [int((final_weak == k).sum().item()) for k in [0, 1, 2]]
                for k in [0, 1, 2]:
                    over = counts[k] - (target_each + slack)
                    if over > 0:
                        idx_k = (final_weak == k).nonzero(as_tuple=False).view(-1)
                        cut = min(over, idx_k.numel())
                        if cut > 0:
                            replace_idx = idx_k[:cut]
                            final_weak[replace_idx] = weak[replace_idx]

            return final_weak


