import torch
from torch import nn


class ConvProjector(nn.Module):
    def __init__(self, in_channels, out_channels, fused: bool = False):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.proj(inputs))


class PromptLearner(nn.Module):
    def __init__(
        self,
        prompt_dim: int,
        prompt_length: int,
        llen: int,
        alen: int,
        vlen: int,
        orig_d_l: int,
        orig_d_a: int,
        orig_d_v: int,
        d_l: int,
        d_a: int,
        d_v: int,
    ):
        super().__init__()

        self.prompt_dim = prompt_dim
        self.prompt_length = prompt_length
        self.llen = llen
        self.alen = alen
        self.vlen = vlen

        self.d_l = d_l
        self.d_a = d_a
        self.d_v = d_v

        self.orig_d_l = orig_d_l
        self.orig_d_a = orig_d_a
        self.orig_d_v = orig_d_v

        # Generative prompts (learnable)
        self.generative_prompts = nn.Parameter(
            torch.zeros(3, self.prompt_dim, self.prompt_length)
        )

        # Cross-modal projections into prompt space
        self.text_to_audio = ConvProjector(self.orig_d_l, self.prompt_dim)
        self.text_to_vision = ConvProjector(self.orig_d_l, self.prompt_dim)
        self.vision_to_audio = ConvProjector(self.orig_d_v, self.prompt_dim)
        self.vision_to_text = ConvProjector(self.orig_d_v, self.prompt_dim)
        self.audio_to_vision = ConvProjector(self.orig_d_a, self.prompt_dim)
        self.audio_to_text = ConvProjector(self.orig_d_a, self.prompt_dim)

        # Prompt fusion to sequence lengths
        self.fuse_text_with_av = ConvProjector(self.prompt_length + self.alen + self.vlen, self.llen, True)
        self.fuse_audio_with_lv = ConvProjector(self.prompt_length + self.llen + self.vlen, self.alen, True)
        self.fuse_vision_with_la = ConvProjector(self.prompt_length + self.alen + self.llen, self.vlen, True)

        # Modality-signal prompts
        self.text_prompt_m = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.audio_prompt_m = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.vision_prompt_m = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))
        self.text_prompt_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.audio_prompt_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.vision_prompt_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))

        # Missing-type prompts
        self.missing_type_prompt = nn.Parameter(torch.zeros(3, self.prompt_length, self.prompt_dim))
        self.map_audio = nn.Parameter(torch.zeros(self.alen, 2 * self.prompt_dim))
        self.map_vision = nn.Parameter(torch.zeros(self.vlen, 2 * self.prompt_dim))
        self.map_text = nn.Parameter(torch.zeros(self.llen, 2 * self.prompt_dim))

        # Temporal convolutional projections
        self.proj_text = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_audio = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_vision = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

    def get_complete_data(self, x_l, x_a, x_v, weak_mode):
        x_l, x_a, x_v = x_l.unsqueeze(0), x_a.unsqueeze(0), x_v.unsqueeze(0)

        if not isinstance(weak_mode, int):
            weak_mode = int(weak_mode)
        weak_mode = max(0, min(2, weak_mode))

        if weak_mode == 0:
            fused = torch.cat(
                [self.generative_prompts[0, :, :], self.audio_to_text(x_a)[0], self.vision_to_text(x_v)[0]],
                dim=1,
            ).unsqueeze(0)
            x_l = self.fuse_text_with_av(fused.transpose(1, 2)).transpose(1, 2) + self.text_prompt_m
            x_a = self.proj_audio(x_a) + self.audio_prompt_nm
            x_v = self.proj_vision(x_v) + self.vision_prompt_nm
        elif weak_mode == 1:
            # Reconstruct audio with text & vision prompts
            fused = torch.cat(
                [self.generative_prompts[1, :, :], self.text_to_audio(x_l)[0], self.vision_to_audio(x_v)[0]],
                dim=1,
            ).unsqueeze(0)
            x_a = self.fuse_audio_with_lv(fused.transpose(1, 2)).transpose(1, 2) + self.audio_prompt_m
            x_v = self.proj_vision(x_v) + self.vision_prompt_nm
            x_l = self.proj_text(x_l) + self.text_prompt_nm
        else:  # weak_mode == 2
            # Reconstruct vision with text & audio prompts
            fused = torch.cat(
                [self.generative_prompts[2, :, :], self.text_to_vision(x_l)[0], self.audio_to_vision(x_a)[0]],
                dim=1,
            ).unsqueeze(0)
            x_v = self.fuse_vision_with_la(fused.transpose(1, 2)).transpose(1, 2) + self.vision_prompt_m
            x_l = self.proj_text(x_l) + self.text_prompt_nm
            x_a = self.proj_audio(x_a) + self.audio_prompt_nm

        return x_l, x_a, x_v

    def get_proj_matrix(self):
        a_v_l = (
            self.audio_prompt_nm @ self.map_audio
            + self.vision_prompt_nm @ self.map_vision
            + self.text_prompt_nm @ self.map_text
        ).unsqueeze(0)
        am_v_l = (
            self.audio_prompt_m @ self.map_audio
            + self.vision_prompt_nm @ self.map_vision
            + self.text_prompt_nm @ self.map_text
        ).unsqueeze(0)
        a_vm_l = (
            self.audio_prompt_nm @ self.map_audio
            + self.vision_prompt_m @ self.map_vision
            + self.text_prompt_nm @ self.map_text
        ).unsqueeze(0)
        a_v_lm = (
            self.audio_prompt_nm @ self.map_audio
            + self.vision_prompt_nm @ self.map_vision
            + self.text_prompt_m @ self.map_text
        ).unsqueeze(0)
        am_vm_l = (
            self.audio_prompt_m @ self.map_audio
            + self.vision_prompt_m @ self.map_vision
            + self.text_prompt_nm @ self.map_text
        ).unsqueeze(0)
        am_v_lm = (
            self.audio_prompt_m @ self.map_audio
            + self.vision_prompt_nm @ self.map_vision
            + self.text_prompt_m @ self.map_text
        ).unsqueeze(0)
        a_vm_lm = (
            self.audio_prompt_nm @ self.map_audio
            + self.vision_prompt_m @ self.map_vision
            + self.text_prompt_m @ self.map_text
        ).unsqueeze(0)
        self.mp = torch.cat(
            [a_v_lm, am_v_l, a_vm_l, am_v_lm, a_vm_lm, am_vm_l, a_v_l], dim=0
        )
        return self.mp

    def compute_batch_prompt(self, weak: torch.Tensor) -> torch.Tensor:
        mp = getattr(self, 'mp', None)
        if mp is None:
            mp = self.get_proj_matrix()

        batch_prompt = None
        for idx in range(len(weak)):
            curr = torch.matmul(self.missing_type_prompt, mp[weak[idx]]).unsqueeze(0)
            if batch_prompt is None:
                batch_prompt = curr
            else:
                batch_prompt = torch.cat([batch_prompt, curr], dim=0)

        batch_prompt = batch_prompt.transpose(0, 1)
        return batch_prompt


