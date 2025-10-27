import torch
import torch.nn as nn

class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MyTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # Ensure masks are boolean to avoid PyTorch Transformer warning
        if mask is not None and mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        if src_key_padding_mask is not None and src_key_padding_mask.dtype is not torch.bool:
            src_key_padding_mask = src_key_padding_mask.to(torch.bool)
        output = src
        hidden_states = []

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            hidden_states.append(output)

        if self.norm is not None:
            output = self.norm(output)

        return output, hidden_states
