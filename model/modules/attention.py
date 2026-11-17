import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    A simplified version of attention from DSTFormer that also considers x tensor to be (B, T, J, C) instead of
    (B * T, J, C)
    """

    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
                                                                                           5)  # (3, B, H, T, J, C)
        if self.mode == 'temporal':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v)
        elif self.mode == 'spatial':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, H, T, J, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)

    def forward_temporal(self, q, k, v):
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)  # (B, H, J, T, C)
        vt = v.transpose(2, 3)  # (B, H, J, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)

class MultiScaleTemporalAttention(nn.Module):
    """
    Multi-scale temporal attention module:
    1) Original scale (T): Short-Window Self-Attention
    2) 2x Downsampled scale (T/2): Global Self-Attention + 2x Upsampling
    3) 4x Downsampled scale (T/4): Global Self-Attention + 4x Upsampling
    """
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., window_size=9):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv_orig = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)

        self.down2 = nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=2, padding=1)
        self.qkv_2x = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)

        self.down4 = nn.Conv1d(dim_in, dim_in, kernel_size=5, stride=4, padding=2)
        self.qkv_4x = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.fusion = nn.Linear(dim_in * 3, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _window_attention(self, x):
        B, T, J, C = x.shape
        W = self.window_size
        half_w = W // 2
        H = self.num_heads
        head_dim = C // H

        qkv = self.qkv_orig(x).reshape(B, T, J, 3, H, head_dim).permute(3, 0, 2, 4, 1, 5) # (3, B, J, H, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_pad = F.pad(k.reshape(B * J * H, T, head_dim).transpose(1, 2), (half_w, half_w), mode='replicate') # (B*J*H, head_dim, T)
        v_pad = F.pad(v.reshape(B * J * H, T, head_dim).transpose(1, 2), (half_w, half_w), mode='replicate') # (B*J*H, head_dim, T)

        k_win = k_pad.unfold(dimension=-1, size=W, step=1).permute(0, 2, 3, 1).reshape(B, J, H, T, W, head_dim) # (B, J, H, T, W, head_dim)
        v_win = v_pad.unfold(dimension=-1, size=W, step=1).permute(0, 2, 3, 1).reshape(B, J, H, T, W, head_dim) # (B, J, H, T, W, head_dim)

        attn = (q.unsqueeze(-2) @ k_win.transpose(-2, -1)) * self.scale  # (B, J, H, T, 1, W)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v_win).squeeze(-2)  # (B, J, H, T, head_dim)
        out = out.permute(0, 3, 1, 2, 4).reshape(B, T, J, C)
        return out # (B, T, J, C)

    def _downsampled_attention(self, x, down_layer, qkv_layer, target_T):
        B, T, J, C = x.shape

        # downsampling to lower temporal resolution
        x_in = x.permute(0, 2, 3, 1).reshape(B * J, C, T)
        x_down = down_layer(x_in)  # (B*J, C, T_down)
        T_down = x_down.shape[-1]

        x_seq = x_down.transpose(1, 2)
        qkv = qkv_layer(x_seq).reshape(B * J, T_down, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B * J, num_heads, T_down, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B * J, T_down, C)

        # upsampling to original temporal resolution
        out_up = F.interpolate(out.transpose(1, 2), size=target_T, mode='linear', align_corners=False)
        out_up = out_up.transpose(1, 2).view(B, J, target_T, C).permute(0, 2, 1, 3)  # (B, T, J, C)
        return out_up

    def forward(self, x):
        B, T, J, C = x.shape

        out_orig = self._window_attention(x)
        out_2x = self._downsampled_attention(x, self.down2, self.qkv_2x, target_T=T)
        out_4x = self._downsampled_attention(x, self.down4, self.qkv_4x, target_T=T)

        out = torch.cat([out_orig, out_2x, out_4x], dim=-1)
        out = self.fusion(out)
        out = self.proj_drop(out)
        return out