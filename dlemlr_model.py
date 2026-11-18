# -*- coding: utf-8 -*-
"""
dlemlr_model.py —— 独立可移植版 Chen et al. (JACS 2024) SimpleViT + 物理核通道适配
零外部路径、零 einops 依赖；与同目录 sim2d.py 配合运行。

规格:
- 保持 Chen 原版 Transformer 主体宽度/深度 (dim=1024, depth=10, heads=10, dim_head=64, mlp_dim=2048)
- 输入 (B, 3, 48, 96) = [K1_img, Y_img, K2_img], patch_size=(12, 16) -> 24 tokens (对齐原论文 5x5=25 tokens)
- 输出 (B, 48, 96) 谱格 + Chen SI Eq. (S11) scale factor gamma 反向还原
"""
import os
import sys
import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from sim2d import TI_MS, ECHO_MS, N_T1, N_T2, T1_GRID, T2_GRID

IMG_H = N_T1   # 48
IMG_W = N_T2   # 96
TI_IMG = np.geomspace(float(TI_MS[0]), float(TI_MS[-1]), IMG_H)
LOG_TI_IMG = np.log10(TI_IMG)
ECHO_IMG_96 = np.geomspace(float(ECHO_MS[0]), float(ECHO_MS[-1]), IMG_W)
LOG_ECHO_IMG_96 = np.log10(ECHO_IMG_96)
ECHO_IMG_48 = np.geomspace(float(ECHO_MS[0]), float(ECHO_MS[-1]), IMG_H)
T1_96 = np.logspace(np.log10(float(T1_GRID[0])), np.log10(float(T1_GRID[-1])), IMG_W)

K1_IMG = (1.0 - 2.0 * np.exp(-TI_IMG[:, None] / T1_96[None, :])).astype(np.float32)
K2_IMG = np.exp(-ECHO_IMG_48[:, None] / T2_GRID[None, :]).astype(np.float32)

MAX_TI_TRAIN = float(TI_MS[-1])
MAX_ECHO_TRAIN = float(ECHO_MS[-1])
LOG_T1_GRID = np.log10(T1_GRID)
LOG_T2_GRID = np.log10(T2_GRID)


def posemb_sincos_2d(patches, temperature=10000):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=10, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        bsz, n_tok, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.view(bsz, n_tok, self.heads, self.dim_head).permute(0, 2, 1, 3)
            for t in qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(bsz, n_tok, self.heads * self.dim_head)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SimpleViTGrid(nn.Module):
    def __init__(
        self,
        image_size=(IMG_H, IMG_W),
        patch_size=(12, 16),
        out_shape=(N_T1, N_T2),
        dim=1024,
        depth=10,
        heads=10,
        mlp_dim=2048,
        channels=3,
        dim_head=64,
    ):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0
        self.hp = ih // ph
        self.wp = iw // pw
        self.ph = ph
        self.pw = pw
        self.channels = channels
        self.out_h, self.out_w = out_shape

        patch_dim = channels * ph * pw
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.out_h * self.out_w),
        )

    def forward(self, img):
        bsz = img.size(0)
        x = (
            img.view(bsz, self.channels, self.hp, self.ph, self.wp, self.pw)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(bsz, self.hp, self.wp, self.ph * self.pw * self.channels)
        )
        x = self.patch_proj(x)
        pe = posemb_sincos_2d(x)
        x = x.view(bsz, self.hp * self.wp, -1) + pe
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.linear_head(x).reshape(bsz, self.out_h, self.out_w)
        return torch.abs(out)


def resample_decay_batch_to_img(Y_batch, ti_ms=TI_MS, echo_ms=ECHO_MS):
    Y_np = np.asarray(Y_batch, dtype=np.float64)
    if Y_np.ndim == 2:
        Y_np = Y_np[None, :, :]
    ti_arr = np.asarray(ti_ms, dtype=np.float64)
    echo_arr = np.asarray(echo_ms, dtype=np.float64)

    gamma_t1 = MAX_TI_TRAIN / float(np.max(ti_arr))
    gamma_t2 = MAX_ECHO_TRAIN / float(np.max(echo_arr))
    ti_scaled = np.log10(ti_arr * gamma_t1)
    echo_scaled = np.log10(echo_arr * gamma_t2)

    scales = np.max(np.abs(Y_np), axis=(1, 2), keepdims=True) + 1e-9
    Yn = Y_np / scales

    bsz, n_ti, n_ech = Yn.shape
    idx_hi = np.searchsorted(ti_scaled, LOG_TI_IMG, side="right")
    idx_hi = np.clip(idx_hi, 1, n_ti - 1)
    idx_lo = idx_hi - 1
    w_ti = np.clip((LOG_TI_IMG - ti_scaled[idx_lo]) / (ti_scaled[idx_hi] - ti_scaled[idx_lo] + 1e-12), 0.0, 1.0)
    Y_ti = (1.0 - w_ti[None, :, None]) * Yn[:, idx_lo, :] + w_ti[None, :, None] * Yn[:, idx_hi, :]

    idx_eh = np.searchsorted(echo_scaled, LOG_ECHO_IMG_96, side="right")
    idx_eh = np.clip(idx_eh, 1, n_ech - 1)
    idx_el = idx_eh - 1
    w_eh = np.clip((LOG_ECHO_IMG_96 - echo_scaled[idx_el]) / (echo_scaled[idx_eh] - echo_scaled[idx_el] + 1e-12), 0.0, 1.0)
    Y_img = ((1.0 - w_eh[None, None, :]) * Y_ti[:, :, idx_el] + w_eh[None, None, :] * Y_ti[:, :, idx_eh]).astype(np.float32)

    k1_b = np.broadcast_to(K1_IMG[None], (bsz, IMG_H, IMG_W))
    k2_b = np.broadcast_to(K2_IMG[None], (bsz, IMG_H, IMG_W))
    concat = np.stack([k1_b, Y_img, k2_b], axis=1).astype(np.float32)
    return concat, gamma_t1, gamma_t2


def unscale_spectrum_grid(M_norm, gamma_t1=1.0, gamma_t2=1.0):
    M = np.asarray(M_norm, dtype=np.float64)
    if abs(gamma_t1 - 1.0) < 1e-4 and abs(gamma_t2 - 1.0) < 1e-4:
        return M

    if abs(gamma_t1 - 1.0) >= 1e-4:
        q1 = LOG_T1_GRID + np.log10(gamma_t1)
        M1 = np.empty_like(M)
        for j in range(M.shape[1]):
            M1[:, j] = np.interp(q1, LOG_T1_GRID, M[:, j], left=0.0, right=0.0)
        M = M1

    if abs(gamma_t2 - 1.0) >= 1e-4:
        q2 = LOG_T2_GRID + np.log10(gamma_t2)
        M2 = np.empty_like(M)
        for i in range(M.shape[0]):
            M2[i, :] = np.interp(q2, LOG_T2_GRID, M[i, :], left=0.0, right=0.0)
        M = M2

    return np.clip(M, 0.0, None)

