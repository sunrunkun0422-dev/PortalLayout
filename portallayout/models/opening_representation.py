"""Single-view geometry/context opening descriptors, independent of matching.

Predicted intervals, probabilities and geometry are fixed inputs. Feature tensors
remain differentiable, so freezing the backbone is the caller's responsibility.
No ground-truth or cross-view input is accepted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


GEOMETRY_ATTRIBUTES = (
    'angular_width_turns', 'width_over_median_depth', 'width_over_wall_length',
    'wall_length_over_median_depth', 'nearest_wall_end_ratio', 'corner_cos_min',
    'corner_cos_max', 'relative_expansion_mean', 'relative_expansion_max',
    'opening_probability_mean', 'opening_probability_max', 'dominant_wall_fraction',
)
WALL_ATTRIBUTES = ('opening_width', 'wall_length', 'nearest_wall_end_ratio',
                   'corner_cos_min', 'corner_cos_max', 'dominant_wall_fraction')
ABLATIONS = ('baseline', 'structure', 'structure_context', 'structure_context_geometry', 'full')


@dataclass(frozen=True)
class OpeningRepresentationConfig:
    feature_1d_dim: int = 512
    feature_2d_dim: int = 1024
    descriptor_dim: int = 256
    context_hidden_dim: int = 64
    attention_heads: int = 4
    attention_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        for name in ('feature_1d_dim', 'feature_2d_dim', 'descriptor_dim', 'context_hidden_dim',
                     'attention_heads', 'attention_layers'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if self.descriptor_dim % self.attention_heads:
            raise ValueError('descriptor dimension must be divisible by attention heads')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0,1)')


@dataclass(frozen=True)
class RepresentationCandidates:
    intervals: torch.Tensor  # [B,K,2], inclusive; start>end crosses the seam
    valid: torch.Tensor      # [B,K], padding intervals are ignored
    candidate_ids: Tuple[Tuple[Optional[str], ...], ...]


@dataclass(frozen=True)
class PredictedOpeningGeometry:
    depth_enclosed: torch.Tensor       # [B,N], raw predicted depth; abs used explicitly
    depth_extended: torch.Tensor       # [B,N]
    height_ratio: torch.Tensor         # [B] or [B,1]; nonpositive/missing -> unavailable
    wall_attributes: torch.Tensor      # [B,K,6], WALL_ATTRIBUTES order
    wall_attribute_valid: torch.Tensor # [B,K,6], explicit missingness


@dataclass(frozen=True)
class OpeningRepresentationOutput:
    descriptors: torch.Tensor
    candidate_ids: Tuple[Tuple[Optional[str], ...], ...]
    valid_mask: torch.Tensor
    projected_feature_2d: torch.Tensor
    baseline_descriptors: torch.Tensor
    diagnostics: Optional[Dict[str, torch.Tensor]] = None


def masked_softmax(logits, mask, dim=-1):
    """All-masked rows return exact zeros without evaluating softmax(-inf,...)."""
    mask = mask.expand_as(logits)
    active = mask.any(dim=dim, keepdim=True)
    safe = torch.where(active, logits.float().masked_fill(~mask, -torch.inf),
                       torch.zeros_like(logits, dtype=torch.float32))
    return torch.softmax(safe, dim=dim).to(logits.dtype) * mask.to(logits.dtype)


def circular_grid_sample(features, uv):
    """Bilinear sampling at normalized image coordinates; periodic only in u.

    Pixel centers lie at (j+.5)/W. Appending edge columns allows u=0/1 to
    interpolate between the final and first columns with align_corners=False.
    """
    b, c, h, w = features.shape
    if uv.shape[0] != b or uv.shape[-1] != 2:
        raise ValueError('sampling coordinates must be [B,...,2]')
    shape = uv.shape[1:-1]
    if uv.numel() == 0:
        return features.new_empty((b,) + shape + (c,))
    padded = torch.cat((features[..., -1:], features, features[..., :1]), dim=-1)
    u = torch.remainder(uv[..., 0], 1.)
    x = 2 * (u * w + 1) / (w + 2) - 1
    y = 2 * uv[..., 1].clamp(0, 1) - 1
    grid = torch.stack((x, y), dim=-1).reshape(b, -1, 1, 2)
    value = F.grid_sample(padded, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return value.squeeze(-1).transpose(1, 2).reshape((b,) + shape + (c,))


def _sample_profile(values, valid, u):
    safe = torch.where(valid, values, torch.zeros_like(values))
    uv = torch.stack((u, torch.full_like(u, .5)), dim=-1)
    sampled = circular_grid_sample(safe[:, None, None, :], uv)[..., 0]
    available = circular_grid_sample(valid[:, None, None, :].to(values.dtype), uv)[..., 0] > .9999
    return sampled, available


class CircularOpeningAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.position_bias = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, heads, bias=False))
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * dim, dim), nn.Dropout(dropout))

    def forward(self, features, centers, valid):
        # Exclude all-padding images, while keeping candidate order unchanged.
        active = torch.where(valid.any(dim=-1))[0]
        if not len(active):
            return features * 0
        x, center, mask = features[active], centers[active], valid[active]
        b, k, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, k, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, key, value = qkv.unbind(0)
        delta = (center[:, :, None] - center[:, None, :]) * (2 * math.pi)
        relative = torch.stack((torch.sin(delta), torch.cos(delta)), dim=-1)
        bias = self.position_bias(relative).permute(0, 3, 1, 2)
        score = q @ key.transpose(-1, -2) / math.sqrt(self.head_dim) + bias
        attention = masked_softmax(score, mask[:, None, None, :])
        update = (self.dropout(attention) @ value).transpose(1, 2).reshape(b, k, d)
        x = x + self.dropout(self.output(update))
        x = (x + self.ffn(self.norm2(x))) * mask[..., None]
        return torch.zeros_like(features).index_copy(0, active, x)


class OpeningRepresentationEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or OpeningRepresentationConfig()
        c = self.config
        d = c.descriptor_dim
        self.projection_1d = nn.Linear(c.feature_1d_dim, d)
        self.projection_2d = nn.Conv2d(c.feature_2d_dim, d, 1)
        self.base_norm = nn.LayerNorm(d)
        self.column_roles = nn.Embedding(5, d)
        self.height_roles = nn.Embedding(4, d)
        self.structure_query, self.structure_key = nn.Linear(d, d), nn.Linear(d, d)
        self.structure_value = nn.Linear(d, d)
        self.structure_norm = nn.LayerNorm(d)
        self.context_query = nn.Linear(d, c.context_hidden_dim)
        self.context_key = nn.Conv2d(d, c.context_hidden_dim, 1)
        self.context_position = nn.Linear(2, c.context_hidden_dim, bias=False)
        self.context_score = nn.Linear(c.context_hidden_dim, 1, bias=False)
        self.context_norm = nn.LayerNorm(d)
        self.geometry_mlp = nn.Sequential(nn.Linear(24, 128), nn.GELU(), nn.Linear(128, d), nn.LayerNorm(d))
        self.gate = nn.Sequential(nn.Linear(4 * d, 128), nn.GELU(), nn.Linear(128, 3))
        self.fusion_norm = nn.LayerNorm(d)
        self.intra_attention = nn.ModuleList([
            CircularOpeningAttention(d, c.attention_heads, c.dropout) for _ in range(c.attention_layers)])

    def _validate(self, feature_1d, feature_2d, probability, candidates, geometry):
        if feature_1d.ndim != 3 or feature_2d.ndim != 4:
            raise ValueError('feature shapes must be [B,N,C1] and [B,C2,H,W]')
        b, n, c = feature_1d.shape
        if b < 1 or n < 3 or c != self.config.feature_1d_dim or feature_2d.shape[:2] != (b, self.config.feature_2d_dim):
            raise ValueError('feature dimensions do not match encoder configuration')
        if min(feature_2d.shape[-2:]) < 1:
            raise ValueError('spatial feature dimensions must be positive')
        if probability.shape != (b, n) or not torch.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError('opening probability must be finite [B,N] in [0,1]')
        if not torch.isfinite(feature_1d).all() or not torch.isfinite(feature_2d).all():
            raise ValueError('visual features must be finite')
        if candidates.valid.ndim != 2 or candidates.valid.shape[0] != b or candidates.valid.dtype != torch.bool:
            raise ValueError('candidate valid mask must be bool [B,K]')
        k = candidates.valid.shape[1]
        if candidates.intervals.shape != (b, k, 2) or candidates.intervals.dtype != torch.long:
            raise ValueError('candidate intervals must be int64 [B,K,2]')
        selected = candidates.intervals[candidates.valid]
        if selected.numel() and not ((selected >= 0) & (selected < n)).all():
            raise ValueError('valid candidate token interval outside [0,N)')
        if len(candidates.candidate_ids) != b or any(len(ids) != k for ids in candidates.candidate_ids):
            raise ValueError('candidate IDs must retain [B,K] order, including padding')
        for row, flags in zip(candidates.candidate_ids, candidates.valid.detach().cpu().tolist()):
            ids = [uid for uid, flag in zip(row, flags) if flag]
            if any(not isinstance(uid, str) or not uid for uid in ids) or len(ids) != len(set(ids)):
                raise ValueError('valid candidate IDs must be unique nonempty strings per image')
        if geometry.depth_enclosed.shape != (b, n) or geometry.depth_extended.shape != (b, n):
            raise ValueError('predicted depth must have shape [B,N]')
        if geometry.height_ratio.shape not in ((b,), (b, 1)):
            raise ValueError('height ratio must have shape [B] or [B,1]')
        if geometry.wall_attributes.shape != (b, k, 6) or geometry.wall_attribute_valid.shape != (b, k, 6):
            raise ValueError('wall attributes and validity must have shape [B,K,6]')
        if geometry.wall_attribute_valid.dtype != torch.bool:
            raise ValueError('wall attribute validity must be bool')
        return b, n, k

    def forward(self, feature_1d, feature_2d, opening_probability, candidates,
                predicted_geometry, return_diagnostics=False, ablation='full'):
        if ablation not in ABLATIONS:
            raise ValueError('unsupported ablation: ' + str(ablation))
        b, n, k = self._validate(feature_1d, feature_2d, opening_probability, candidates, predicted_geometry)
        device, dtype = self.projection_1d.weight.device, self.projection_1d.weight.dtype
        f1, f2 = feature_1d.to(device=device, dtype=dtype), feature_2d.to(device=device, dtype=dtype)
        probability = opening_probability.detach().to(device=device, dtype=dtype)
        valid = candidates.valid.to(device)
        intervals = candidates.intervals.to(device)
        projected = self.projection_2d(f2)
        if k == 0:
            base = self.projection_1d(f1[:, :0])
            return OpeningRepresentationOutput(base, candidates.candidate_ids, valid, projected, base,
                {} if return_diagnostics else None)
        start = torch.where(valid, intervals[..., 0], 0)
        length = torch.where(valid, torch.remainder(intervals[..., 1] - start, n) + 1, 0)
        token = torch.arange(n, device=device)
        token_mask = (torch.remainder(token[None, None, :] - start[..., None], n) < length[..., None]) & valid[..., None]
        raw_weights = probability[:, None, :] * token_mask
        mass = raw_weights.sum(-1, keepdim=True)
        weights = torch.where(mass > 1e-8, raw_weights / mass.clamp_min(1e-8),
                              token_mask.to(dtype) / length[..., None].clamp_min(1))
        pooled = weights @ f1
        base = self.base_norm(self.projection_1d(pooled)) * valid[..., None]
        width = length.to(dtype) / n
        left = start.to(dtype) / n
        center = torch.remainder(left + .5 * width, 1.)

        # Fixed predicted geometry is sanitized with explicit availability.
        g = predicted_geometry
        enclosed = g.depth_enclosed.detach().to(device=device, dtype=dtype).abs()
        extended = g.depth_extended.detach().to(device=device, dtype=dtype).abs()
        enc_valid = torch.isfinite(enclosed) & (enclosed > 1e-6)
        ext_valid = torch.isfinite(extended) & (extended > 1e-6)
        enclosed = torch.where(enc_valid, enclosed, torch.zeros_like(enclosed))
        extended = torch.where(ext_valid, extended, torch.zeros_like(extended))
        ratio = g.height_ratio.detach().to(device=device, dtype=dtype).reshape(b)
        ratio_valid = torch.isfinite(ratio) & (ratio > 1e-6)
        ratio = torch.where(ratio_valid, ratio, torch.ones_like(ratio))

        # Five horizontal roles, three wall-band heights plus the floor line.
        offsets = f1.new_tensor([-.25, 0., .5, 1., 1.25])
        sample_u = torch.remainder(left[..., None] + width[..., None] * offsets, 1.)
        depth, depth_valid = _sample_profile(enclosed, enc_valid, sample_u)
        floor_v = .5 + torch.atan2(torch.ones_like(depth), depth.clamp_min(1e-6)) / math.pi
        ceiling_v = .5 - torch.atan2(ratio[:, None, None].expand_as(depth), depth.clamp_min(1e-6)) / math.pi
        fractions = f1.new_tensor([.2, .5, .8])
        wall_v = ceiling_v[..., None] + (floor_v-ceiling_v)[..., None] * fractions
        sample_v = torch.cat((wall_v, floor_v[..., None]), -1)
        wall_available = depth_valid & ratio_valid[:, None, None] & valid[..., None]
        sample_valid = torch.cat((wall_available[..., None].expand(-1, -1, -1, 3),
                                  (depth_valid & valid[..., None])[..., None]), -1).reshape(b, k, 20)
        uv = torch.stack((sample_u[..., None].expand_as(sample_v), sample_v), -1).reshape(b, k, 20, 2)
        samples = circular_grid_sample(projected, uv)
        roles = (self.column_roles.weight[:, None] + self.height_roles.weight[None, :]).reshape(20, -1)
        role_samples = samples + roles[None, None]
        structure_logits = (self.structure_query(base)[..., None, :] * self.structure_key(role_samples)).sum(-1) / math.sqrt(base.shape[-1])
        structure_attention = masked_softmax(structure_logits, sample_valid)
        structure_available = sample_valid.any(-1)
        f_open = self.structure_norm((structure_attention[..., None] * self.structure_value(role_samples)).sum(-2))
        f_open = f_open * structure_available[..., None]

        # Candidate-relative soft angular support, with a predicted wall band.
        h, w = projected.shape[-2:]
        grid_u = (torch.arange(w, device=device, dtype=dtype) + .5) / w
        grid_v = (torch.arange(h, device=device, dtype=dtype) + .5) / h
        delta = torch.remainder(grid_u[None, None, :] - center[..., None] + .5, 1.) - .5
        support_width = (2 * width).clamp(max=1.)
        angular = torch.sigmoid((support_width[..., None] * .5 - delta.abs()) * w)
        angular = torch.where(support_width[..., None] >= 1, torch.ones_like(angular), angular)
        column_depth, column_valid = _sample_profile(enclosed, enc_valid, grid_u[None].expand(b, -1))
        bottom = .5 + torch.atan2(torch.ones_like(column_depth), column_depth.clamp_min(1e-6)) / math.pi
        top = .5 - torch.atan2(ratio[:, None].expand_as(column_depth), column_depth.clamp_min(1e-6)) / math.pi
        vertical = torch.sigmoid((grid_v[None, :, None] - top[:, None, :]) * h) * torch.sigmoid((bottom[:, None, :] - grid_v[None, :, None]) * h)
        band_valid = column_valid & ratio_valid[:, None]
        vertical = torch.where(band_valid[:, None, :], vertical, torch.ones_like(vertical))
        soft_cover = angular[..., None, :] * vertical[:, None, :, :] * valid[..., None, None]
        relative_position = torch.stack((torch.sin(delta * 2 * math.pi), torch.cos(delta * 2 * math.pi)), -1)
        context_hidden = (self.context_key(projected).permute(0, 2, 3, 1)[:, None] +
                          self.context_query(base)[..., None, None, :] +
                          self.context_position(relative_position)[:, :, None, :, :])
        context_logits = self.context_score(torch.tanh(context_hidden))[..., 0] + torch.log(soft_cover.clamp_min(1e-8))
        context_attention = masked_softmax(context_logits.flatten(-2), valid[..., None]).reshape(b, k, h, w)
        f_ctx = torch.einsum('bkhw,bdhw->bkd', context_attention, projected)
        f_ctx = self.context_norm(f_ctx) * valid[..., None]

        # Twelve geometry attributes with twelve independent missingness bits.
        raw_attr = g.wall_attributes.detach().to(device=device, dtype=dtype)
        attr_valid = g.wall_attribute_valid.to(device) & torch.isfinite(raw_attr) & valid[..., None]
        attr = torch.where(attr_valid, raw_attr, torch.zeros_like(raw_attr))
        wall_width, wall_length = attr[..., 0], attr[..., 1]
        attr_valid = attr_valid.clone()
        attr_valid[..., :2] &= attr[..., :2] > 1e-6
        # A masked median must not include absent depth values as zeros.
        median = torch.nanmedian(torch.where(enc_valid, enclosed, enclosed.new_full((), torch.nan)), dim=-1).values
        median_valid = torch.isfinite(median) & (median > 1e-6)
        median = torch.where(median_valid, median, torch.ones_like(median))
        contrast_valid = enc_valid & ext_valid
        contrast = F.relu(extended - enclosed) / enclosed.clamp_min(1e-6)
        contrast = torch.where(contrast_valid, contrast, torch.zeros_like(contrast))
        usable = token_mask & contrast_valid[:, None, :]
        contrast_mean = (contrast[:, None] * usable).sum(-1) / usable.sum(-1).clamp_min(1)
        contrast_max = torch.where(usable, contrast[:, None], torch.zeros_like(raw_weights)).amax(-1)
        prob_mean = (probability[:, None] * token_mask).sum(-1) / length.clamp_min(1)
        prob_max = torch.where(token_mask, probability[:, None], torch.zeros_like(raw_weights)).amax(-1)
        geometry_values = torch.stack((width, wall_width/median[:, None],
            wall_width/wall_length.clamp_min(1e-6), wall_length/median[:, None],
            attr[..., 2], attr[..., 3], attr[..., 4], contrast_mean, contrast_max,
            prob_mean, prob_max, attr[..., 5]), -1)
        geometry_valid = torch.stack((valid, attr_valid[..., 0] & median_valid[:, None],
            attr_valid[..., 0] & attr_valid[..., 1], attr_valid[..., 1] & median_valid[:, None],
            attr_valid[..., 2], attr_valid[..., 3], attr_valid[..., 4], usable.any(-1),
            usable.any(-1), valid, valid, attr_valid[..., 5]), -1)
        geometry_values = torch.where(geometry_valid, geometry_values, torch.zeros_like(geometry_values))
        geometry_available = geometry_valid.any(-1)
        geom = self.geometry_mlp(torch.cat((geometry_values, geometry_valid.to(dtype)), -1)) * valid[..., None]

        stage = ABLATIONS.index(ablation)
        branch_valid = torch.stack((structure_available & (stage >= 1),
                                    valid & (stage >= 2), geometry_available & (stage >= 3)), -1)
        gate_logits = self.gate(torch.cat((base, f_open, f_ctx, geom), -1))
        gates = masked_softmax(gate_logits, branch_valid)
        enhancement = (torch.stack((f_open, f_ctx, geom), -2) * gates[..., None]).sum(-2)
        descriptors = self.fusion_norm(base + enhancement) * valid[..., None]
        baseline = self.fusion_norm(base) * valid[..., None]
        if stage == 4:
            for layer in self.intra_attention:
                descriptors = layer(descriptors, center, valid)
        diagnostics = None
        if return_diagnostics:
            diagnostics = {'base_features': base, 'structure_features': f_open,
                'context_features': f_ctx, 'geometry_features': geom, 'fusion_weights': gates,
                'branch_valid': branch_valid, 'sampling_uv': uv, 'sampling_valid': sample_valid,
                'structure_attention': structure_attention, 'soft_cover': soft_cover,
                'context_attention': context_attention, 'wall_band_valid': band_valid,
                'geometry_attributes': geometry_values, 'geometry_attribute_valid': geometry_valid,
                'candidate_centers': center, 'candidate_widths': width}
        return OpeningRepresentationOutput(descriptors, candidates.candidate_ids, valid, projected, baseline, diagnostics)
