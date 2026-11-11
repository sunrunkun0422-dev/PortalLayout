"""Pair-conditioned opening retrieval and converged partial optimal transport.

This module accepts predicted tensors only. Threshold calibration and supervision
live outside the forward path. A sample is one unordered panorama pair.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
from torch import nn


@dataclass(frozen=True)
class OpeningPairConfig:
    dim: int = 256
    heads: int = 4
    chunk_size: int = 16
    sinkhorn_checks: Tuple[int, ...] = (50, 100, 200, 500, 1000)
    marginal_tolerance: float = 1e-3

    def __post_init__(self):
        if self.dim < 1 or self.heads < 1 or self.dim % self.heads or self.chunk_size < 1:
            raise ValueError('invalid attention configuration')
        if (not self.sinkhorn_checks or min(self.sinkhorn_checks) < 1 or
                tuple(sorted(set(self.sinkhorn_checks))) != tuple(self.sinkhorn_checks) or
                self.marginal_tolerance <= 0):
            raise ValueError('invalid Sinkhorn convergence configuration')


@dataclass
class OpeningPairView:
    descriptors: torch.Tensor            # [K,D]
    memory: torch.Tensor                 # [D,H,W]
    intervals: torch.Tensor              # [K,2], inclusive, circular
    valid: torch.Tensor                  # [K]
    candidate_ids: tuple
    token_count: int
    geometry: Optional[torch.Tensor] = None       # [K,12], normalized attributes
    geometry_valid: Optional[torch.Tensor] = None
    feature_1d: Optional[torch.Tensor] = None
    opening_probability: Optional[torch.Tensor] = None


def log_partial_transport(scores, bin_score, checks=(50, 100, 200, 500, 1000), tolerance=1e-3):
    """Symmetric log-Sinkhorn; average dual orientations to remove update-order bias.

    Both orientations have the same target marginals. Their average converges to
    the same transport and is exactly equivariant to transposing the input.
    Empty sets are handled by the caller. Convergence is checked, never assumed.
    """
    if scores.ndim != 2 or min(scores.shape) < 1 or not torch.isfinite(scores).all():
        raise ValueError('transport requires a nonempty finite matrix')
    m, n = scores.shape
    s, alpha = scores.float(), bin_score.float()
    z = torch.cat((torch.cat((s, alpha.expand(m, 1)), 1), alpha.expand(1, n+1)), 0)
    mu = torch.cat((s.new_ones(m), s.new_tensor([n]))) / (m+n)
    nu = torch.cat((s.new_ones(n), s.new_tensor([m]))) / (m+n)
    lm, ln = mu.log(), nu.log()
    u, v = torch.zeros_like(lm), torch.zeros_like(ln)
    ur, vr = torch.zeros_like(ln), torch.zeros_like(lm)
    converged = False
    for iteration in range(1, max(checks)+1):
        u = lm - torch.logsumexp(z + v[None], 1)
        v = ln - torch.logsumexp(z + u[:, None], 0)
        ur = ln - torch.logsumexp(z.T + vr[None], 1)
        vr = lm - torch.logsumexp(z.T + ur[:, None], 0)
        if iteration in checks:
            lp = torch.logaddexp(z + u[:, None] + v[None],
                                z + vr[:, None] + ur[None]) - math.log(2.)
            p = lp.detach().exp()
            error = torch.maximum(((p.sum(1)-mu).abs()/mu).max(),
                                  ((p.sum(0)-nu).abs()/nu).max())
            converged = bool(torch.isfinite(error) and error <= tolerance)
            if converged:
                break
    return {'log_transport': lp, 'mu': mu, 'nu': nu, 'iterations': iteration,
            'marginal_error': float(error), 'converged': converged}


class OpeningPairMatcher(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or OpeningPairConfig()
        d = self.config.dim
        # Directed conditioning; same weights in both directions. Geometry is a
        # soft cue, not an absolute orientation/metric compatibility filter.
        self.query = nn.Sequential(nn.Linear(2*d+48, d), nn.GELU(), nn.Linear(d, d))
        self.memory_norm = nn.LayerNorm(d)
        self.kv = nn.Linear(d, 2*d)
        self.score = nn.Sequential(nn.Linear(4*d+48, d), nn.GELU(), nn.Linear(d, 1))
        self.bin_score = nn.Parameter(torch.tensor(1.))

    def _validate(self, v):
        k = len(v.descriptors)
        if (v.descriptors.shape != (k, self.config.dim) or v.memory.ndim != 3 or
                v.memory.shape[0] != self.config.dim or v.intervals.shape != (k, 2) or
                v.valid.shape != (k,) or len(v.candidate_ids) != k or v.token_count < 1):
            raise ValueError('invalid opening pair view shape')
        ids = [v.candidate_ids[i] for i in range(k) if v.valid[i]]
        if len(ids) != len(set(ids)) or any(x is None for x in ids):
            raise ValueError('candidate IDs must be unique and valid')
        if not torch.isfinite(v.descriptors[v.valid]).all() or not torch.isfinite(v.memory).all():
            raise ValueError('nonfinite visual features')
        intervals = v.intervals[v.valid]
        if ((intervals < 0) | (intervals >= v.token_count)).any():
            raise ValueError('invalid candidate intervals')
        if (v.geometry is None) != (v.geometry_valid is None):
            raise ValueError('geometry needs an explicit validity mask')
        if v.geometry is not None and (v.geometry.shape != (k, 12) or v.geometry_valid.shape != (k, 12)):
            raise ValueError('invalid geometry shape')

    def _geometry(self, v, indices):
        if v.geometry is None:
            return v.descriptors.new_zeros((len(indices), 24))
        mask = v.geometry_valid[indices] & torch.isfinite(v.geometry[indices])
        values = torch.where(mask, v.geometry[indices], torch.zeros_like(v.geometry[indices]))
        return torch.cat((values, mask.to(values.dtype)), -1).detach()

    def _memory(self, view):
        h, w = view.memory.shape[-2:]
        k, v = self.kv(self.memory_norm(view.memory.flatten(1).T)).chunk(2, -1)
        def heads(x):
            return x.reshape(h*w, self.config.heads, -1).transpose(0, 1)
        return heads(k), heads(v)

    def _retrieve(self, query, kv, target, indices):
        k, v = kv
        heads = self.config.heads
        q = query.reshape(len(query), heads, -1).transpose(0, 1)
        n = target.token_count
        interval = target.intervals[indices]
        width = ((interval[:, 1]-interval[:, 0]) % n + 1).to(query.dtype)/n
        center = (interval[:, 0].to(query.dtype)/n + width/2) % 1
        h, w = target.memory.shape[-2:]
        u = (torch.arange(w, device=query.device, dtype=query.dtype)+.5)/w
        distance = ((u[None]-center[:, None]+.5) % 1-.5).abs()
        # Opening plus half-width on either side; nonzero tails retain context.
        cover = torch.sigmoid((width[:, None]-distance)/(.1*width[:, None]+1/w))
        bias = (cover+1e-6).log()[:, None, :].expand(-1, h, -1).reshape(len(query), h*w)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.config.dim/heads) + bias[None]
        attention = logits.float().softmax(-1).to(query.dtype)
        value = (attention @ v).transpose(0, 1).reshape(len(query), -1)
        return value, attention.mean(0)

    def forward(self, a, b, diagnostics=False):
        self._validate(a); self._validate(b)
        ia, ib = torch.where(a.valid)[0], torch.where(b.valid)[0]
        m, n = len(ia), len(ib)
        result = {'candidate_ids_a': a.candidate_ids, 'candidate_ids_b': b.candidate_ids,
                  'valid_a': a.valid, 'valid_b': b.valid, 'indices_a': ia, 'indices_b': ib,
                  'status': 'no_candidates', 'converged': True, 'iterations': 0,
                  'marginal_error': 0., 'scores': a.descriptors.new_zeros((m, n)),
                  'dustbin_score': self.bin_score}
        if not m or not n:
            result.update(conditional_ab=a.descriptors.new_ones((m, n+1)),
                          conditional_ba=b.descriptors.new_ones((n, m+1)))
            return result
        ka, kb = self._memory(a), self._memory(b)
        ai = ia.repeat_interleave(n); bi = ib.repeat(m)
        scores, queries, attentions_ab, attentions_ba = [], [], [], []
        for start in range(0, m*n, self.config.chunk_size):
            i, j = ai[start:start+self.config.chunk_size], bi[start:start+self.config.chunk_size]
            x, y = a.descriptors[i], b.descriptors[j]
            ga, gb = self._geometry(a, i), self._geometry(b, j)
            qa = self.query(torch.cat((x, y, ga, gb), -1))
            qb = self.query(torch.cat((y, x, gb, ga), -1))
            u, wa = self._retrieve(qa, kb, b, j)
            v, wb = self._retrieve(qb, ka, a, i)
            scores.append(self.score(torch.cat((x+y, (x-y).abs(), u+v, (u-v).abs(),
                                                ga+gb, (ga-gb).abs()), -1)).squeeze(-1))
            if diagnostics:
                queries.append(qa); attentions_ab.append(wa); attentions_ba.append(wb)
        s = torch.cat(scores).reshape(m, n)
        ot = log_partial_transport(s, self.bin_score, self.config.sinkhorn_checks,
                                   self.config.marginal_tolerance)
        lp = ot['log_transport']
        result.update(ot, scores=s, status='assigned' if ot['converged'] else 'numerical_unconverged',
                      log_conditional_ab=lp[:-1]-ot['mu'][:-1, None].log(),
                      log_conditional_ba=lp[:, :-1].T-ot['nu'][:-1, None].log())
        result['conditional_ab'] = result['log_conditional_ab'].exp()
        result['conditional_ba'] = result['log_conditional_ba'].exp()
        if diagnostics:
            result['diagnostics'] = {'queries_ab': torch.cat(queries).reshape(m, n, -1),
                'attention_ab': torch.cat(attentions_ab).reshape(m, n, *b.memory.shape[-2:]),
                'attention_ba': torch.cat(attentions_ba).reshape(m, n, *a.memory.shape[-2:])}
        return result


def assignment_loss(output, target_a, target_b):
    """Targets index original padded candidate arrays; -1 dustbin, -2 unknown.

    Only complete, explicitly supervised rows/columns contribute. Unknown is
    never converted to a dustbin target. Nonconverged samples contribute nothing.
    """
    if output['status'] != 'assigned':
        return None
    losses = []
    for source, target, labels, key in (
            (output['indices_a'], output['indices_b'], target_a, 'log_conditional_ab'),
            (output['indices_b'], output['indices_a'], target_b, 'log_conditional_ba')):
        lookup = {int(v): j for j, v in enumerate(target)}
        for row, original in enumerate(source):
            value = int(labels[int(original)])
            if value == -2:
                continue
            if value < -2 or (value >= 0 and value not in lookup):
                raise ValueError('target references an invalid candidate')
            col = len(target) if value == -1 else lookup[value]
            losses.append(-output[key][row, col])
    return torch.stack(losses).mean() if losses else None


def mutual_matches(output, threshold):
    """Diagnostic candidates or calibrated correspondences, depending on caller."""
    if output['status'] != 'assigned':
        return []
    a, b = output['conditional_ab'], output['conditional_ba']
    m, n = a.shape[0], b.shape[0]
    ja, jb = a.argmax(-1), b.argmax(-1)
    matches = []
    for i in range(m):
        j = int(ja[i])
        if j < n and int(jb[j]) == i:
            confidence = float(torch.minimum(a[i, j], b[j, i]).detach())
            if confidence >= threshold:
                matches.append({'index_a': int(output['indices_a'][i]),
                                'index_b': int(output['indices_b'][j]), 'confidence': confidence})
    return matches
