"""Legacy token cross-attention baseline with the SAME frozen candidates/signals.

Retains legacy bidirectional softmax (not Sinkhorn). Its probabilities require
separate calibration; old thresholds are never imported into the floor pipeline.
"""
from dataclasses import dataclass
import torch
from torch import nn
from portallayout.models.cross_scene_matcher import OpeningGuidedCrossAttentionMatcher


@dataclass(frozen=True)
class LegacyFloorConfig:
    feature_dim: int = 512
    heads: int = 8
    hidden_dim: int = 128
    dropout: float = 0.
    candidate_temperature: float = .2


class LegacyFloorMatcher(nn.Module):
    matcher_type = 'legacy'
    trains_encoder = False

    def __init__(self, config=None):
        super().__init__()
        self.config = config or LegacyFloorConfig()
        self.core = OpeningGuidedCrossAttentionMatcher(**vars(self.config))
        self.core.opening_head.requires_grad_(False)

    def forward(self, a, b, diagnostics=False):
        ia, ib = torch.where(a.valid)[0], torch.where(b.valid)[0]
        m, n = len(ia), len(ib)
        result = dict(candidate_ids_a=a.candidate_ids, candidate_ids_b=b.candidate_ids,
                      indices_a=ia, indices_b=ib, valid_a=a.valid, valid_b=b.valid,
                      status='no_candidates', converged=True, iterations=0, marginal_error=None,
                      assignment_method='legacy_bidirectional_softmax', scores=a.descriptors.new_zeros((m,n)),
                      dustbin_score=self.core.dustbin_score)
        if not m or not n:
            result.update(conditional_ab=a.descriptors.new_ones((m,n+1)), conditional_ba=b.descriptors.new_ones((n,m+1)))
            return result
        def inputs(view, indices):
            features, probability = view.feature_1d, view.opening_probability
            if features is None or probability is None:
                raise ValueError('legacy matching requires frozen token features and probabilities')
            if features.shape != (view.token_count,self.config.feature_dim) or probability.shape != (view.token_count,):
                raise ValueError('legacy frozen feature shape mismatch')
            tokens = torch.arange(view.token_count,device=features.device)[None]
            interval = view.intervals[indices]
            start,end = interval[:,0,None],interval[:,1,None]
            mask = torch.where(start <= end,(tokens >= start)&(tokens <= end),(tokens >= start)|(tokens <= end))
            p = probability.detach()[None]
            # Depth-derived outputs are unused by this adapter; the legacy head
            # is bypassed and cannot change the fixed opening predictions.
            signal = dict(opening_probability=p, opening_logits=torch.logit(p.clamp(1e-6,1-1e-6)),
                          expansion_depth=torch.zeros_like(p), delta=torch.zeros_like(p))
            return features.detach()[None], mask[None], signal
        fa,ma,sa = inputs(a,ia); fb,mb,sb = inputs(b,ib)
        zero = torch.zeros_like(sa['opening_probability'])
        out = self.core(fa,fb,zero,zero,zero,zero,candidate_masks_a=ma,candidate_masks_b=mb,
                        cached_opening_signals=(sa,sb))
        logits = out['candidate_assignment_logits'][0]
        lab = logits[:-1].log_softmax(-1); lba = logits.T[:-1].log_softmax(-1)
        finite = bool(torch.isfinite(lab).all() and torch.isfinite(lba).all())
        result.update(status='assigned' if finite else 'numerical_unconverged',converged=finite,
                      scores=out['candidate_logits'][0],log_conditional_ab=lab,log_conditional_ba=lba,
                      conditional_ab=lab.exp(),conditional_ba=lba.exp())
        if diagnostics: result['diagnostics'] = {'attention_ab':out['Aff_AB'][0],'attention_ba':out['Aff_BA'][0]}
        return result
