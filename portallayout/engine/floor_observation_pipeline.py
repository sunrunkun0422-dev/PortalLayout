"""RGB-only floor front end: each observation is encoded once, GT stays offline."""
from __future__ import annotations

import hashlib
import io
import json
from importlib import metadata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from portallayout.datasets.zind_floor_dataset import safe_path, validate_public_manifest
from portallayout.models.cross_scene_matcher import (
    OpeningSignalHead, opening_probabilities_to_intervals, resolve_enclosed_extended_depth,
)
from portallayout.engine.cross_scene_estimator import opening_candidates_from_intervals
from portallayout.utils.opening_checkpoint import load_opening_head_checkpoint, resolve_opening_probability_threshold
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic
from portallayout.engine.single_view_prediction import load_bi_layout, prediction_to_local_geometry

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ObservationPredictionSettings:
    branch_order: str = 'extended_first'
    image_height: int = 512
    image_width: int = 1024
    opening_threshold: float = None
    minimum_opening_tokens: int = 2
    maximum_openings: int = 12
    feature_pyramid_level: int = 2
    spatial_height: int = 16
    spatial_width: int = 64

    def __post_init__(self):
        for name in ('image_height', 'image_width', 'minimum_opening_tokens', 'maximum_openings',
                     'feature_pyramid_level', 'spatial_height', 'spatial_width'):
            if type(getattr(self, name)) is not int:
                raise ValueError(name + ' must be an integer')
        if self.branch_order not in ('extended_first', 'enclosed_first'):
            raise ValueError('invalid branch order')
        if (self.image_height, self.image_width) != (512, 1024):
            raise ValueError('this Bi-Layout checkpoint adapter requires 512x1024 input')
        if self.opening_threshold is not None and not 0 <= self.opening_threshold <= 1:
            raise ValueError('opening threshold must be in [0,1]')
        if self.feature_pyramid_level not in range(4):
            raise ValueError('feature pyramid level must be in [0,3]')
        if min(self.minimum_opening_tokens, self.maximum_openings,
               self.spatial_height, self.spatial_width) < 1:
            raise ValueError('proposal and spatial dimensions must be positive')


def frontend_contract(config_path, checkpoint_path, opening_checkpoint, settings, execution_device='cpu'):
    """Content-based identity, portable across paths; GT identity is absent."""
    files = ('portallayout/engine/floor_observation_pipeline.py', 'portallayout/utils/room_feature_cache.py',
             'portallayout/engine/single_view_prediction.py', 'portallayout/models/bi_layout.py',
             'portallayout/models/modules/horizon_net_feature_extractor.py', 'portallayout/models/cross_scene_matcher.py',
             'portallayout/utils/opening_checkpoint.py', 'portallayout/postprocessing/post_process.py',
             'portallayout/engine/cross_scene_estimator.py', 'portallayout/geometry/conversion.py',
             'portallayout/postprocessing/dula/layout.py', 'portallayout/geometry/joint_layout.py')
    return {'schema_version': 'floor-frontend-contract.v1',
            'models': {name: content_hash(path) for name, path in (
                ('bi_config_sha256', config_path), ('bi_weights_sha256', checkpoint_path),
                ('opening_weights_sha256', opening_checkpoint))},
            'settings': asdict(settings),
            'code_sha256': {name: content_hash(REPO_ROOT / name) for name in files},
            'software': {name: metadata.version(name) for name in
                         ('torch', 'torchvision', 'numpy', 'scipy', 'Pillow', 'opencv-python', 'shapely')},
            'execution': {'device': str(execution_device), 'precision': 'float32_no_autocast',
                          'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
                          'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
                          'matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32)},
            'preprocessing': {'resize': 'PIL_BICUBIC', 'input': 'RGB_float32_CHW_0_1',
                              'normalization': 'ImageNet_inside_HorizonNet',
                              'heading_alignment': 'none', 'augmentation': 'none'},
            'geometry': {'frame': 'bi_panorama_local_xz', 'units': 'camera_height_normalized',
                         'metric_scale': 'unknown', 'postprocess': 'panorama_preserving_manhattan'},
            'feature_storage': 'float16', 'depth_storage': 'float32',
            'pretraining_exposure': 'unknown', 'uses_ground_truth': False}


def _opening_records(observation_id, probability, geometry, settings, threshold):
    intervals = opening_probabilities_to_intervals(
        torch.from_numpy(probability), threshold=threshold,
        min_width_tokens=settings.minimum_opening_tokens, max_intervals=settings.maximum_openings)
    mapped, mapping_error = [], None
    points = np.asarray(geometry['polygon_local'], dtype=float)
    if intervals and geometry['valid']:
        # Existing interval mapper consumes x/z directly. Avoid legacy xyz2json,
        # whose default 1.6 height and z reflection change the coordinate frame.
        layout = {'layoutPoints': {'points': [{'xyz': [float(x), 1., float(z)]} for x, z in points]}}
        try:
            mapped = opening_candidates_from_intervals(layout, intervals, probability)
        except (ValueError, RuntimeError, IndexError) as exc:
            mapping_error = str(exc)
    result = []
    for i, (start, end) in enumerate(intervals):
        tokens = np.arange(start, end + 1) if start <= end else np.r_[start:len(probability), 0:end + 1]
        record = {'candidate_id': 'opening_' + fingerprint([observation_id, start, end])[:24],
                  'candidate_index': i, 'token_interval_inclusive': [start, end],
                  'crosses_seam': start > end, 'score': float(probability[tokens].mean()),
                  'is_calibrated_probability': False, 'source': 'trained_opening_head',
                  'geometry_status': 'unavailable', 'wall_index': None, 'endpoints_local': None}
        if mapped:
            c = mapped[i]
            a, b = points[c.wall_index], points[(c.wall_index + 1) % len(points)]
            record.update(geometry_status='mapped_dominant_wall', wall_index=c.wall_index,
                          wall_ratios=[c.start_ratio, c.end_ratio],
                          endpoints_local=[(a + c.start_ratio * (b-a)).tolist(),
                                           (a + c.end_ratio * (b-a)).tolist()],
                          geometry_quality=c.metrics)
        if mapping_error:
            record['geometry_error'] = mapping_error
        result.append(record)
    return result


class BiLayoutObservationEncoder:
    def __init__(self, model, opening_head, settings, threshold, device):
        self.model = model.eval().requires_grad_(False)
        self.opening_head = opening_head.eval().requires_grad_(False)
        self.settings, self.threshold = settings, float(threshold)
        self.device = torch.device(device)

    @classmethod
    def from_checkpoints(cls, config_path, checkpoint_path, opening_checkpoint, settings, device):
        model, report = load_bi_layout(str(config_path), str(checkpoint_path),
                                     torch.device(device), True, strict_checkpoint=True)
        payload = torch.load(str(opening_checkpoint), map_location='cpu')
        args = payload['model']
        head = OpeningSignalHead(**{name: args[name] for name in
            ('feature_dim', 'hidden_dim', 'kernel_size', 'prior_strength', 'prior_relative_scale')})
        opening_report = load_opening_head_checkpoint(
            head, str(opening_checkpoint), expected_feature_dim=model.patch_dim,
            expected_branch_order=settings.branch_order, expected_token_count=model.patch_num,
            bi_layout_config_path=str(config_path), bi_layout_checkpoint_path=str(checkpoint_path))
        if payload['train_cache_contract'].get('image_shape') != [settings.image_height, settings.image_width]:
            raise ValueError('opening checkpoint preprocessing image shape mismatch')
        threshold = resolve_opening_probability_threshold(settings.opening_threshold, opening_report)
        encoder = cls(model, head.to(device), settings, threshold['value'], device)
        encoder.report = {'bi_layout': report, 'opening': opening_report, 'threshold': threshold}
        return encoder

    def __call__(self, observation_ids, images):
        with torch.inference_mode():
            output = self.model(images.to(self.device), return_features=True, return_feature_pyramid=True)
            enclosed, extended = resolve_enclosed_extended_depth(output, self.settings.branch_order)
            opening = self.opening_head(output['layout_feature'], enclosed, extended)
            spatial = output['feature_2d_pyramid'][self.settings.feature_pyramid_level]
            size = (self.settings.spatial_height, self.settings.spatial_width)
            if size[0] > spatial.shape[-2] or size[1] > spatial.shape[-1]:
                raise ValueError('spatial cache may downsample, but must not upsample encoder tensors')
            spatial = F.adaptive_avg_pool2d(spatial, size)
            tensors = {'feature_1d': output['layout_feature'].half().cpu().numpy(),
                       'feature_2d': spatial.half().cpu().numpy(),
                       'depth_enclosed': enclosed.float().cpu().numpy(),
                       'depth_extended': extended.float().cpu().numpy(),
                       'ratio': output['ratio'].float().cpu().numpy(),
                       'opening_logits': opening['opening_logits'].float().cpu().numpy(),
                       'opening_probability': opening['opening_probability'].float().cpu().numpy()}
        predictions = []
        for i, uid in enumerate(observation_ids):
            arrays = {name: value[i] for name, value in tensors.items()}
            if not all(np.isfinite(x).all() for x in arrays.values()):
                raise ValueError('nonfinite model output for ' + uid)
            geometry = {name: prediction_to_local_geometry(arrays['depth_' + name])
                        for name in ('enclosed', 'extended')}
            openings = _opening_records(uid, arrays['opening_probability'], geometry['enclosed'],
                                        self.settings, self.threshold)
            predictions.append((arrays, {
                'coordinate_spec': {'frame': 'bi_panorama_local_xz', 'units': 'camera_height_normalized',
                                    'camera_height_m': None, 'scale_status': 'unknown',
                                    'token_u': '(token_index+0.5)/token_count',
                                    'ray_xz': '[sin(2*pi*(u-0.5)),cos(2*pi*(u-0.5))]'},
                'geometry': geometry, 'height_ratio': float(arrays['ratio'][0]),
                'openings': openings, 'no_opening_detected': not openings,
                'feature_2d': {'source': 'actual_encoder_pyramid_before_height_compression',
                               'pyramid_level': self.settings.feature_pyramid_level,
                               'pooling': 'adaptive_average', 'learned_projection': False}}))
        return predictions


def read_rgb_observation(raw_root, observation, settings):
    data = safe_path(raw_root, observation['rgb']['relative_path']).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as image:
        if list(image.size) != observation['image_size_wh']:
            raise ValueError('image size differs from input manifest')
        image = image.convert('RGB').resize((settings.image_width, settings.image_height), Image.BICUBIC)
        value = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255.)
    return digest, value


def predict_floor_observations(manifests, raw_root, encoder, cache, settings, batch_size=2, *, write_report=True):
    """No GT accessor is called. Failures remain listed in every affected floor."""
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    observations = {}
    for manifest in manifests:
        validate_public_manifest(manifest)
        for observation in manifest['observations']:
            uid = observation['observation_id']
            if uid in observations and observations[uid] != observation:
                raise ValueError('conflicting observation identity: ' + uid)
            observations[uid] = observation
    results, pending = {}, []
    inferred, reused = 0, 0

    def flush():
        nonlocal inferred
        if not pending:
            return
        ids = [x[0] for x in pending]
        output = encoder(ids, torch.stack([x[2] for x in pending]))
        if len(output) != len(pending):
            raise ValueError('encoder output batch size mismatch')
        for (uid, sha, _), (arrays, prediction) in zip(pending, output):
            results[uid] = cache.put(uid, sha, arrays, prediction)
            inferred += 1
        pending.clear()

    for uid, observation in observations.items():
        try:
            sha, image = read_rgb_observation(raw_root, observation, settings)
        except (OSError, ValueError) as exc:
            results[uid] = {'observation_id': uid, 'status': 'input_error', 'reason': str(exc)}
            continue
        cached = cache.get(uid, sha)
        if cached is not None:
            results[uid] = cached
            reused += 1
        else:
            pending.append((uid, sha, image))
            if len(pending) == batch_size:
                flush()
    flush()
    refs = []
    for manifest in manifests:
        record = {'schema_version': 'floor-observation-predictions.v1',
                  'sample_id': manifest['sample_id'], 'floor_uid': manifest['floor_uid'],
                  'contract_fingerprint': cache.fingerprint,
                  'selection_policy': manifest['selection_policy'],
                  'observation_order': manifest['observation_order'],
                  'observations': [results[uid] for uid in manifest['observation_order']],
                  'global_layout_status': 'not_run', 'room_membership': 'unknown'}
        path = 'floors/' + fingerprint(manifest['sample_id']) + '.json'
        write_json_atomic(cache.root / path, record)
        refs.append({'sample_id': manifest['sample_id'], 'floor_uid': manifest['floor_uid'], 'predictions_file': path})
    errors = [x for x in results.values() if x['status'] != 'predicted']
    report = {'status': 'complete' if not errors else 'partial', 'floor_samples': len(manifests),
              'unique_observations': len(observations), 'observations_inferred': inferred,
              'observations_reused': reused, 'input_errors': errors, 'floors': refs,
              'contract_fingerprint': cache.fingerprint,
              'ground_truth_read': False, 'global_layout_status': 'not_run'}
    if write_report:
        write_json_atomic(cache.root / 'run_report.json', report)
    return report
