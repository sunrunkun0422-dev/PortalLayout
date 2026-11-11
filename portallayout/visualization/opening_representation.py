"""RGB-only diagnostics for structure sampling, soft coverage and fusion gates."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from portallayout.datasets.zind_floor_dataset import safe_path, validate_public_manifest
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def _bands(axis, interval, n):
    start, end = interval
    ranges = [(start/n, (end+1)/n)] if start <= end else [(start/n, 1), (0, (end+1)/n)]
    for left, right in ranges:
        axis.axvspan(left, right, color='cyan', alpha=.10)
        axis.axvline(left, color='cyan', linewidth=.8)
        axis.axvline(right, color='cyan', linewidth=.8)


def plot_opening_representation(image, prediction, data, path, title, training_status):
    openings = prediction['openings']
    count = len(openings)
    figure, axes = plt.subplots(max(count, 1), 3, figsize=(18, 3.5 * max(count, 1) + .6), squeeze=False)
    n = len(data.get('source_probability', np.zeros(256)))
    for i in range(max(count, 1)):
        for j in (0, 1):
            axes[i, j].imshow(image, extent=(0, 1, 1, 0), aspect='auto')
            axes[i, j].set(xlim=(0, 1), ylim=(1, 0), xlabel='panorama u', ylabel='panorama v')
        if count == 0:
            axes[i, 0].set_title('No predicted opening; empty descriptors retained')
            axes[i, 1].axis('off')
            axes[i, 2].axis('off')
            break
        opening = openings[i]
        _bands(axes[i, 0], opening['token_interval_inclusive'], n)
        _bands(axes[i, 1], opening['token_interval_inclusive'], n)
        uv = data['diag_sampling_uv'][i]
        valid = data['diag_sampling_valid'][i]
        weights = data['diag_structure_attention'][i]
        sample_sizes = 12 + 65 * weights / max(float(weights.max()), 1e-8)
        colors = np.repeat(np.arange(5), 4)
        axes[i, 0].scatter(uv[valid, 0], uv[valid, 1], c=colors[valid], cmap='viridis',
                           s=sample_sizes[valid], edgecolor='white', linewidth=.5)
        axes[i, 0].set_title('Structure proxy samples (20); size = attention')
        attention = data['diag_context_attention'][i]
        cover = data['diag_soft_cover'][i]
        normalized = attention / max(float(attention.max()), 1e-8)
        axes[i, 1].imshow(normalized, extent=(0, 1, 1, 0), aspect='auto', cmap='magma',
                          alpha=.55, vmin=0, vmax=1, interpolation='bilinear')
        # Coverage is displayed separately in the inset; no learned interpretation is claimed.
        inset = axes[i, 1].inset_axes([.62, .02, .36, .25])
        inset.imshow(cover, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title('soft coverage', fontsize=7, color='white', backgroundcolor='black')
        axes[i, 1].set_title('Context attention (display normalized by maximum)')
        gates = data['diag_fusion_weights'][i]
        axes[i, 2].barh(['structure', 'context', 'geometry'], gates, color=['#218c8d','#3576b6','#d09338'])
        axes[i, 2].set_xlim(0, 1)
        axes[i, 2].set_title('Fusion gates; candidate {}'.format(i))
        available = int(data['diag_geometry_attribute_valid'][i].sum())
        axes[i, 2].set_xlabel('{}\ninterval={}  geometry={}/12 valid\nscore={:.3f} (uncalibrated)'.format(
            opening['candidate_id'], opening['token_interval_inclusive'], available, opening['score']),
            fontsize=8, labelpad=8)
    label = 'INITIALIZED / UNTRAINED' if training_status == 'initialized_untrained' else 'TRAINED CHECKPOINT'
    figure.suptitle('{} | {}\nGeometry samples approximate wall/opening structure, not exact door-frame keypoints'.format(title, label), fontsize=12)
    figure.tight_layout(rect=[0, 0, 1, .92 if count < 2 else .94], h_pad=3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(figure)


def render_cached_representations(source_root, output_root, dataset_root, raw_root, max_observations=20):
    source, output = Path(source_root).resolve(), Path(output_root).resolve()
    report = json.loads((output / 'run_report.json').read_text())
    observed, previews = set(), []
    for floor in report['floors']:
        predictions = json.loads(safe_path(output, floor['representations_file']).read_text())
        manifest = validate_public_manifest(json.loads(safe_path(dataset_root,
            'inputs/' + floor['sample_id'] + '/manifest.json').read_text()))
        rgb = {x['observation_id']: x for x in manifest['observations']}
        for ref in predictions['observations']:
            if ref['status'] != 'encoded' or ref['metadata_file'] in observed or len(previews) >= max_observations:
                continue
            observed.add(ref['metadata_file'])
            record = json.loads(safe_path(output, ref['metadata_file']).read_text())
            source_record = json.loads(safe_path(source, record['source_metadata_file']).read_text())
            if content_hash(safe_path(output, ref['tensor_file'])) != record['tensor_sha256']:
                raise ValueError('refusing to visualize a corrupt representation tensor')
            with np.load(safe_path(output, ref['tensor_file']), allow_pickle=False) as tensors:
                arrays = {k: tensors[k] for k in tensors.files}
            source_tensor_path = safe_path(source, source_record['tensor_file'])
            with np.load(source_tensor_path, allow_pickle=False) as tensors:
                arrays['source_probability'] = tensors['opening_probability'].copy()
            image_path = safe_path(raw_root, rgb[record['observation_id']]['rgb']['relative_path'])
            if content_hash(image_path) != source_record['image_sha256']:
                raise ValueError('RGB image changed since frozen prediction generation')
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert('RGB').resize((1024, 512), Image.BICUBIC)).copy()
            path = output / 'review' / (fingerprint(ref['metadata_file']) + '.png')
            plot_opening_representation(pixels, source_record['prediction'], arrays, path,
                                        record['observation_id'], record['training_status'])
            previews.append({'observation_id': record['observation_id'],
                'image': path.relative_to(output).as_posix(), 'openings': len(record['candidate_ids']),
                'training_status': record['training_status']})
    title = 'Single-view opening representation diagnostics'
    lines = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>' + title + '</title>',
             '<style>body{font:16px sans-serif;margin:24px;background:#f4f6f8}img{width:100%;background:white}section{margin:28px 0}h1{color:#165b64}</style>',
             '<h1>' + title + '</h1><p>Model status: <strong>' + html.escape(report['training_status']) +
             '</strong>. These plots verify implementation and sampling; they do not establish correspondence accuracy.</p>']
    for preview in previews:
        image = html.escape(Path(preview['image']).name)
        lines.append('<section><h2>{} — {} predicted openings</h2><a href="{}"><img src="{}"></a></section>'.format(
            html.escape(preview['observation_id']), preview['openings'], image, image))
    lines.append('</html>')
    index = output / 'review/index.html'
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text('\n'.join(lines), encoding='utf-8')
    write_json_atomic(output / 'review/manifest.json', previews)
    return {'index': 'review/index.html', 'visualized_observations': len(previews)}
