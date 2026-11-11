#!/usr/bin/env python3
"""Inventory Markdown, duplicate content and conservative cleanup candidates.

Read-only except for the explicit report directory. Never deletes files.
"""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def walk_files(root):
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != '.git' and not (Path(base) / d).is_symlink())
        for name in sorted(names):
            p = Path(base) / name
            if not p.is_symlink():
                yield p


def moved_path(path, file_moves):
    if path in file_moves:
        return file_moves[path]
    for old, new in [('src/dataset/', 'data/'), ('src/output/', 'outputs/'), ('src/config/', 'configs/layout/')]:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def disposition(path):
    if path.startswith(('experiments/', 'outputs/')):
        return 'retain_experiment_evidence', '保留：与运行、样本或失败记录绑定，不能按同名/模板判冗余'
    if path.startswith(('data/', 'checkpoints/')):
        return 'retain_artifact_contract', '保留：所属数据或权重的使用说明/来源'
    if path.startswith('deliverables/'):
        return 'retain_deliverable', '保留：交付包使用说明或讲稿'
    if path.startswith('docs/archive/'):
        return 'archive', '历史合订本，原日期适用，不作为当前指南'
    return 'retain_documentation', '保留：当前指南、证据包或组件说明'


def inventory(output):
    baseline = read_json(ROOT / 'docs/project/markdown_before.json')
    migration = read_json(ROOT / 'docs/project/document_migration.json').get('merged_sources', {})
    file_moves = read_json(ROOT / 'docs/project/module_migration.json').get('files', {})
    current = {}
    cache_files = []
    metadata_files = []
    for p in walk_files(ROOT):
        if output == p or output in p.parents:
            continue
        rel = p.relative_to(ROOT).as_posix()
        if '__pycache__' in p.parts or '.pytest_cache' in p.parts or p.suffix in ('.pyc', '.pyo'):
            cache_files.append({'path': rel, 'bytes': p.stat().st_size})
        if p.name == '.DS_Store':
            metadata_files.append({'path': rel, 'bytes': p.stat().st_size})
        if p.suffix.lower() != '.md':
            continue
        raw = p.read_bytes()
        text = raw.decode('utf-8', errors='replace')
        current[rel] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                        'title': next((l.lstrip('# ').strip() for l in text.splitlines() if l.startswith('#')), p.name),
                        'text': text}
    rows = []
    covered = set()
    for item in baseline:
        old = item['path']
        dest = migration.get(old)
        if dest:
            path = dest['archive']
            action, reason = 'merged_to_archive', '原文已合并；重复维护入口已移除，使用当前主题指南'
            sha = dest['sha256']
            integrity = 'source_hash_recorded' if sha == item['sha256'] else 'source_changed_before_merge'
            target = path + '#' + dest['anchor']
        else:
            path = moved_path(old, file_moves)
            target = path
            action, reason = disposition(path)
            sha = current.get(path, {}).get('sha256', '')
            integrity = 'byte_identical' if sha == item['sha256'] else 'updated' if sha else 'MISSING'
        covered.add(path)
        rows.append({'original_path': old, 'current_location': target, 'category': action,
                     'recommendation': reason, 'original_bytes': item['bytes'],
                     'original_sha256': item['sha256'], 'integrity': integrity, 'title': item['title']})
    for path, item in current.items():
        if path not in covered:
            action, reason = disposition(path)
            rows.append({'original_path': '', 'current_location': path, 'category': action,
                         'recommendation': reason, 'original_bytes': item['bytes'],
                         'original_sha256': item['sha256'], 'integrity': 'new', 'title': item['title']})
    groups = defaultdict(list)
    for path, item in current.items():
        groups[item['sha256']].append(path)
    duplicates = [{'sha256': sha, 'paths': paths,
                   'decision': '文本相同不等于上下文重复；需保留实验、数据和交付包完整性'}
                  for sha, paths in groups.items() if len(paths) > 1]
    # Import edges are evidence only. Dynamic registry, executable tools and
    # external notebooks are intentionally not labelled dead code.
    sources = list((ROOT / 'portallayout').rglob('*.py')) + list((ROOT / 'tools').rglob('*.py')) + list((ROOT / 'tests').rglob('*.py')) + list(ROOT.glob('*.py'))
    modules = {}
    for p in (ROOT / 'portallayout').rglob('*.py'):
        if p.name == '__init__.py' or p.name.startswith('_'):
            continue
        modules[p.relative_to(ROOT).with_suffix('').as_posix().replace('/', '.')] = p.relative_to(ROOT).as_posix()
    references = defaultdict(set)
    for p in sources:
        tree = ast.parse(p.read_text())
        here = p.relative_to(ROOT).as_posix()
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    package = p.parent.relative_to(ROOT).as_posix().replace('/', '.')
                    base = importlib.util.resolve_name('.' * node.level + base, package)
                names.append(base)
                names.extend(base + '.' + a.name for a in node.names)
            elif isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.append(node.value)
        for module, path in modules.items():
            if here != path and any(name == module or name.startswith(module + '.') or path in name for name in names):
                references[module].add(here)
    candidates = [{'path': path, 'module': module, 'decision': '仅静态无直接引用；检查命令行/动态注册/第三方组件后再决定，未批准删除'}
                  for module, path in modules.items() if not references[module]]
    # Report links in frozen snapshots that still refer to merged source docs.
    historical_links = []
    for path, item in current.items():
        if not path.startswith(('experiments/', 'outputs/', 'data/', 'checkpoints/')):
            continue
        for target in re.findall(r'\]\(([^)\n]+)\)', item['text']):
            clean = target.split('#')[0]
            if not clean or '://' in clean or clean.startswith('/'):
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), clean))
            if resolved in migration:
                dest = migration[resolved]
                historical_links.append({'source': path, 'old_target': target,
                                         'current_target': dest['archive'] + '#' + dest['anchor']})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'markdown_inventory.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {'generated_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
               'baseline_markdown_count': len(baseline), 'current_markdown_count': len(current),
               'merged_source_count': len(migration), 'merged_archive_count': len({d['archive'] for d in migration.values()}),
               'baseline_integrity': dict(Counter(r['integrity'] for r in rows if r['original_path'])),
               'categories': dict(Counter(r['category'] for r in rows)),
               'missing_baseline': [r for r in rows if r['integrity'] == 'MISSING'],
               'cache_bytes': sum(x['bytes'] for x in cache_files), 'cache_file_count': len(cache_files),
               'finder_metadata_bytes': sum(x['bytes'] for x in metadata_files),
               'duplicate_markdown_groups': len(duplicates), 'static_review_candidates': len(candidates),
               'historical_link_redirects': len(historical_links)}
    for name, value in [('summary', summary), ('markdown_inventory', rows), ('duplicate_markdown', duplicates),
                        ('cleanup_candidates', {'regenerable_caches': cache_files, 'finder_metadata': metadata_files,
                                                'static_python_review_only': candidates}),
                        ('historical_link_redirects', historical_links)]:
        (output / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary['missing_baseline'] else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'docs/project/audit')
    args = p.parse_args(argv)
    return inventory(args.output.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
