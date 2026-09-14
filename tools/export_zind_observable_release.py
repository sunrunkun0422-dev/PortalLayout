import gzip,hashlib,json,shutil,csv
from pathlib import Path
root=Path.cwd();out=root/'experiments/zind_observable_v1';build=root/'src/dataset/ZInD-ObservablePairs-v1/builds/train_auto_v2';training=root/'src/dataset/ZInD-ObservablePairs-v1/training/best1_balanced_seed123';run=root/'checkpoints/End_To_End/zind_observable_v1/best1_balanced_seed123_20260910'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
def compress(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 with src.open('rb') as fi,dst.open('wb') as fo:
  with gzip.GzipFile(fileobj=fo,mode='wb',filename='',mtime=0) as gz:shutil.copyfileobj(fi,gz)
 return {'file':str(dst.relative_to(out)),'sha256':sha(dst),'uncompressed_sha256':sha(src),'bytes':dst.stat().st_size}
parts=[];combined=hashlib.sha256();handle=None;count=0;idx=0
folder=out/'dataset/master_shards';folder.mkdir(parents=True,exist_ok=True)
with (build/'main_index.jsonl').open('rb') as f:
 for line in f:
  if count%10000==0:
   if handle:handle.close();stream.close();parts.append({'file':str(path.relative_to(out)),'sha256':sha(path),'rows':10000})
   idx+=1;path=folder/f'part-{idx:03d}.jsonl.gz';stream=path.open('wb');handle=gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0)
  handle.write(line);combined.update(line);count+=1
handle.close();stream.close();parts.append({'file':str(path.relative_to(out)),'sha256':sha(path),'rows':count%10000 or 10000})
archives=[compress(build/'connections.jsonl',out/'dataset/connections.jsonl.gz')]
copy(build/'all_combinations.jsonl.gz',out/'dataset/all_combinations.jsonl.gz')
for name in ['summary.json','build_identity.json','config.json','source_hashes.json','data_issues.json','pilot_sampling.json','incremental_provenance.json']:
 copy(build/name,out/'dataset'/name)
for s in ['train','val']:
 data=json.loads((training/f'{s}.json').read_text())
 dest=out/'training'/f'{s}_ids.json';dest.parent.mkdir(parents=True,exist_ok=True)
 dest.write_text(json.dumps({'split':s,'protocol':data['datasetContract']['evaluation_protocol'],'source_manifest_sha256':sha(training/f'{s}.json'),'ordered_pair_ids':[r['id'] for r in data['pairs']]},indent=2)+'\n')
 del data
copy(training/'summary.json',out/'training/selection_summary.json')
for name in ['best_metrics.json','latest_metrics.json','metrics.jsonl']:copy(run/name,out/'results'/name)
job=root/'src/output/zind_observable_training_20260910'
for name in ['gradient_preflight.json','job_config.json','code_hashes.json','COMPLETED.json','adapter_tests.log','训练结果摘要.md']:copy(job/name,out/'results'/name)
for name in ['audit.json','coordinate_contract.json','delivery_check.json','human_review_status.json']:copy(root/'src/output/zind_observable_pairs_v1/audit'/name,out/'audit'/name)
copy(root/'src/output/zind_observable_pairs_v1/figures/build_summary.png',out/'figures/build_summary.png')
weights={n:{'local_path':str((run/n).relative_to(root)),'sha256':sha(run/n),'bytes':(run/n).stat().st_size,'uploaded':False} for n in ['best.pt','last.pt']}
release={'dataset':'ZInD-ObservablePairs-v1','build':'train_auto_v2','master_rows':count,'master_sha256':combined.hexdigest(),'ordered_master_shards':parts,'other_archives':archives,'weights':weights,'original_rgb_uploaded':False,'human_verified_count':0,'official_val_test_used':False}
(out/'release_manifest.json').write_text(json.dumps(release,ensure_ascii=False,indent=2)+'\n')
with (out/'results/epochs.csv').open('w',newline='') as f:
 w=csv.writer(f);keys=['end_to_end_candidate_top1','positive_candidate_recall','conditional_candidate_top1','negative_rejection_rate','harmonic_flow_score','opening_token_f1'];w.writerow(['epoch']+keys)
 for line in (run/'metrics.jsonl').open():
  d=json.loads(line);m=d['validation']['predicted_openings'];w.writerow([d['epoch']+1]+[m[k] for k in keys])
print('Exported',count,'rows in',len(parts),'shards; bytes',sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
