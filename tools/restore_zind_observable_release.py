"""Restore published master-index shards with exact SHA-256 verification."""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,default=Path('experiments/zind_observable_v1'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output must be a new directory')
 m=json.loads((a.release/'release_manifest.json').read_text());a.output.mkdir(parents=True)
 target=a.output/'main_index.jsonl';temporary=target.with_suffix('.tmp');h=hashlib.sha256();rows=0
 with temporary.open('wb') as f:
  for part in m['ordered_master_shards']:
   path=a.release/part['file'];assert digest(path)==part['sha256'],path
   with gzip.open(path,'rb') as source:
    for line in source:f.write(line);h.update(line);rows+=1
 assert h.hexdigest()==m['master_sha256'] and rows==m['master_rows'],'master integrity mismatch'
 temporary.replace(target)
 for archive in m['other_archives']:
  source=a.release/archive['file'];assert digest(source)==archive['sha256']
  relative=Path(archive['file']);dest=a.output/(relative.name[:-3] if relative.parts[0]=='dataset' else str(relative.with_suffix('')))
  dest.parent.mkdir(parents=True,exist_ok=True)
  with gzip.open(source,'rb') as fi,dest.open('wb') as fo:shutil.copyfileobj(fi,fo)
  assert digest(dest)==archive['uncompressed_sha256']
 for source in (a.release/'dataset').glob('*.json'):shutil.copyfile(source,a.output/source.name)
 shutil.copyfile(a.release/'dataset/all_combinations.jsonl.gz',a.output/'all_combinations.jsonl.gz')
 print('Restored',rows,'master records; SHA-256',h.hexdigest())
if __name__=='__main__':main()
