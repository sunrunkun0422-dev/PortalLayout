#!/usr/bin/env python3
"""Create separate blind and GT packages. Give reviewers the GT package only later."""
import argparse,json,hashlib,sys,shutil,zipfile
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_zind_observable_pairs import write,sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('use a new output directory')
    identity=json.loads((a.build/'build_identity.json').read_text());assert identity['split']=='train','pilot review is train-only'
    root=Path(identity['root']);sampling=json.loads((a.build/'pilot_sampling.json').read_text());ids={r['id'] for r in sampling['samples']};records={}
    for l in (a.build/'main_index.jsonl').open():
        r=json.loads(l)
        if r['id'] in ids:records[r['id']]=r
    dataset_id=sha(a.build/'main_index.jsonl');public=[];gt=[];mapping={};asset_hashes={}
    a.output.mkdir(parents=True);blind=a.output/'stage1_blind';second=a.output/'stage2_gt';(blind/'images').mkdir(parents=True);second.mkdir()
    for sample in sampling['samples']:
        r=records[sample['id']];rid=hashlib.sha256((dataset_id+r['id']).encode()).hexdigest()[:20];entry={'id':rid};mapping[rid]=r['id']
        for side in ['a','b']:
            relative=r['inputs']['image_'+side];name=hashlib.sha256((dataset_id+relative).encode()).hexdigest()[:20]+'.jpg';dest=blind/'images'/name
            if not dest.exists():
                with Image.open(root/relative) as im:im.convert('RGB').resize(tuple(r['inputs']['input_resolution'])).save(dest,quality=92)
            entry['image_'+side]='images/'+name;asset_hashes['images/'+name]=sha(dest)
        public.append(entry);gt.append({**entry,'adjacency':r['labels']['adjacency'],'scope':r['relation_scope'],'gt':r['gt'],
            'conflicts':r['quality']['geometry_conflicts'],'negative_certificate':r['quality']['negative_certificate']})
    shutil.copytree(blind/'images',second/'images')
    for stage,folder,rows in [(1,blind,public),(2,second,gt)]:
        shutil.copyfile(ROOT/'tools/zind_observable_review.html',folder/'index.html')
        (folder/'data.js').write_text('const DATA='+json.dumps({'stage':stage,'dataset_id':dataset_id,'cases':rows},ensure_ascii=False,allow_nan=False)+';\n')
        write(folder/'asset_hashes.json',asset_hashes)
        with zipfile.ZipFile(a.output/f'stage{stage}_package.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(folder.rglob('*')):
                if file.is_file():z.write(file,str(file.relative_to(folder)))
    write(a.output/'coordinator_mapping.json',{'dataset_id':dataset_id,'blind_to_master_id':mapping,'master_index':str((a.build/'main_index.jsonl').resolve()),'pilot_sampling':str((a.build/'pilot_sampling.json').resolve())})
    # This is a structural leak check, not security against access to coordinator files.
    assert all(set(x)=={'id','image_a','image_b'} for x in public)
    (a.output/'README.md').write_text('# 双阶段人工核验材料\n\n先向两位独立审核者分别提供 `stage1_package.zip`，仅含匿名 RGB 与自由框选界面。第一阶段导出后交由协调者归档，之后才提供 `stage2_package.zip`。第二阶段必须导入本人第一阶段 JSON，记录其文件 SHA-256，不修改第一次选择。\n\n不要提前共享 stage2 包或协调者文件；本地静态材料依靠分包发放隔离，不是访问控制服务器。解压后可打开 index.html；若浏览器限制本地 WebCrypto/图片，分别在当前阶段目录使用本地 HTTP 服务。\n\n审核者需填写独立 ID 与依据，可选择无对应或无法确定，也可在任意 RGB 区域框选多组对应。跨接缝区域勾选对应选项，取拖框外侧区间。没有自动填入或伪造的人审记录。普通负样本与困难代理负样本均需复核；几何 positive/negative 仍是 auto_proposed。\n\n实际人审文件回收后使用 `tools/import_zind_observable_reviews.py` 生成独立 reviewed 快照及严格视图。两位审核者分歧应记录第三方裁决，不能默认为一致。\n')
    write(a.output/'verification.json',{'pairs':len(public),'stage1_fields_whitelisted':True,'stage1_gt_present':False,'packages_physically_separate':True,'human_reviews_completed':0,'dataset_id':dataset_id})
    print(f'Prepared {len(public)} anonymous pairs, separate stage1/stage2 packages; no human reviews fabricated.')
if __name__=='__main__':main()
