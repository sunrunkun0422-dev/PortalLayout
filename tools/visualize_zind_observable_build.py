#!/usr/bin/env python3
"""Real RGB/GT inspection figures, separate from blind review materials."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_zind_observable_pairs import write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);identity=json.loads((a.build/'build_identity.json').read_text());root=Path(identity['root'])
    rows=json.loads((a.audit/'example_records.json').read_text());summary=json.loads((a.build/'summary.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    colors=['#00855b','#9a434a','#395f92','#997521'];inventory=[]
    for i,(category,r) in enumerate(rows.items(),1):
        fig=plt.figure(figsize=(17,10));grid=fig.add_gridspec(2,2,width_ratios=[3.0,1.15],hspace=.14,wspace=.08)
        for j,side in enumerate(['A','B']):
            ax=fig.add_subplot(grid[j,0]);path=root/r['inputs']['image_'+side.lower()]
            with Image.open(path) as im:ax.imshow(im.convert('RGB').resize((1024,512)))
            for k,target in enumerate(r['gt']['correspondences']):
                proj=target[side]['projection'];c=colors[k%len(colors)]
                if proj.get('sample_u') and proj.get('top_v'):
                    u=np.array(proj['sample_u'])*1023;breaks=np.r_[0,np.where(np.abs(np.diff(u))>512)[0]+1,len(u)]
                    for lo,hi in zip(breaks[:-1],breaks[1:]):
                        for edge in ['top_v','bottom_v']:ax.plot(u[lo:hi],np.array(proj[edge][lo:hi])*511,color=c,lw=2)
                    for n in [0,-1]:ax.plot([u[n],u[n]],[proj['top_v'][n]*511,proj['bottom_v'][n]*511],color=c,lw=2)
                    ax.text((proj['start_u']+.5*proj['length_u'])%1*1023,25,f'GT {k+1}',color=c,weight='bold',bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
            ax.set_title(f'RGB {side} + offline GT projection',loc='left',fontsize=12);ax.set_axis_off()
        ax=fig.add_subplot(grid[0,1]);
        for j,poly in enumerate(r['gt']['unit_boundaries_floor']):
            if poly:
                q=np.array(poly);ax.plot(q[:,0],q[:,1],color=['#9a434a','#395f92'][j],lw=1.5,label=['Unit A','Unit B'][j])
        for k,m in enumerate(r['gt']['all_geometry_matches']):
            for key in ['opening_a','opening_b']:
                q=np.array(m[key]['floor_endpoints']);ax.plot(q[:,0],q[:,1],color=colors[k%len(colors)],lw=4)
        ax.set_aspect('equal',adjustable='datalim');ax.set_title('Offline floor geometry',loc='left');ax.legend(fontsize=9);ax.set_xlabel('floor coordinates (not necessarily meters)')
        ax=fig.add_subplot(grid[1,1]);ax.axis('off');lines=[f'House {r["house"]} / {r["floor"]}',r['relation_scope'],f'Adjacency: {r["labels"]["adjacency"]}',f'GT correspondences: {len(r["gt"]["correspondences"])}',f'Auto proxy pass: {r["auto_proposal"]["auto_pass"]}','Human visibility: PENDING','Identifiability: UNREVIEWED','']
        for k,t in enumerate(r['gt']['correspondences']):
            lines.append(f'GT {k+1}:')
            for side in ['A','B']:
                v=t[side];lines.extend([f'  {side}: {v["width_pixels"]:.1f}px / {v["projection"]["angle_deg"]:.1f} deg',f'  Proxy: {v["proxy_visibility"]}'])
            if k>=2:lines.append('(full GT set in JSON / review UI)');break
        if r['quality']['negative_certificate']:lines.extend(['Absence certificate:',f'polygon clearance = {r["quality"]["negative_certificate"]["clearance_m"]:.2f} m'])
        if r['quality']['geometry_conflicts']:lines.extend(['Geometry needs review.','See linked record for all conflicts.'])
        ax.text(0,1,'\n'.join(lines),va='top',fontsize=11,linespacing=1.6)
        fig.suptitle(category.replace('|',' / '),x=.055,ha='left',color='#803b43',fontsize=15)
        fig.text(.06,.025,'GT-assisted inspection only. Annotation geometry cannot confirm door state, furniture occlusion, or RGB identifiability.',fontsize=10,color='#5c5c5c')
        name=f'case_{i:02d}';fig.savefig(a.output/(name+'.png'),dpi=145,bbox_inches='tight');plt.close(fig)
        write(a.output/(name+'.json'),{'selection':'first deterministic index occurrence in diagnostic category; no Matcher used','category':category,'record':r})
        inventory.append({'image':name+'.png','record':name+'.json','category':category,'pair_id':r['id']})
    fig,axes=plt.subplots(1,2,figsize=(13,4.7));scopes=['cross_complete_room','within_complete_partial'];left=np.zeros(2)
    for label,color in [('positive','#91424b'),('negative','#7f8891'),('unknown','#d8d8d8')]:
        values=np.array([summary['relation_scopes'][scope][label] for scope in scopes]);axes[0].barh([0,1],values,left=left,color=color,label=label)
        for j,n in enumerate(values):axes[0].text(left[j]+n/2,j,f'{n/1000:.1f}k',ha='center',va='center',fontsize=9,color='white' if label!='unknown' else '#333')
        left+=values
    axes[0].set_yticks([0,1]);axes[0].set_yticklabels(['Across complete rooms','Within complete room'])
    axes[0].set_xticks([0,50000,100000,150000,200000]);axes[0].set_xticklabels(['0','50k','100k','150k','200k'])
    axes[0].set_xlabel('Selected panorama pairs (correlated within connections)');axes[0].legend(loc='lower right',fontsize=9);axes[0].set_title('Three adjacency states remain separate',loc='left')
    x=summary['reselection'];n=x['positive_connections_compared'];v=[x['primary_nearest_proxy_pass']/n*100,x['reselected_proxy_pass']/n*100]
    axes[1].bar(['Primary-first nearest','Reselected maximin'],v,color=['#8a8a8a','#91424b'],width=.5)
    for i,y in enumerate(v):axes[1].text(i,y+1,f'{y:.2f}%',ha='center',weight='bold')
    axes[1].set_ylim(0,105);axes[1].set_ylabel('Geometry-proxy passing connections (%)');axes[1].set_title(f'View reselection: +{v[1]-v[0]:.2f} pp proxy pass',loc='left')
    fig.text(.09,.015,f'{n:,} fixed positive connections. This is not Matcher accuracy or an exact replay of the old dataset.',fontsize=10,color='#803b43')
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(a.output/'build_summary.png',dpi=170);fig.savefig(a.output/'build_summary.svg');plt.close(fig)
    write(a.output/'inventory.json',inventory)
    cards=''.join(f'<article><h3>{x["category"]}</h3><a href="{x["image"]}"><img src="{x["image"]}"></a><a href="{x["record"]}">完整样本证据 JSON</a></article>' for x in inventory)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>ObservablePairs 离线核验图册</title><style>body{font:17px sans-serif;max-width:1500px;margin:30px auto;color:#333}img{width:100%}article{margin:40px 0;border-top:1px solid #ddd}h1{color:#803b43}</style><h1>ObservablePairs 离线 GT 核验图册</h1><p>仅供第二阶段核验与构建检查；不要提前发给盲审者。所有案例均未人工确认。样本按诊断类别与固定索引顺序选取，没有使用 Matcher 成败筛选。</p><img src="build_summary.png">'+cards)
    print('Created',len(inventory),'real-image diagnostic panels and summary figure.')
if __name__=='__main__':main()
