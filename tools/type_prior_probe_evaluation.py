"""Evaluation and attribution for validate_type_conditioned_prior.py."""
from tools.validate_type_conditioned_prior import *

def predict(model,views,data):
    output={};model.eval()
    with torch.no_grad():
        for k in views:
            d=data[k];z,q,w=model(d['x'],d['h'],d['logits'])
            output[k]=dict(p=z.sigmoid().numpy(),q=q.softmax(-1).numpy() if q is not None else None,w=w.numpy(),
                expansion=(d['expansion']*z.sigmoid()/d['logits'].sigmoid().clamp_min(1e-12)).numpy())
    return output

def instances(pred,views,threshold,iou=.3):
    rows=[];gtcounts=Counter();errors=[]
    for k,v in views.items():
        cs=v['candidates'];gt=[mask(c['tokenInterval']) for c in cs];gtcounts.update(c.get('type','unknown') for c in cs)
        used=set();pr=pred[k]
        for it in opening_probabilities_to_intervals(torch.tensor(pr['p']),threshold,2,12):
            pm=mask(it);ious=[float((pm&g).sum()/max(1,(pm|g).sum())) for g in gt]
            available=[j for j in range(len(cs)) if j not in used]
            j=max(available,key=lambda j:ious[j]) if available else -1;hit=j>=0 and ious[j]>=iou
            typ=('door','opening','neutral')[int(pr['q'][pm].mean(0).argmax())] if pr['q'] is not None else 'untyped'
            if hit:
                used.add(j);target=cs[j]['tokenInterval']
                errors.append(float(np.mean([min(abs(it[i]-target[i]),256-abs(it[i]-target[i])) for i in (0,1)])))
            rows.append(dict(image=k,house=v['house'],start=it[0],end=it[1],score=float(pr['p'][pm].mean()),tp=int(hit),
                gt_type=cs[j].get('type','unknown') if hit else None,pred_type=typ,gt_index=j if hit else None,
                iou=ious[j] if j>=0 else 0.,threshold=float(threshold),iou_threshold=iou))
    tp=sum(r['tp'] for r in rows);total=sum(gtcounts.values());n=len(rows);counts=Counter(r['image'] for r in rows)
    ranked=sorted(rows,key=lambda r:-r['score']);hits=np.cumsum([r['tp'] for r in ranked])
    prec=hits/np.arange(1,len(ranked)+1) if ranked else np.array([])
    env=np.maximum.accumulate(prec[::-1])[::-1] if ranked else prec
    ap=float(sum(env[i] for i,r in enumerate(ranked) if r['tp'])/max(total,1))
    summary=dict(tp=tp,fp=n-tp,fn=total-tp,precision=tp/max(n,1),recall=tp/max(total,1),f1=2*tp/max(n+total,1),
        proposal_AP=ap,endpoint_MAE_tokens=float(np.mean(errors)) if errors else None,
        candidates_mean=n/max(len(views),1),candidates_quantiles=np.quantile([counts[k] for k in views],[0,.25,.5,.75,1]).tolist())
    for typ in ('door','opening'):
        localized=sum(r['tp'] and r['gt_type']==typ for r in rows)
        typed_tp=sum(r['tp'] and r['gt_type']==typ and r['pred_type']==typ for r in rows)
        predicted=sum(r['pred_type']==typ for r in rows)
        summary[typ]=dict(gt=gtcounts[typ],localized=localized,localization_recall=localized/max(gtcounts[typ],1),
            typed_precision=typed_tp/max(predicted,1) if predicted else None,
            typed_recall=typed_tp/max(gtcounts[typ],1) if any(r['pred_type']!='untyped' for r in rows) else None,
            typed_f1=2*typed_tp/max(predicted+gtcounts[typ],1) if any(r['pred_type']!='untyped' for r in rows) else None)
    return rows,summary

def choose_threshold(pred,views,budget):
    candidates=[]
    for t in np.unique(np.r_[np.arange(.1,.901,.05),.47,1.]):
        _,s=instances(pred,views,float(t))
        if s['fp']<=budget:candidates.append((s['recall'],-abs(t-.47),float(t),s['fp']))
    _,_,t,fp=max(candidates);return t,fp

def train_adapter(group,seed,views,data):
    torch.manual_seed(seed);model=PriorAdapter(group)
    keys=[k for k,v in views.items() if v['group']=='fit'];cal=[k for k,v in views.items() if v['group']=='calibration']
    def batch(ks):return [torch.stack([data[k][field] for k in ks]) for field in ('x','h','logits','binary','type')]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.0001);best=None;trace=[];start=time.time()
    for epoch in range(5):
        model.train();order=torch.randperm(len(keys)).tolist()
        for offset in range(0,len(order),8):
            x,h,z,y,t=batch([keys[i] for i in order[offset:offset+8]])
            logits,q,_=model(x,h,z);loss=opening_detection_loss(logits,y)['loss_total']
            if q is not None:loss=loss+.2*F.cross_entropy(q.reshape(-1,3),t.reshape(-1),ignore_index=-100)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        with torch.no_grad():
            x,h,z,y,t=batch(cal);logits,_,_=model(x,h,z);objective=float(opening_detection_loss(logits,y)['loss_total'])
        trace.append(dict(epoch=epoch+1,calibration_loss=objective))
        if best is None or objective<best[0]:best=(objective,copy.deepcopy(model.state_dict()),epoch+1)
    model.load_state_dict(best[1]);model.eval()
    return model,dict(seed=seed,group=group,selected_epoch=best[2],trace=trace,seconds=time.time()-start,
        parameters=sum(p.numel() for p in model.parameters()),strengths=(8*model.strength.sigmoid()).detach().tolist() if hasattr(model,'strength') else None)

def pair_eval(pred,pairs,data,matcher,threshold,mode):
    from tools.train_zind_cross_scene_matcher import predicted_candidate_masks,candidate_targets_from_affinity
    rows=[]
    with torch.no_grad():
        for r in pairs:
            keys=[r['image_'+s] for s in ('a','b')];ds=[data[k] for k in keys];ps=[torch.tensor(pred[k]['p'])[None] for k in keys]
            masks,valid=zip(*[predicted_candidate_masks(p,threshold,2,12) for p in ps]);affinity=torch.zeros(1,256,256)
            for ia,ib in r['supervision']['correspondence_pairs']:
                ma=torch.tensor(mask(r['candidates_a'][ia]['tokenInterval']));mb=torch.tensor(mask(r['candidates_b'][ib]['tokenInterval']))
                affinity[0]+=ma[:,None]*mb[None,:]
            target=candidate_targets_from_affinity(masks[0],affinity,masks[1])>0
            guidance=[d['logits'].sigmoid()[None] for d in ds] if mode=='candidate_only' else ps
            # The existing external-guidance API is named 'gt'; supplied arrays here are ONLY predictions.
            out=matcher(ds[0]['feature'][None],ds[1]['feature'][None],ds[0]['enclosed'][None],ds[0]['extended'][None],
                ds[1]['enclosed'][None],ds[1]['extended'][None],candidate_masks_a=masks[0],candidate_masks_b=masks[1],
                candidate_valid_a=valid[0],candidate_valid_b=valid[1],opening_guidance_a=guidance[0],opening_guidance_b=guidance[1],opening_guidance_mode='gt')
            ia,ib=out['best_candidate_pair'][0].tolist();accepted=ia>=0 and ib>=0;pos=r['supervision']['is_match']
            recalled=bool(target.any()) and pos;correct=bool(target[0,ia,ib]) if accepted and pos else False
            side_recall=[]
            for side,s in enumerate(('a','b')):
                eligible={p[side] for p in r['supervision']['correspondence_pairs']};found=False
                for j in eligible:
                    gm=torch.tensor(mask(r['candidates_'+s][j]['tokenInterval']))
                    inter=(masks[side][0]&gm).sum(-1);union=(masks[side][0]|gm).sum(-1)
                    found=found or bool(((inter/union.clamp_min(1))>=.3).any())
                side_recall.append(found)
            rows.append(dict(id=r['id'],house=r['scene_id'],positive=pos,recalled=recalled,correct=correct,accepted=accepted,
                side_a_recalled=side_recall[0],side_b_recalled=side_recall[1],candidates_a=int(valid[0].sum()),candidates_b=int(valid[1].sum()),
                scope=r['relation_scope'],status=('success' if correct else 'selection_failure' if recalled else 'miss') if pos else ('false_positive' if accepted else 'reject')))
    return rows

def summarize_pairs(rows):
    p=sum(r['positive'] for r in rows);n=len(rows)-p;hit=sum(r['recalled'] for r in rows);correct=sum(r['correct'] for r in rows)
    fp=sum(r['accepted'] and not r['positive'] for r in rows);accepted=sum(r['accepted'] for r in rows)
    return dict(positive=p,negative=n,recalled=hit,correct=correct,R=hit/max(p,1),C=correct/max(hit,1),E=correct/max(p,1),
        negative_rejection=1-fp/max(n,1),negative_fp=fp,accepted=accepted,correct_match_precision=correct/max(accepted,1),
        side_a_recall=sum(r['side_a_recalled'] for r in rows)/max(p,1),side_b_recall=sum(r['side_b_recalled'] for r in rows)/max(p,1))

def experiment(a):
    start=time.time();sel=json.loads((a.output/'selection.json').read_text());views=sel['views'];data={}
    protocol=json.loads((a.output/'protocol.json').read_text())
    assert digest(a.checkpoint)==protocol['checkpoint_sha256'], 'checkpoint changed'
    for k,v in views.items():
        data[k]=torch.load(a.output/'predictions'/(v['cache_key']+'.pt'),map_location='cpu')
        assert data[k]['checkpoint_sha256']==protocol['checkpoint_sha256'], 'stale prediction cache'
        y,t=labels(v['candidates']);data[k]['binary']=torch.tensor(y);data[k]['type']=torch.tensor(t)
    subsets={g:{k:v for k,v in views.items() if v['group']==g} for g in ('fit','calibration','monitor')}
    p,matcher=load_matcher(a.checkpoint);threshold=float(p['opening_probability_threshold']);assert abs(threshold-.47)<1e-6
    output={};allpair={};allpred={};allrows={}
    for seed in (123,456,789):
        for group in ('A0','A1','A2','A3'):
            if group=='A0' and seed!=123:continue
            name='{}_{}'.format(group,seed)
            if group=='A0':model=PriorAdapter(group);training=dict(parameters=0,group=group,seed=seed)
            else:model,training=train_adapter(group,seed,views,data)
            torch.save(model.state_dict(),a.output/(name+'.pt'));pr=predict(model,views,data);allpred[name]=pr
            _,calstats=instances(pr,subsets['calibration'],threshold)
            if group=='A0':budget=calstats['fp']
            calibrated,calfp=choose_threshold(pr,subsets['calibration'],budget)
            item=dict(training=training,calibrated_threshold=calibrated,calibration_FP_budget=budget,calibration_FP=calfp)
            confusion=np.zeros((3,3),int)
            for k in subsets['monitor']:
                if pr[k]['q'] is not None:
                    truth=data[k]['type'].numpy();yp=pr[k]['q'].argmax(-1);good=truth>=0
                    np.add.at(confusion,(truth[good],yp[good]),1)
            f1=2*np.diag(confusion)/np.maximum(1,confusion.sum(0)+confusion.sum(1))
            item['type_confusion']=confusion.tolist();item['type_macro_f1']=float(f1.mean()) if group in ('A2','A3') else None
            yp=np.concatenate([pr[k]['p']>=threshold for k in subsets['monitor']]);yy=np.concatenate([data[k]['binary'].numpy()>0 for k in subsets['monitor']])
            item['token_f1']=float(2*(yp&yy).sum()/max(1,yp.sum()+yy.sum()))
            for setting,t in [('fixed',threshold),('calibrated',calibrated)]:
                item[setting]={}
                for iou in (.3,.5,.7):
                    rows,s=instances(pr,subsets['monitor'],t,iou);item[setting][str(iou)]=s
                    if iou==.3:allrows[name+'_'+setting]=rows
                for mode in ('candidate_only','normal_guidance'):
                    rows=pair_eval(pr,sel['pairs']['monitor'],data,matcher,t,mode);allpair[name+'_'+setting+'_'+mode]=rows
                    item[setting][mode]=summarize_pairs(rows)
            output[name]=item;write(a.output/'results.json',output)
            print(name,json.dumps(item['fixed']['0.3']),flush=True)
    write(a.output/'pair_predictions.json',allpair);write(a.output/'candidate_predictions.json',allrows)
    torch.save(allpred,a.output/'adapter_predictions.pt')
    write(a.output/'experiment_runtime.json',dict(seconds=time.time()-start,device='cpu',peak_cuda_bytes=None))
    diagnostics(a,views,data,allpred,allpair,allrows)

def diagnostics(a,views,data,allpred,allpair,allrows):
    rows=[]
    for k,v in views.items():
        d=data[k]
        for c in v['candidates']:
            m=mask(c['tokenInterval']);rel=d['relative_delta'].numpy();xy=np.array(c['localEndpoints'])
            rows.append(dict(image=k,house=v['house'],group=v['group'],type=c.get('type','unknown'),id=c['id'],
                gt_width_tokens=int(m.sum()),gt_local_distance=float(np.linalg.norm(xy.mean(0))),seam=c['tokenInterval'][0]>c['tokenInterval'][1],
                mean_delta=float(d['delta'].numpy()[m].mean()),mean_relative_delta=float(rel[m].mean()),
                mean_log_ratio=float(d['log_ratio'].numpy()[m].mean()),mean_h=float(d['h'].numpy()[m].mean()),
                mean_logit=float(d['logits'].numpy()[m].mean()),zero_delta_fraction=float((d['delta'].numpy()[m]==0).mean()),
                negative_prior_fraction=float((d['h'].numpy()[m]<.5).mean()),
                global_normalization_attenuation=float(np.mean((rel/(rel+.1))[m]-d['h'].numpy()[m]))))
    with (a.output/'gt_instance_diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    boot={};rng=np.random.RandomState(123)
    for seed in (123,456,789):
        for base in ('A0_123','A1_'+str(seed),'A2_'+str(seed)):
            for setting in ('fixed','calibrated'):
                key='A3_{}_{}_candidate_only'.format(seed,setting);aa=allpair[key];bb=allpair[base+'_'+setting+'_candidate_only']
                houses=sorted({r['house'] for r in aa});by_a={h:[r for r in aa if r['house']==h] for h in houses};by_b={h:[r for r in bb if r['house']==h] for h in houses};diffs=[]
                for _ in range(2000):
                    draw=rng.choice(houses,len(houses),replace=True)
                    x=summarize_pairs([r for h in draw for r in by_a[h]]);y=summarize_pairs([r for h in draw for r in by_b[h]])
                    diffs.append([x[m]-y[m] for m in ('R','E','negative_rejection')])
                boot[key+'_vs_'+base]=dict(houses=len(houses),metrics=['R','E','negative_rejection'],ci95=np.quantile(diffs,[.025,.975],axis=0).tolist())
    write(a.output/'house_bootstrap.json',boot)
    baseline={r['id']:r for r in allpair['A0_123_fixed_candidate_only']}
    write(a.output/'transitions.json',{key:dict(Counter(baseline[r['id']]['status']+' -> '+r['status'] for r in rs)) for key,rs in allpair.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for typ in ('door','opening'):
        vals=[r['mean_h'] for r in rows if r['group']=='monitor' and r['type']==typ];axs[0].hist(vals,bins=np.linspace(0,1,16),alpha=.5,label=typ)
    axs[0].legend();axs[0].set_xlabel('GT interval mean predicted geometry prior h')
    for seed in (123,456,789):
        vals=np.concatenate([p['w'] for k,p in allpred['A3_'+str(seed)].items() if views[k]['group']=='monitor']);axs[1].hist(vals,bins=20,alpha=.4,label=str(seed))
    axs[1].legend();axs[1].set_xlabel('A3 predicted lambda (all monitor tokens)');fig.tight_layout();fig.savefig(a.output/'prior_distributions.png');plt.close(fig)
    cats={'recovered':[r for r in allpair['A3_123_fixed_candidate_only'] if r['correct'] and not baseline[r['id']]['correct']],
        'broken':[r for r in allpair['A3_123_fixed_candidate_only'] if not r['correct'] and baseline[r['id']]['correct']],
        'negative_false_positive':[r for r in allpair['A3_123_fixed_candidate_only'] if not r['positive'] and r['accepted']],
        'opening':[r for r in rows if r['group']=='monitor' and r['type']=='opening'],
        'low_delta_door':[r for r in rows if r['group']=='monitor' and r['type']=='door' and r['mean_relative_delta']<.1],
        'seam':[r for r in rows if r['group']=='monitor' and r['seam']]}
    write(a.output/'case_category_counts.json',{k:len(v) for k,v in cats.items()});ordered=[]
    for typ in ('opening','low_delta_door','seam'):ordered.extend(sorted(cats[typ],key=lambda r:stable(123,r['image'],r['id']))[:4])
    keys=list(dict.fromkeys(r['image'] for r in ordered));remaining=sorted([k for k,v in views.items() if v['group']=='monitor' and k not in keys],key=lambda k:stable(123,k));keys=(keys+remaining)[:12]
    cases=a.output/'cases';cases.mkdir(exist_ok=True)
    for i,k in enumerate(keys):
        v=views[k];d=data[k];fig,ax=plt.subplots(3,1,figsize=(12,7))
        with Image.open(Path(v['data_root'])/k) as im:ax[0].imshow(im.resize((1024,512)))
        ax[0].axis('off');ax[0].set_title(k,fontsize=8)
        for c in v['candidates']:ax[1].plot(np.where(mask(c['tokenInterval']),1.,np.nan),lw=5,label=c['type']+' '+c['id'])
        ax[1].plot(allpred['A0_123'][k]['p'],label='A0');ax[1].plot(allpred['A3_123'][k]['p'],label='A3 seed123');ax[1].axhline(.47,color='k',ls=':');ax[1].legend(fontsize=6,ncol=4)
        ax[2].plot(d['h'],label='geometry h');ax[2].plot(d['relative_delta'],label='relative delta');ax[2].legend();fig.tight_layout();fig.savefig(cases/('{:02d}.png'.format(i+1)));plt.close(fig)
    write(a.output/'case_selection.json',dict(rule='hash seed123, opening/low-delta door/seam then hash fill',images=keys,
        virtual_opening='not manually verified; opening annotation does not certify virtual boundary'))
