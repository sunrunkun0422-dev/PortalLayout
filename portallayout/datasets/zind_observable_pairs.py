"""Independent ZInD observable-pair construction; GT never enters RGB inputs.

Geometry labels and visibility proxies are annotation-derived, not human review.
Missing metric scale remains missing. No Matcher/torch/checkpoint dependency.
"""
from __future__ import annotations
import hashlib
import itertools
import math
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, Point, LineString
from portallayout.datasets.zind_pair_mining import zind_transform_matrix, transform_points


def stable_id(*parts):
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:24]


def finite_positive(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def endpoint_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    direct = np.linalg.norm(a-b, axis=1).mean()
    reverse = np.linalg.norm(a-b[::-1], axis=1).mean()
    return float(min(direct, reverse)), bool(reverse < direct)


def projection(segment, heights=None):
    """Continuous normalized spherical coordinates; longitude convention = official.

    Intervals use [start, start+length] on a circle, not sorted endpoints.
    Pixel conversion uses (W-1,H-1), consistent with ZInD's renderer.
    """
    p = np.asarray(segment, dtype=float)
    u = (np.arctan2(-p[:, 0], p[:, 1])/(2*np.pi)+.5) % 1
    delta = (u[1]-u[0]+.5) % 1-.5
    start = float((u[0] if delta >= 0 else u[1]) % 1)
    length = float(abs(delta))
    out = {'start_u': start, 'length_u': length, 'wraps_seam': start+length > 1,
           'endpoint_u': u.tolist(), 'angle_deg': length*360}
    if heights is not None:
        # ZInD stores bottom/top z relative to the panorama camera center.
        samples = np.linspace(p[0], p[1], 17)
        dist = np.linalg.norm(samples, axis=1)
        out['top_v'] = (.5-np.arctan2(heights[1], dist)/np.pi).tolist()
        out['bottom_v'] = (.5-np.arctan2(heights[0], dist)/np.pi).tolist()
        out['sample_u'] = ((np.arctan2(-samples[:, 0], samples[:, 1])/(2*np.pi)+.5) % 1).tolist()
    return out


def circular_iou(a, b):
    def spans(d):
        s, n = float(d['start_u']) % 1, float(d['length_u'])
        if not 0 < n <= 1: return []
        return [(s, min(1, s+n))]+([(0, s+n-1)] if s+n > 1 else [])
    aa, bb = spans(a), spans(b)
    inter = sum(max(0, min(x1,y1)-max(x0,y0)) for x0,x1 in aa for y0,y1 in bb)
    union = sum(x1-x0 for x0,x1 in aa+bb)-inter
    return inter/union if union > 0 else 0.


def polygon(vertices):
    try:
        a = np.asarray(vertices, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all(): return None
        p = Polygon(a)
        return p if p.is_valid and p.area > 1e-8 else None
    except (ValueError, TypeError): return None


def parse_interfaces(layout, matrix, ref):
    records, errors = [], []
    for kind in ['doors','openings','windows']:
        entries = layout.get(kind)
        if not isinstance(entries, list) or len(entries) % 3:
            errors.append(ref+'/'+kind+':missing_or_invalid_triplets'); continue
        for i in range(0,len(entries),3):
            try:
                xy=np.asarray(entries[i:i+2],dtype=float); z=np.asarray(entries[i+2],dtype=float)
                if xy.shape!=(2,2) or z.shape!=(2,) or not np.isfinite(xy).all() or not np.isfinite(z).all() or z[1]<=z[0] or np.linalg.norm(xy[1]-xy[0])<1e-6: raise ValueError()
                records.append({'kind':kind[:-1], 'local_endpoints':xy.tolist(),
                    'floor_endpoints':transform_points(xy,matrix).tolist(), 'z_bottom_top':z.tolist(),
                    'source':ref+'/'+kind+'/'+str(i//3)})
            except (ValueError,TypeError): errors.append(ref+'/'+kind+'/'+str(i//3)+':invalid_geometry')
    return records, errors


def image_quality(path, cfg):
    try:
        with Image.open(path) as im:
            size=im.size
            im.draft('RGB',(cfg['quality_thumbnail_width'],cfg['quality_thumbnail_width']//2))
            im.thumbnail((cfg['quality_thumbnail_width'],cfg['quality_thumbnail_width']//2))
            a=np.asarray(im.convert('L'),dtype=float)
        extreme=float(((a<cfg['image_extreme_low'])|(a>cfg['image_extreme_high'])).mean()); std=float(a.std())
        return {'decoded':True,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'size':list(size),'aspect_ratio_valid':size[0]==2*size[1],
                'gray_std':std,'extreme_pixel_fraction':extreme,
                'edge_energy':float(np.abs(np.diff(a,axis=1)).mean()),
                'proxy_pass':bool(size[0]==2*size[1] and std>=cfg['minimum_gray_std'] and extreme<=cfg['maximum_extreme_pixel_fraction']),
                'pixel_occlusion_verified':False}
    except (OSError,ValueError) as e: return {'decoded':False,'proxy_pass':False,'error':str(e)}


def load_house(payload, house, root, cfg):
    """Keep all localized primary AND secondary views, including missing layouts."""
    floors={}; errors=[]; counts=Counter()
    for fid, rooms in payload.get('merger',{}).items():
        # Bare pano IDs are not always unique in a floor (real example: 0363).
        # Disambiguate collisions by complete/partial path before any lookup or
        # cache insertion. Keep distinct annotated images/poses, never overwrite.
        pano_id_counts=Counter(pid for prs in rooms.values() for pns in prs.values() for pid in pns)
        duplicate_ids={pid for pid,n in pano_id_counts.items() if n>1}
        if duplicate_ids:
            errors.append({'floor':fid,'duplicate_bare_pano_ids':sorted(duplicate_ids),'action':'qualified_with_complete_and_partial_room'})
            counts['duplicate_bare_pano_ids']+=len(duplicate_ids)
        metric=payload.get('scale_meters_per_coordinate',{}).get(fid)
        metric=float(metric) if finite_positive(metric) else None
        floor={'metric_scale':metric,'panos':{},'units':[]}; floors[fid]=floor
        complete={}; partial={}
        for cr, prs in sorted(rooms.items()):
            for pr, pns in sorted(prs.items()):
                for pid, raw in sorted(pns.items()):
                    counts['panoramas']+=1; counts['primary' if raw.get('is_primary') else 'secondary']+=1
                    ref=f'merger/{fid}/{cr}/{pr}/{pid}'
                    try:
                        t=raw['floor_plan_transformation']
                        if not finite_positive(t['scale']): raise ValueError('invalid scale')
                        mat=zind_transform_matrix(t)
                        if not np.isfinite(mat).all(): raise ValueError('invalid matrix')
                        path=(Path(root)/house/raw['image_path']).resolve()
                        path.relative_to(Path(root).resolve())
                    except (KeyError,ValueError,TypeError) as e:
                        errors.append({'ref':ref,'error':str(e)}); counts['invalid_panoramas']+=1;continue
                    qualified_pid=cr+'/'+pr+'/'+pid if pid in duplicate_ids else pid
                    pano={'id':qualified_pid,'complete':cr,'partial':pr,'source':ref,'matrix':mat,
                          'raw':raw,'image':str(path),'image_relative':str(path.relative_to(root)),
                          'primary':bool(raw.get('is_primary')),'quality':image_quality(path,cfg)}
                    floor['panos'][qualified_pid]=pano;complete.setdefault(cr,[]).append(pano);partial.setdefault((cr,pr),[]).append(pano)
                    for name in ['layout_raw','layout_visible','layout_complete']:counts[name+'_present']+=name in raw
                    if not pano['quality']['decoded']:counts['unreadable_images']+=1
        for cr,panos in sorted(complete.items()):
            floor['units'].append(make_unit(cr,cr,None,panos,'layout_complete',metric,cfg))
        for (cr,pr),panos in sorted(partial.items()):
            floor['units'].append(make_unit(cr+'/'+pr,cr,pr,panos,'layout_raw',metric,cfg))
        counts['missing_metric_floors']+=metric is None
    return floors,errors,dict(counts)


def make_unit(uid, complete, partial, panos, source, metric, cfg):
    panos=sorted(panos,key=lambda p:(not p['primary'],p['id']))
    unit={'id':uid,'complete':complete,'partial':partial,'panos':panos,'layout_source':source,
          'openings':[],'issues':[],'polygon':None,'inventory_complete':True}
    counts=[];inventory=[]
    for pano in panos:
        layout=pano['raw'].get(source)
        if not isinstance(layout,dict):
            unit['issues'].append(pano['id']+':missing_'+source);unit['inventory_complete']=False;continue
        poly=polygon(layout.get('vertices'))
        if poly is None:unit['issues'].append(pano['id']+':invalid_polygon');unit['inventory_complete']=False;continue
        pg=polygon(transform_points(poly.exterior.coords[:-1],pano['matrix']))
        if unit['polygon'] is None:unit['polygon']=pg
        elif unit['polygon'].intersection(pg).area/max(unit['polygon'].union(pg).area,1e-9)<cfg['minimum_polygon_annotation_iou']:
            unit['issues'].append(pano['id']+':polygon_disagreement');unit['inventory_complete']=False
        if not pg.buffer(cfg['camera_membership_tolerance_m']/metric if metric else 0).covers(Point(pano['matrix'][:2,2])) or pano['raw'].get('is_inside') is not True:
            unit['issues'].append(pano['id']+':camera_membership_unconfirmed');unit['inventory_complete']=False
        its,err=parse_interfaces(layout,pano['matrix'],pano['source']+'/'+source)
        unit['issues']+=err
        if err:unit['inventory_complete']=False
        counts.append(sum(x['kind']!='window' for x in its))
        present=set()
        for it in its:
            if it['kind']=='window':continue
            xy=np.array(it['floor_endpoints']);width=np.linalg.norm(xy[1]-xy[0]); found=None
            for j,old in enumerate(unit['openings']):
                d,_=endpoint_error(xy,old['floor_endpoints'])
                relative=d/max(width,1e-6)
                if old['kind']==it['kind'] and relative<=cfg['dedup_relative_tolerance'] and (metric is None or d*metric<=cfg['dedup_tolerance_m']):found=j;break
            if found is None:
                found=len(unit['openings']);it={**it,'id':f'{uid}/o{found}','sources':[], 'height_floor_z':None}
                # Height above the floor, still in floor-coordinate units.
                h=pano['raw'].get('camera_height')
                if finite_positive(h):it['height_floor_z']=((np.array(it['z_bottom_top'])+h)*float(pano['raw']['floor_plan_transformation']['scale'])).tolist()
                unit['openings'].append(it)
            unit['openings'][found]['sources'].append({'ref':it['source'],'pano':pano['id']})
            present.add(found)
        inventory.append(present)
    if counts and (min(counts)!=max(counts) or any(len(x)!=len(unit['openings']) for x in inventory)):
        unit['issues'].append('interface_inventory_disagreement');unit['inventory_complete']=False
    if not counts:unit['inventory_complete']=False
    return unit


def spatial_relation(a,b,segment,metric,cfg):
    pa,pb=a['polygon'],b['polygon'];seg=np.asarray(segment)
    if pa is None or pb is None:return {'valid':False,'reason':'invalid_polygon'}
    tang=seg[1]-seg[0];norm=np.array([-tang[1],tang[0]])/max(np.linalg.norm(tang),1e-9)
    mid=seg.mean(axis=0);step=cfg['side_probe_m']/metric
    p,q=Point(mid+norm*step),Point(mid-norm*step)
    opposite=(pa.covers(p) and not pb.covers(p) and pb.covers(q) and not pa.covers(q)) or (pa.covers(q) and not pb.covers(q) and pb.covers(p) and not pa.covers(p))
    boundary=max(pa.boundary.distance(Point(x)) for x in seg)*metric
    boundary=max(boundary,max(pb.boundary.distance(Point(x)) for x in seg)*metric)
    overlap=pa.intersection(pb).area/max(min(pa.area,pb.area),1e-9)
    return {'valid':bool(opposite and boundary<=cfg['boundary_tolerance_m'] and overlap<=cfg['max_polygon_overlap_ratio']),
            'opposite_local_interior_sides':bool(opposite),'boundary_error_m':float(boundary),'polygon_overlap_ratio':float(overlap)}


def geometry_relation(a,b,metric,cfg):
    """Confirm conjunction of geometric tests; absences alone never imply negative."""
    pairs=[];conflicts=[];nearest=None
    if metric is None:return {'adjacency':'unknown','matches':[],'conflicts':['missing_metric_scale'],'negative_certificate':None}
    for ia,ib in itertools.product(a['openings'],b['openings']):
        xa,xb=np.array(ia['floor_endpoints']),np.array(ib['floor_endpoints'])
        d,rev=endpoint_error(xa,xb);dm=d*metric;nearest=dm if nearest is None else min(nearest,dm)
        if dm>cfg['near_conflict_m']:continue
        va,vb=xa[1]-xa[0],xb[1]-xb[0];wa,wb=np.linalg.norm(va),np.linalg.norm(vb)
        angle=math.degrees(math.acos(float(np.clip(abs(np.dot(va,vb))/(wa*wb),-1,1))))
        rel=abs(wa-wb)/max(wa,wb);sp=spatial_relation(a,b,(xa+(xb[::-1] if rev else xb))/2,metric,cfg)
        ha,hb=ia['height_floor_z'],ib['height_floor_z']
        hz=None if ha is None or hb is None else max(0,min(ha[1],hb[1])-max(ha[0],hb[0]))/max(1e-9,min(ha[1]-ha[0],hb[1]-hb[0]))
        rec={'opening_a':ia,'opening_b':ib,'endpoint_error_m':dm,'reversed_b':rev,
             'width_a_m':float(wa*metric),'width_b_m':float(wb*metric),'width_relative_error':float(rel),
             'direction_error_deg':angle,'height_overlap_ratio':hz,'spatial':sp}
        good=dm<=cfg['endpoint_tolerance_m'] and rel<=cfg['width_relative_tolerance'] and angle<=cfg['direction_tolerance_deg'] and sp['valid'] and ia['kind']==ib['kind'] and hz is not None and hz>=cfg['minimum_height_overlap_ratio']
        if good:pairs.append(rec)
        else:conflicts.append({'a':ia['id'],'b':ib['id'],'reason':'near_or_incompatible_geometry', 'evidence':{k:v for k,v in rec.items() if k not in ['opening_a','opening_b']}})
    ac=Counter(p['opening_a']['id'] for p in pairs);bc=Counter(p['opening_b']['id'] for p in pairs)
    if any(n>1 for n in list(ac.values())+list(bc.values())):conflicts.append({'reason':'one_to_many_geometry_correspondence'})
    if not a['inventory_complete'] or not b['inventory_complete']:conflicts.append({'reason':'annotation_inventory_or_polygon_inconsistent'})
    if conflicts:label='unknown'
    elif pairs:label='positive'
    else:label='unknown'
    cert=None
    if not pairs and not conflicts and a['polygon'] is not None and b['polygon'] is not None:
        clearance=a['polygon'].distance(b['polygon'])*metric
        if clearance>=cfg['negative_polygon_clearance_m']:
            label='negative';cert={'kind':'disjoint_complete_annotation_polygons_with_clearance', 'clearance_m':float(clearance),
                'exhaustive_interface_inventory_consistent':True,'shared_boundary_impossible_under_annotation_geometry':True,
                'human_verified':False}
        else:conflicts.append({'reason':'absence_of_match_not_proof_of_nonadjacency','polygon_clearance_m':float(clearance)})
    return {'adjacency':label,'matches':pairs,'conflicts':conflicts,'nearest_interface_error_m':nearest,'negative_certificate':cert}


def ray_clear_fraction(camera,segment,polygon_vertices,internal,contact,n):
    """2D annotated-wall proxy: hits at target wall within tolerance are permitted.
    Endpoint rays included; foreground wall/internal hits are occluders.
    """
    camera=np.asarray(camera,dtype=float);segment=np.asarray(segment,dtype=float)
    edges=[]
    for ring in [polygon_vertices]+list(internal or []):
        arr=np.asarray(ring,dtype=float)
        if arr.ndim!=2 or arr.shape[1]!=2 or len(arr)<3 or not np.isfinite(arr).all():return None
        edges.extend(zip(arr,np.roll(arr,-1,axis=0)))
    if not edges:return None
    starts=np.array([x[0] for x in edges]);diff=np.array([x[1]-x[0] for x in edges])
    result=[]
    for target in np.linspace(segment[0],segment[1],n):
        vec=target-camera;dist=np.linalg.norm(vec)
        if dist<=contact:result.append(False);continue
        rhs=starts-camera;den=vec[0]*diff[:,1]-vec[1]*diff[:,0]
        valid=np.abs(den)>1e-12;ts=np.zeros(len(edges));us=np.zeros(len(edges))
        ts[valid]=(rhs[valid,0]*diff[valid,1]-rhs[valid,1]*diff[valid,0])/den[valid]
        us[valid]=(rhs[valid,0]*vec[1]-rhs[valid,1]*vec[0])/den[valid]
        block=valid&(ts>1e-8)&(ts<1-contact/dist)&(us>=-1e-8)&(us<=1+1e-8)
        # Collinear foreground segment must not disappear through denominator mask.
        col=(~valid)&(np.abs(rhs[:,0]*vec[1]-rhs[:,1]*vec[0])<1e-10)
        proj0=np.dot(rhs,vec)/(dist*dist);proj1=np.dot(starts+diff-camera,vec)/(dist*dist)
        colblock=col&(np.minimum(proj0,proj1)<1-contact/dist)&(np.maximum(proj0,proj1)>contact/dist)
        result.append(not bool(block.any() or colblock.any()))
    return {'samples_clear':result,'fraction':float(np.mean(result)),'endpoints_clear':[result[0],result[-1]],'pixel_visibility_verified':False}


def opening_visibility(pano,opening,metric,cfg):
    local=transform_points(opening['floor_endpoints'],np.linalg.inv(pano['matrix']))
    h=opening['height_floor_z'];height=None
    if h is not None and finite_positive(pano['raw'].get('camera_height')):
        height=np.array(h)/float(pano['raw']['floor_plan_transformation']['scale'])-pano['raw']['camera_height']
    proj=projection(local,height);width=proj['length_u']*(cfg['input_width']-1)
    mid=local.mean(axis=0);v=local[1]-local[0]
    incidence=abs(np.cross(v,mid))/(max(np.linalg.norm(v)*np.linalg.norm(mid),1e-9))
    scale=float(pano['raw']['floor_plan_transformation']['scale'])
    distance=None if metric is None else float(np.linalg.norm(mid)*scale*metric)
    complete=pano['raw'].get('layout_complete');rays=None
    if isinstance(complete,dict) and polygon(complete.get('vertices')) is not None:
        contact=max(np.linalg.norm(mid)*cfg['ray_contact_relative_tolerance'],cfg['ray_contact_tolerance_m']/(scale*metric) if metric is not None else 0)
        rays=ray_clear_fraction([0,0],local,complete['vertices'],complete.get('internal',[]),contact,cfg['ray_samples'])
    visible=pano['raw'].get('layout_visible');support=None;support_kind='missing_visible_layout'
    if isinstance(visible,dict):
        its,_=parse_interfaces(visible,pano['matrix'],pano['source']+'/layout_visible')
        matches=[LineString(t['floor_endpoints']) for t in its if t['kind']==opening['kind']]
        samples=np.linspace(*np.asarray(opening['floor_endpoints']),cfg['ray_samples'])
        if metric is not None:
            tol=cfg['visible_support_tolerance_m']/metric
            support=float(np.mean([any(l.distance(Point(q))<=tol for l in matches) for q in samples]));support_kind='visible_interface_sample_support'
            # Raw virtual openings may disappear from complete/visible interfaces.
            # Use derived visible footprint support, explicitly weaker provenance.
            vp=polygon(visible.get('vertices'))
            if opening['kind']=='opening' and vp is not None:
                loc_samples=transform_points(samples,np.linalg.inv(pano['matrix']))
                footprint=float(np.mean([vp.buffer(tol/scale).covers(Point(q)) for q in loc_samples]))
                if footprint>support:support=footprint;support_kind='derived_visible_footprint_for_virtual_opening'
    reasons=[]
    checks={'pixel_width':width>=cfg['minimum_width_pixels'], 'angle':proj['angle_deg']>=cfg['minimum_angle_deg'],
        'incidence':float(incidence)>=cfg['minimum_incidence_cosine'], 'distance':None if distance is None else distance<=cfg['maximum_distance_m'],
        'rays':None if rays is None else rays['fraction']>=cfg['minimum_ray_clear_fraction'],
        'endpoints':None if rays is None else all(rays['endpoints_clear']),
        'visible_support':None if support is None else support>=cfg['minimum_visible_support_fraction'],
        'image':pano['quality']['proxy_pass']}
    for key,value in checks.items():
        if value is not True:reasons.append(key+('_unknown' if value is None else '_failed'))
    state='clear' if not reasons else ('uncertain' if any(x is None for x in checks.values()) else ('not_visible' if rays and rays['fraction']==0 else 'partial'))
    vals={'pixel_width':min(1,width/(cfg['minimum_width_pixels']*cfg['score_width_saturation_multiplier'])), 'incidence':float(incidence),
          'rays':rays['fraction'] if rays else 0., 'visible_support':support or 0.,'image':float(pano['quality']['proxy_pass'])}
    score=sum(vals[k]*w for k,w in cfg['quality_weights'].items())/sum(cfg['quality_weights'].values())
    return {'projection':proj,'width_pixels':float(width),'distance_m':distance,'incidence_cosine':float(incidence),
            'ray_proxy':rays,'visible_support_fraction':support,'visible_support_kind':support_kind,
            'proxy_visibility':state,'proxy_pass':not reasons,'reasons':reasons,'quality_score':score,
            'endpoints_observation_proxy':checks['endpoints'],'pixel_visibility':'uncertain',
            'door_open_closed':'unknown','furniture_occlusion':'unknown','local_endpoints':local.tolist()}


def public_input(record):
    """Whitelist: safe RGB-only loader output has no GT, unit names or pose."""
    return {'id':record['id'],'input_configuration':'rgb_pair', 'image_a':record['inputs']['image_a'],
            'image_b':record['inputs']['image_b'],'input_resolution':record['inputs']['input_resolution']}


def select_combinations(combos,cfg):
    remaining=list(combos);selected=[];seen_a=set();seen_b=set()
    while remaining and len(selected)<cfg['max_pairs_per_connection']:
        # Maximize worst side first; complementary views break near-quality ties.
        best=min(remaining,key=lambda c:(-int(c['auto_pass']),-(c['worst_quality']+cfg['view_diversity_weight']*((c['pano_a'] not in seen_a)+(c['pano_b'] not in seen_b))),-c['mean_quality'],c['id']))
        selected.append(best);remaining.remove(best);seen_a.add(best['pano_a']);seen_b.add(best['pano_b'])
    ids={r['id'] for r in selected}
    for c in combos:
        c['selected']=c['id'] in ids
        c['selection_reason']='maximin_quality_with_view_complementarity' if c['selected'] else 'outside_configured_connection_budget'
    return selected


def build_floor(house,fid,floor,split,root,cfg):
    metric=floor['metric_scale'];all_units=floor['units'];vcache={}
    for a,b in itertools.combinations(all_units,2):
        if a['partial'] is None and b['partial'] is None:scope='cross_complete_room'
        elif a['partial'] is not None and b['partial'] is not None and a['complete']==b['complete']:scope='within_complete_partial'
        else:continue
        cid=stable_id(house,fid,scope,a['id'],b['id']);geo=geometry_relation(a,b,metric,cfg)
        combos=[]
        for pa,pb in itertools.product(a['panos'],b['panos']):
            pid=stable_id(cid,pa['id'],pb['id']);vis=[]
            for target in geo['matches']:
                sides=[]
                for p,o in [(pa,target['opening_a']),(pb,target['opening_b'])]:
                    key=(p['id'],o['id'])
                    if key not in vcache:vcache[key]=opening_visibility(p,o,metric,cfg)
                    sides.append(vcache[key])
                vis.append({'opening_ids':[target['opening_a']['id'],target['opening_b']['id']], 'A':sides[0],'B':sides[1]})
            best=min(vis,key=lambda v:(-int(v['A']['proxy_pass'] and v['B']['proxy_pass']),-min(v['A']['quality_score'],v['B']['quality_score']))) if vis else None
            both_images=pa['quality']['proxy_pass'] and pb['quality']['proxy_pass']
            auto=bool(best and best['A']['proxy_pass'] and best['B']['proxy_pass'] and geo['adjacency']=='positive')
            worst=min(best['A']['quality_score'],best['B']['quality_score']) if best else float(both_images)
            mean=(best['A']['quality_score']+best['B']['quality_score'])/2 if best else float(both_images)
            combos.append({'id':pid,'pano_a':pa['id'],'pano_b':pb['id'],'auto_pass':auto,
                'worst_quality':worst,'mean_quality':mean,'both_images_readable':bool(pa['quality']['decoded'] and pb['quality']['decoded']),
                'visibility_targets':vis,'selected':False})
        eligible=[c for c in combos if c['both_images_readable']]
        selected=select_combinations(eligible,cfg)
        for c in combos:
            if not c['both_images_readable']:c['selection_reason']='unreadable_rgb'
        camera_distance=[]
        for c in combos:
            pa,pb=floor['panos'][c['pano_a']],floor['panos'][c['pano_b']]
            d=np.linalg.norm(pa['matrix'][:2,2]-pb['matrix'][:2,2]);camera_distance.append(d)
        mincam=min(camera_distance)*metric if camera_distance and metric is not None else None
        widths_a=[np.linalg.norm(np.diff(o['floor_endpoints'],axis=0)) for o in a['openings'] if o['kind']=='door']
        widths_b=[np.linalg.norm(np.diff(o['floor_endpoints'],axis=0)) for o in b['openings'] if o['kind']=='door']
        similar=any(abs(x-y)/max(x,y)<cfg['width_relative_tolerance'] for x in widths_a for y in widths_b)
        difficulty=[]
        if mincam is not None and mincam<=cfg['hard_negative_camera_distance_m']:difficulty.append('spatially_close')
        if similar:difficulty.append('door_type_width_similar_proxy')
        if min(len(widths_a),len(widths_b))>=cfg['repeated_door_inventory_minimum']:difficulty.append('repeated_door_inventory_proxy')
        connection={'id':cid,'house':house,'floor':fid,'scope':scope,'units':[a['id'],b['id']],
            'geometry':geo,'annotation_issues':[a['issues'],b['issues']], 'metric_scale':metric,
            'selected_pair_ids':[c['id'] for c in selected], 'combination_count':len(combos),
            'negative_difficulty_proxies':difficulty,'minimum_camera_distance_m':mincam,
            'status':'auto_proposed','human_verified':False}
        records=[]
        legacy=min(eligible,key=lambda c:(not floor['panos'][c['pano_a']]['primary'],not floor['panos'][c['pano_b']]['primary'],
            np.linalg.norm(floor['panos'][c['pano_a']]['matrix'][:2,2]-floor['panos'][c['pano_b']]['matrix'][:2,2]),c['id'])) if eligible else None
        connection['primary_nearest_comparator']={'pair_id':legacy['id'],'worst_quality':legacy['worst_quality'],'auto_pass':legacy['auto_pass'],'definition':'new geometry fixed; primary-first nearest-camera comparator, NOT old dataset exact replay'} if legacy else None
        for c in selected:
            pa,pb=floor['panos'][c['pano_a']],floor['panos'][c['pano_b']]
            record={'id':c['id'],'connection_id':cid,'version':cfg['schema_version'],'split':split,
                'house':house,'floor':fid,'relation_scope':scope,'units':[a['id'],b['id']],
                'panoramas':[c['pano_a'],c['pano_b']],'primary':[pa['primary'],pb['primary']],
                'inputs':{'configuration':'rgb_pair','image_a':pa['image_relative'],'image_b':pb['image_relative'],
                          'input_resolution':[cfg['input_width'],cfg['input_height']]},
                'labels':{'adjacency':geo['adjacency'],'visibility':{'A':None if geo['adjacency']=='negative' else 'uncertain','B':None if geo['adjacency']=='negative' else 'uncertain','applies_to':'per_target','source':'pending_review'},
                          'identifiability':'unreviewed','review_state':'pending_review'},
                'auto_proposal':{k:v for k,v in c.items() if k not in ['id','visibility_targets']},
                'gt':{'correspondences':c['visibility_targets'],'all_geometry_matches':geo['matches'],
                      'relative_transform_b_to_a':(np.linalg.inv(pa['matrix'])@pb['matrix']).tolist(),
                      'metric_scale_floor':metric,'a_local_to_meters':None if metric is None else metric*pa['raw']['floor_plan_transformation']['scale'],
                      'coordinate_convention':'pano local xy; B_to_A=inv(T_A_floor)@T_B_floor; GT offline only',
                      'unit_boundaries_floor':[list(a['polygon'].exterior.coords) if a['polygon'] is not None else None,list(b['polygon'].exterior.coords) if b['polygon'] is not None else None]},
                'quality':{'image_a':pa['quality'],'image_b':pb['quality'],'geometry_conflicts':geo['conflicts'],
                           'negative_certificate':geo['negative_certificate'],'negative_difficulty_proxies':difficulty},
                'provenance':{'annotation':f'{house}/zind_data.json','pano_refs':[pa['source'],pb['source']],
                              'layout_sources':[a['layout_source'],b['layout_source']],'ground_truth_used_for_inference':False},
                'reviews':[]}
            records.append(record)
        # Every considered combination retained; full selected evidence in main index.
        compact=[{k:v for k,v in c.items() if k!='visibility_targets'} for c in combos]
        for dst,src in zip(compact,combos):
            dst['visibility_reasons']=[{'opening_ids':v['opening_ids'],'A':v['A']['reasons'],'B':v['B']['reasons']} for v in src['visibility_targets']]
        yield connection,records,compact


class ObservablePairsDataset:
    """RGB-only access by default. Offline supervision is a separate accessor."""
    def __init__(self,index,root,view=None):
        import json
        self.root=Path(root);records=[json.loads(l) for l in Path(index).read_text().splitlines() if l]
        self.byid={r['id']:r for r in records}
        self.ids=list(self.byid) if view is None else json.loads(Path(view).read_text())['ids']
        if not set(self.ids)<=set(self.byid):raise ValueError('view references missing records')
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        r=self.byid[self.ids[i]];value=public_input(r)
        for side in ['a','b']:
            path=(self.root/value['image_'+side]).resolve();path.relative_to(self.root.resolve())
            with Image.open(path) as im:value['rgb_'+side]=im.convert('RGB').resize(tuple(value['input_resolution']))
        return value
    def supervision(self,i):
        r=self.byid[self.ids[i]]
        return {'offline_only':True,'labels':r['labels'],'gt':r['gt']}
    def gt_candidate_diagnostic(self,i):
        return {'input_configuration':'gt_opening_candidates_DIAGNOSTIC_ONLY','gt_assisted':True,
                'rgb':self[i],'offline_annotation':self.supervision(i)}
