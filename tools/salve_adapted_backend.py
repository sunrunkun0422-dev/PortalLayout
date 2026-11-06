"""SALVe PGO and native confidence-based stitching, preserving all components.

Fixed 0.93 verifier threshold; no GT poses/shapes enter reconstruction.
Official loaders read annotations for IDs/metadata. GT labels only locate files.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--salve-root',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--predictions-root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--building',default='0007');p.add_argument('--floor',default='floor_02')
    args=p.parse_args();sys.path.insert(0,str(args.salve_root.resolve()))
    import numpy as np
    import networkx as nx
    import gtsam
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import mapping
    from salve.common import edge_classification as ec
    from salve.common.edgewdopair import EdgeWDOPair
    from salve.common.sim2 import Sim2
    from salve.dataset.hnet_prediction_loader import load_inferred_floor_pose_graph
    from salve.algorithms.spanning_tree import greedily_construct_st_Sim2
    from salve.algorithms.pose2_slam import planar_slam,OdometryMeasurement
    from salve.utils.axis_alignment_utils import align_pair_measurement_by_vanishing_angle
    from salve.stitching import shape
    from salve.stitching.constants import DEFAULT_CAMERA_HEIGHT
    from salve.stitching.models.locations import Pose,Point2d
    threshold=.93
    pg=load_inferred_floor_pose_graph(args.building,args.floor,str(args.raw_root.resolve()),str(args.predictions_root.resolve()))
    all_m=ec.get_edge_classifications_from_serialized_preds(args.building,args.floor,str(args.run/'verifier'),str(args.run/'hypotheses'))[(args.building,args.floor)]
    accepted=[m for m in all_m if m.y_hat==1 and m.prob>=threshold]
    corrected=[];best={}
    for m in accepted:
        edge=EdgeWDOPair.from_wdo_pair_uuid(m.i1,m.i2,m.wdo_pair_uuid)
        pose=align_pair_measurement_by_vanishing_angle(m.i1,m.i2,m.i2Si1,edge,pg.nodes,visualize=False)
        pose=m.i2Si1 if pose is None else pose
        corrected.append((m,pose))
        if (m.i1,m.i2) not in best or m.prob>best[(m.i1,m.i2)][0].prob:best[(m.i1,m.i2)]=(m,pose)
    graph=nx.Graph();graph.add_nodes_from(pg.nodes);graph.add_edges_from(best)
    components=sorted((sorted(c) for c in nx.connected_components(graph)),key=lambda c:(-len(c),c))
    out=args.run/'reconstruction';out.mkdir(exist_ok=True)
    report={'adapted_baseline':True,'building':args.building,'floor':args.floor,'threshold':threshold,
            'scored_hypotheses':len(all_m),'accepted_hypotheses':len(accepted),'accepted_unique_edges':len(best),
            'panorama_count':len(pg.nodes),'components':[],'gt_poses_used_for_reconstruction':False,
            'coordinate_units':'camera-height normalized; stitching internal coordinates use DEFAULT_CAMERA_HEIGHT',
            'original_paper_metrics_reproduced':False}
    fig,axes=plt.subplots(1,len(components),figsize=(max(6,6*len(components)),6),squeeze=False)
    for cid,nodes in enumerate(components):
        local={k:v[1] for k,v in best.items() if k[0] in nodes and k[1] in nodes}
        if local:
            initial=greedily_construct_st_Sim2(local,verbose=False)
            init=[None if s is None else gtsam.Pose2(*s.translation,np.deg2rad(s.theta_deg)) for s in initial]
            measures=[OdometryMeasurement(m.i1,m.i2,gtsam.Pose2(*s.translation,np.deg2rad(s.theta_deg))) for m,s in corrected if m.i1 in nodes and m.i2 in nodes]
            optimized,_=planar_slam(init,measures,{},[],optimize_poses_only=True)
            poses={i:Sim2(optimized[i].rotation().matrix(),optimized[i].translation(),1.) for i in nodes}
        else:poses={nodes[0]:Sim2(np.eye(2),np.zeros(2),1.)}
        dense={};confidence={};locations={};pose_json={}
        for i in nodes:
            paths=list((args.predictions_root/'horizon_net'/args.building).glob(f'{args.floor}_*_pano_{i}.json'))
            if len(paths)!=1:raise RuntimeError('Ambiguous MHNet prediction')
            pred=json.loads(paths[0].read_text())['predictions']['room_shape']['raw_predictions']
            dense[i],confidence[i]=shape.generate_dense_shape(pred['floor_boundary'],pred['floor_boundary_uncertainty'])
            s=poses[i];t=s.translation*DEFAULT_CAMERA_HEIGHT
            locations[i]=Pose(Point2d(*t),-s.theta_deg)
            pose_json[str(i)]={'rotation':s.rotation.tolist(),'translation':s.translation.tolist(),'scale':s.scale}
        groups=shape.group_panos_by_room(dense,locations)
        component_dir=out/f'component_{cid}';component_dir.mkdir(exist_ok=True)
        _,_,fused=shape.refine_predicted_shape(groups,dense,confidence,locations,str(component_dir))
        (component_dir/'floorplan.geojson').write_text(json.dumps(mapping(fused)))
        ax=axes[0,cid]
        for poly in (list(fused.geoms) if fused.geom_type=='MultiPolygon' else [fused]):
            x,y=poly.exterior.xy;ax.fill(x,y,alpha=.3);ax.plot(x,y,color='black',lw=.8)
        for i in nodes:
            t=poses[i].translation*DEFAULT_CAMERA_HEIGHT;ax.scatter(*t,s=15);ax.text(*t,str(i))
        ax.set_aspect('equal');ax.set_title(f'Component {cid}: {len(nodes)} / {len(pg.nodes)} panos\n'+('PGO + native stitching' if local else 'Unlocalized singleton'))
        report['components'].append({'id':cid,'nodes':nodes,'localized_relative_to_anchor':len(nodes)>1,'anchor':min(nodes),'poses':pose_json,'room_groups':groups,'stitched_area_internal_units':fused.area})
    fig.suptitle('SALVe adaptation: DAV2 metric depth (not original SALVe benchmark)')
    fig.tight_layout();fig.savefig(out/'floorplan.png',dpi=160);fig.savefig(out/'floorplan.svg');plt.close(fig)
    # Post-hoc diagnostic only; never used to accept edges or choose a component.
    from salve.common.posegraph2d import get_gt_pose_graph
    gt=get_gt_pose_graph(args.building,args.floor,str(args.raw_root))
    report['accepted_edge_diagnostics']=[{'i1':m.i1,'i2':m.i2,'confidence':m.prob,'candidate_label':m.y_true,'raw_rotation_error_deg':float(m.compute_measurement_relative_pose_error_from_gt(gt)[0]),'raw_translation_error_normalized':float(m.compute_measurement_relative_pose_error_from_gt(gt)[1])} for m in accepted]
    report['connected_panos']=sum(len(c) for c in components if len(c)>1)
    report['unlocalized_singletons']=[c[0] for c in components if len(c)==1]
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
