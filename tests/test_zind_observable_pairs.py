import copy
import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from dataset.zind_observable_pairs import (geometry_relation,projection,circular_iou,ray_clear_fraction,
    parse_interfaces,select_combinations,public_input,finite_positive,load_house,build_floor)
from dataset.zind_pair_mining import zind_transform_matrix,transform_points
from tools.build_zind_observable_pairs import pilot_sample,make_views

CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/zind_observable_pairs_v1.json').read_text())


def units():
    def op(name):return {'id':name,'kind':'door','floor_endpoints':[[0,-.4],[0,.4]],'height_floor_z':[0,2],'sources':[]}
    return ({'polygon':Polygon([[-2,-1],[0,-1],[0,1],[-2,1]]),'inventory_complete':True,'openings':[op('a')]},
            {'polygon':Polygon([[0,-1],[2,-1],[2,1],[0,1]]),'inventory_complete':True,'openings':[op('b')]})


class ObservableGeometryTests(unittest.TestCase):
    def test_duplicate_bare_pano_ids_keep_distinct_images_and_transforms(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'house').mkdir()
            raw={'is_primary':True,'is_inside':True,'camera_height':1,'layout_raw':{'vertices':[[-1,-1],[1,-1],[1,1],[-1,1]],'doors':[],'openings':[],'windows':[]}}
            raw['layout_complete']=copy.deepcopy(raw['layout_raw']);raw['layout_visible']=copy.deepcopy(raw['layout_raw'])
            a=copy.deepcopy(raw);b=copy.deepcopy(raw)
            for i,pano in enumerate([a,b]):
                pano['image_path']=f'{i}.jpg';pano['floor_plan_transformation']={'scale':1,'rotation':0,'translation':[i*3,0]}
                Image.new('RGB',(64,32),(i*50+80,100,120)).save(root/'house'/pano['image_path'])
            payload={'scale_meters_per_coordinate':{'floor':1},'merger':{'floor':{'complete_room_1':{'partial_room_1':{'pano_34':a},'partial_room_2':{'pano_34':b}}}}}
            floors,errors,counts=load_house(payload,'house',root,CFG);panos=floors['floor']['panos']
            self.assertEqual(len(panos),2);self.assertEqual(counts['duplicate_bare_pano_ids'],1)
            self.assertEqual(len({p['image_relative'] for p in panos.values()}),2)
            rows=[r for _,rr,_ in build_floor('house','floor',floors['floor'],'train',root,CFG) for r in rr]
            self.assertEqual(len(rows),1);self.assertNotEqual(rows[0]['inputs']['image_a'],rows[0]['inputs']['image_b'])
            self.assertNotEqual(rows[0]['panoramas'][0],rows[0]['panoramas'][1])
    def test_reverse_and_multiple_real_openings(self):
        a,b=units();b['openings'][0]['floor_endpoints'].reverse()
        r=geometry_relation(a,b,1,CFG);self.assertEqual(r['adjacency'],'positive');self.assertTrue(r['matches'][0]['reversed_b'])
        for unit in [a,b]:
            x=copy.deepcopy(unit['openings'][0]);x['id']+='2';x['floor_endpoints']=[[0,.65],[0,.9]];unit['openings'].append(x)
        r=geometry_relation(a,b,1,CFG);self.assertEqual(r['adjacency'],'positive');self.assertEqual(len(r['matches']),2)
    def test_scale_not_silently_one(self):
        a,b=units();self.assertEqual(geometry_relation(a,b,None,CFG)['adjacency'],'unknown')
        self.assertFalse(finite_positive(float('nan')));self.assertFalse(finite_positive(0))
    def test_near_conflict_and_one_to_many(self):
        a,b=units();b['openings'][0]['floor_endpoints']=[[0,-.1],[0,.7]]
        self.assertEqual(geometry_relation(a,b,1,CFG)['adjacency'],'unknown')
        a,b=units();duplicate=copy.deepcopy(b['openings'][0]);duplicate['id']='b2';b['openings'].append(duplicate)
        self.assertEqual(geometry_relation(a,b,1,CFG)['adjacency'],'unknown')
    def test_same_geometry_wrong_room_membership(self):
        a,b=units();b['polygon']=a['polygon'];self.assertEqual(geometry_relation(a,b,1,CFG)['adjacency'],'unknown')
    def test_absence_does_not_mean_negative(self):
        a,b=units();a['openings']=[];b['openings']=[]
        self.assertEqual(geometry_relation(a,b,1,CFG)['adjacency'],'unknown')
        b['polygon']=Polygon([[2,-1],[4,-1],[4,1],[2,1]])
        self.assertEqual(geometry_relation(a,b,1,CFG)['adjacency'],'negative')
        a['inventory_complete']=False;self.assertEqual(geometry_relation(a,b,1,CFG)['adjacency'],'unknown')
    def test_triplets_and_window_semantics(self):
        l={'doors':[[0,1],[1,1],[-1,1]],'windows':[[2,1],[3,1],[0,1]],'openings':[]}
        r,e=parse_interfaces(l,np.eye(3),'x');self.assertFalse(e);self.assertEqual(len(r),2);self.assertEqual(r[0]['local_endpoints'],[[0.,1.],[1.,1.]])
        l['doors'].pop();self.assertTrue(parse_interfaces(l,np.eye(3),'x')[1])
    def test_ray_target_contact_vs_front_occluder(self):
        wall=[[-2,-2],[2,-2],[2,2],[-2,2]];target=[[2,-.4],[2,.4]]
        r=ray_clear_fraction([0,0],target,wall,[],.02,17);self.assertEqual(r['fraction'],1)
        inner=[[[.8,-.4],[1.1,-.4],[1.1,.4],[.8,.4]]]
        r=ray_clear_fraction([0,0],target,wall,inner,.02,17);self.assertEqual(r['fraction'],0)
    def test_wrap_projection_and_set_iou(self):
        p=projection([[-.05,-1],[.05,-1]],[-1,1]);self.assertTrue(p['wraps_seam']);self.assertLess(p['length_u'],.03)
        a={'start_u':.99,'length_u':.02};b={'start_u':0,'length_u':.01};self.assertAlmostEqual(circular_iou(a,b),.5)
    def test_coordinate_round_trip(self):
        t=zind_transform_matrix({'translation':[2,3],'rotation':90,'scale':2});p=transform_points([[1,0]],t)
        np.testing.assert_allclose(p,[[2,5]],atol=1e-12);np.testing.assert_allclose(transform_points(p,np.linalg.inv(t)),[[1,0]],atol=1e-12)
    def test_maximin_and_missing_images(self):
        rows=[{'id':'a','auto_pass':True,'worst_quality':.3,'mean_quality':.65,'pano_a':'a','pano_b':'b'},
              {'id':'b','auto_pass':True,'worst_quality':.7,'mean_quality':.7,'pano_a':'c','pano_b':'d'}]
        self.assertEqual(select_combinations(rows,CFG)[0]['id'],'b')
    def test_rgb_input_and_verified_view_do_not_leak(self):
        r={'id':'opaque','inputs':{'image_a':'a.jpg','image_b':'b.jpg','input_resolution':[1024,512]},'gt':{'pose':'SECRET'},
           'house':'SECRET','relation_scope':'cross_complete_room','labels':{'adjacency':'positive'},'auto_proposal':{'auto_pass':True}}
        self.assertNotIn('SECRET',str(public_input(r)));self.assertFalse(make_views([r])['core_verified'])
    def test_probability_sampling_reproducibility(self):
        rows=[]
        for h in range(10):
            for i in range(4):rows.append({'id':f'{h}:{i}','house':str(h),'relation_scope':'cross_complete_room','labels':{'adjacency':'positive'},'auto_proposal':{'auto_pass':bool(i%2)},'quality':{'negative_difficulty_proxies':[]}})
        a,_=pilot_sample(rows,10,123);b,_=pilot_sample(list(reversed(rows)),10,123)
        self.assertEqual(a,b);self.assertEqual(len(a),10)
        self.assertTrue(all(abs(x['inclusion_probability']-.25)<1e-10 for x in a))


if __name__=='__main__':unittest.main()
