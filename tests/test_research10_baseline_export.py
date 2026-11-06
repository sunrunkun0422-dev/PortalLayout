import builtins
import copy
import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from portallayout.models.opening_representation import OpeningRepresentationEncoder, OpeningRepresentationConfig
from portallayout.models.opening_pair_matcher import OpeningPairMatcher, OpeningPairConfig
from portallayout.utils.opening_pair_checkpoint import checkpoint_payload, save_checkpoint
from portallayout.utils.room_feature_cache import ObservationFeatureCache, content_hash, fingerprint, write_json_atomic
from tests import test_opening_representation as representation_fixture
from tools.research10_baseline_export import (load_context, make_plan, validate_plan,
                                               load_bound_encoder, score_plan)


def make_fixture(tmp_path, missing=False):
    dataset = tmp_path/'dataset'; config_dir = tmp_path/'protocol'
    rows = []; hashes = {}; roles = {r:[r+'_house'] for r in ('fit','selection','calibration','audit')}
    for role in roles:
        floor, sample = 'floor_'+role, 'sample_'+role
        obs = [role+'_v0', role+'_v1']
        manifest = {'schema_version':'zind-floorset.input.v1','sample_id':sample,'floor_uid':floor,
                    'task':'unposed_floor_layout_from_panoramas','input_mode':'panorama_only',
                    'selection_policy':'all_panoramas','projection':'equirectangular','pose_prior':'none',
                    'calibration_mode':'scale_unknown','observation_order':obs,
                    'observations':[{'observation_id':x,'rgb':{'root_key':'zind_raw','relative_path':x+'.jpg'},
                                     'image_size_wh':[64,32]} for x in obs]}
        rel = 'inputs/'+sample+'.json';write_json_atomic(dataset/rel, manifest)
        rows.append({'floor_uid':floor,'house_id':roles[role][0],'role':role,'input_manifest':rel})
        hashes[floor] = content_hash(dataset/rel)
    (dataset/'samples').mkdir(exist_ok=True)
    (dataset/'samples/all_panoramas.jsonl').write_text(''.join(
        __import__('json').dumps({'sample_id':'sample_'+r['role'],'floor_uid':r['floor_uid'],
                                  'house_id':r['house_id'],'official_split':'train',
                                  'input_manifest':r['input_manifest']})+'\n' for r in rows))
    write_json_atomic(dataset/'splits/official_house_split.json',{'train':[x[0] for x in roles.values()],'val':[],'test':[]})
    cohort = {'schema_version':'research10.cohort.v1','dataset_root':str(dataset),
              'roles':roles,'cohort':rows,'input_manifest_sha256':hashes}
    cp=config_dir/'cohort.json';write_json_atomic(cp,cohort)
    protocol={'schema_version':'research10.house-protocol.v1',**roles,'source_sha256':content_hash(cp)}
    protocol['fingerprint']=fingerprint(protocol);pp=config_dir/'house_protocol.json';write_json_atomic(pp,protocol)
    cache_root=tmp_path/'cache'
    writer=ObservationFeatureCache(cache_root,{'cohort_sha256':content_hash(cp),'uses_ground_truth':False})
    record, arrays=representation_fixture.RepresentationAdapterTest().source()
    arrays['opening_logits']=np.ones(32,np.float32)
    for role in ('fit','selection','calibration'):
        refs=[]
        for i in range(2):
            uid=role+'_v'+str(i)
            if missing and role=='selection' and i==1:
                refs.append({'status':'prediction_unavailable','observation_id':uid});continue
            pred=copy.deepcopy(record['prediction']);pred['openings'][0]['candidate_id']=role+'_c'+str(i)
            refs.append(writer.put(uid,hashlib.sha256(uid.encode()).hexdigest(),arrays,pred))
        write_json_atomic(cache_root/'floors'/('floor_'+role+'.json'),
                          {'sample_id':'sample_'+role,'floor_uid':'floor_'+role,
                           'contract_fingerprint':writer.fingerprint,'observations':refs,
                           'observation_order':[role+'_v0',role+'_v1']})
    context=load_context(cache_root,dataset,pp)
    config=OpeningRepresentationConfig(feature_1d_dim=8,feature_2d_dim=12,descriptor_dim=16,context_hidden_dim=8,dropout=.1)
    encoder=OpeningRepresentationEncoder(config);matcher=OpeningPairMatcher(OpeningPairConfig(dim=16,heads=4,chunk_size=2))
    contract={'source_contract':context['cache'].fingerprint,'source_snapshot':context['cache'].snapshot,
              'house_protocol':protocol,'fit_houses':['fit_house']}
    checkpoint=tmp_path/'weights.pt'
    save_checkpoint(checkpoint,checkpoint_payload(encoder,matcher,training_status='smoke_trained',training_contract=contract))
    return context,checkpoint,encoder,matcher,contract


def test_real_encoder_cache_scores_frozen_selection_without_gt(tmp_path):
    context,checkpoint,_,_,_=make_fixture(tmp_path)
    before={k:v for k,v in context['cache'].before.items()}
    plan=make_plan(context,('selection',),1)
    assert plan['population_pairs']==1 and plan['selected_pairs']==1
    encoder,state=load_bound_encoder(context,checkpoint,'cpu')
    original_open, original_io = builtins.open, io.open
    def guarded(original):
        def call(path,*args,**kwargs):
            if isinstance(path,(str,Path)) and ('/labels/' in str(path) or 'zind_data' in str(path)):
                raise AssertionError('ground truth read')
            return original(path,*args,**kwargs)
        return call
    with patch('builtins.open',side_effect=guarded(original_open)),patch('io.open',side_effect=guarded(original_io)):
        rows=score_plan(context,plan,encoder)
    assert len(rows)==1 and rows[0]['status']=='scored' and rows[0]['candidate_pair_count']==1
    item=rows[0]['scores'][0]
    assert (item['candidate_id_a'],item['candidate_id_b'])==('selection_c0','selection_c1')
    assert item['appearance_cosine']['available'] and item['width_ratio']['score'] is None
    assert item['width_ratio']['reason']=='a_scale_unknown'
    assert state['training_status']=='smoke_trained' and context['cache'].before==before


def test_unchanged_fixed_candidates_and_roles_required(tmp_path):
    context,_,_,_,_=make_fixture(tmp_path);plan=make_plan(context,('fit',),1)
    bad=copy.deepcopy(plan);bad['pairs'][0]['candidate_ids_a']=['not-the-candidate']
    bad['fingerprint']=fingerprint({k:v for k,v in bad.items() if k!='fingerprint'})
    with pytest.raises(ValueError,match='candidate identities'):
        validate_plan(bad,context)
    bad=copy.deepcopy(plan);bad['pairs'][0]['role']='calibration'
    bad['fingerprint']=fingerprint({k:v for k,v in bad.items() if k!='fingerprint'})
    with pytest.raises(ValueError,match='role'):
        validate_plan(bad,context)
    bad=copy.deepcopy(plan);bad['labels']={'target':1}
    bad['fingerprint']=fingerprint({k:v for k,v in bad.items() if k!='fingerprint'})
    with pytest.raises(ValueError,match='unsupported'):
        validate_plan(bad,context)


def test_checkpoint_source_and_protocol_must_match(tmp_path):
    context,_,encoder,matcher,contract=make_fixture(tmp_path)
    for key in ('source_contract','source_snapshot','house_protocol'):
        bad=copy.deepcopy(contract);bad[key]='wrong'
        path=tmp_path/(key+'.pt')
        save_checkpoint(path,checkpoint_payload(encoder,matcher,training_status='smoke_trained',training_contract=bad))
        with pytest.raises(ValueError,match='checkpoint source'):
            load_bound_encoder(context,path,'cpu')
    bad=copy.deepcopy(contract);bad['fit_houses']=['audit_house']
    path=tmp_path/'bad_houses.pt'
    save_checkpoint(path,checkpoint_payload(encoder,matcher,training_status='smoke_trained',training_contract=bad))
    with pytest.raises(ValueError,match='training house'):
        load_bound_encoder(context,path,'cpu')


def test_missing_prediction_is_not_silently_dropped(tmp_path):
    context,checkpoint,_,_,_=make_fixture(tmp_path,missing=True)
    plan=make_plan(context,('selection',),1)
    assert plan['pairs'][0]['candidate_ids_b'] is None
    encoder,_=load_bound_encoder(context,checkpoint,'cpu')
    rows=score_plan(context,plan,encoder)
    assert rows[0]['status']=='prediction_unavailable' and rows[0]['scores']==[]
    assert rows[0]['candidate_pair_count']==0


def test_shared_height_assumption_is_explicit_and_prediction_only(tmp_path):
    context,checkpoint,_,_,_=make_fixture(tmp_path)
    plan=make_plan(context,('selection',),1);encoder,_=load_bound_encoder(context,checkpoint,'cpu')
    rows=score_plan(context,plan,encoder,scale_assumption=True)
    width=rows[0]['scores'][0]['width_ratio']
    assert width['available'] and np.isclose(width['score'],1.)
