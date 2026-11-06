from tools.research10_r04_fixed_compare import evaluate_pair, labels_for_pair, summarize, top1


def test_reversed_weak_pair_transposes_labels_by_observation_identity():
    weak = {'observation_a': 'b', 'observation_b': 'a', 'candidate_ids_a': ['b1', 'b2'],
            'candidate_ids_b': ['a1'], 'weak_cells': [[1], [-1]]}
    a, b, cells = labels_for_pair(weak, 'a', 'b')
    assert a == ['a1'] and b == ['b1', 'b2']
    assert cells.tolist() == [[1, -1]]


def test_positive_unknown_and_missing_scores_keep_same_denominator():
    weak = {'observation_a': 'a', 'observation_b': 'b', 'candidate_ids_a': ['a1'],
            'candidate_ids_b': ['b1', 'b2'], 'weak_cells': [[1, -1]]}
    raw = {'observation_a': 'a', 'observation_b': 'b', 'candidate_ids_a': ['a1'],
           'candidate_ids_b': ['b1', 'b2'], 'numerical_status': 'assigned', 'scores': [[2., 1.]]}
    entries = []
    for candidate, score in [('b1', .1), ('b2', .9)]:
        entries.append({'candidate_id_a': 'a1', 'candidate_id_b': candidate,
                        'appearance_cosine': {'available': True, 'score': score},
                        'width_ratio': {'available': False, 'score': None}})
    base = {'pair_id': 'p', 'floor_uid': 'f', 'house_id': 'h', 'observation_a': 'a',
            'observation_b': 'b', 'status': 'scored', 'scores': entries}
    row = evaluate_pair(base, weak, raw, 'cross_room')
    assert row['methods']['matcher_raw_logit']['outcome'] == 'correct'
    assert row['methods']['single_view_descriptor_cosine']['outcome'] == 'unknown'
    assert row['methods']['opening_width_ratio']['outcome'] == 'no_score'
    result = summarize([row])
    assert result['pairs_with_known_positive_candidate'] == 1
    assert result['methods']['matcher_raw_logit']['correct_over_known_positive_pairs'] == 1
    assert result['methods']['single_view_descriptor_cosine']['known_positive_pair_outcomes'] == {'unknown': 1}
    assert result['methods']['opening_width_ratio']['known_positive_pair_outcomes'] == {'no_score': 1}


def test_top_score_ties_have_deterministic_candidate_identity():
    result = top1([('a2', 'b', .5), ('a1', 'b', .5)], {('a2', 'b'): 1, ('a1', 'b'): 0})
    assert result['selected'] == ['a1', 'b']
    assert result['outcome'] == 'incorrect'
