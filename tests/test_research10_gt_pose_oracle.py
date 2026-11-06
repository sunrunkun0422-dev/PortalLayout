from tools.research10_gt_pose_oracle import candidate_status, summarize


def test_unknowns_and_failed_candidates_remain_in_fixed_link_denominator():
    rows = [dict(role='selection', relationship='cross_room', house_id='h', status=status)
            for status in ('success', 'failure', 'unknown')]
    result = summarize(rows)['selection']['cross_room']
    assert result['links'] == 3
    assert (result['success'], result['failure'], result['unknown']) == (1, 1, 1)
    assert result['success_including_unknown'] == 1/3


def test_gray_pose_hypothesis_is_not_converted_to_a_negative():
    assert candidate_status([0, -1]) == 'unknown'
    assert candidate_status([1, -1]) == 'success'
    assert candidate_status([0, 0]) == 'failure'
    assert candidate_status([]) == 'failure'
