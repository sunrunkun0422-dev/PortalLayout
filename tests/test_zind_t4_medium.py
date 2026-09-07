import unittest

import torch

from tools.diagnose_zind_t4_medium import _mask_intervals, house_first_select


def _record(pair_id, house, positive):
    return {
        "id": pair_id,
        "scene_id": house,
        "supervision": {"is_match": positive},
    }


class T4MediumDiagnosticTest(unittest.TestCase):
    def test_house_first_selection_is_balanced_and_reproducible(self):
        records = []
        for index in range(8):
            house = "h{:02d}".format(index)
            records.append(_record("{}_p".format(house), house, True))
            records.append(_record("{}_n".format(house), house, False))
        payload = {"split": "train", "pairs": records}
        first, first_meta = house_first_select(payload, 4, 4, 123)
        second, second_meta = house_first_select(payload, 4, 4, 123)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(first_meta["pair_order_sha256"], second_meta["pair_order_sha256"])
        self.assertEqual(sum(row["supervision"]["is_match"] for row in first), 4)
        self.assertEqual(first_meta["house_count"], 8)
        self.assertFalse(first_meta["selection_uses_model_performance"])
        self.assertFalse(first_meta["test_data_accessed"])

    def test_mask_intervals_preserve_wrap_and_padding(self):
        masks = torch.zeros((1, 3, 8), dtype=torch.bool)
        masks[0, 0, 2:5] = True
        masks[0, 1, [6, 7, 0, 1]] = True
        valid = torch.tensor([[True, True, False]])
        intervals = _mask_intervals(masks, valid)
        self.assertEqual(intervals.tolist(), [[[2, 4], [6, 1], [-1, -1]]])


if __name__ == "__main__":
    unittest.main()
