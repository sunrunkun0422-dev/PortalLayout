import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import portallayout.models.modules as modules
from portallayout.models.bi_layout import Bi_Layout
from portallayout.models.modules.horizon_net_feature_extractor import HorizonNetFeatureExtractor


class DummyFeatureExtractor(nn.Module):
    def __init__(self, backbone, second=False, scale=8):
        super().__init__()
        self.second = bool(second)

    def forward(self, image, second=False, return_pyramid=False):
        batch = image.shape[0]
        primary = image.new_zeros((batch, 8, 256))
        if second:
            if not self.second:
                raise AssertionError("second feature output was not initialized")
            result = (primary, primary + 1.0)
        else:
            result = primary
        return (result, (image * 2,)) if return_pyramid else result


class DummyPositionEncoding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, features):
        return torch.zeros_like(features)


class DummyTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, features, position, query_embed, query_pos):
        return features + self.bias


class DummyEncoder(nn.Module):
    def forward(self, image):
        return [image]


class ConstantHeightStage(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, conv_list, out_w):
        batch = conv_list[0].shape[0]
        return conv_list[0].new_full((batch, 1, out_w), self.value)


def build_test_model(**overrides):
    options = {
        "backbone": "resnet50",
        "decoder_name": "DummyTransformer",
        "depth": 1,
        "output_number": 2,
        "feature_channel": 8,
        "embedding_channel": 8,
        "use_same_head": False,
        "share_TF": True,
        "two_conv_out": False,
    }
    options.update(overrides)
    with patch(
        'portallayout.models.bi_layout.HorizonNetFeatureExtractor', DummyFeatureExtractor
    ), patch(
        'portallayout.models.bi_layout.PositionEmbeddingSine', DummyPositionEncoding
    ), patch.object(
        modules, "DummyTransformer", DummyTransformer, create=True
    ):
        return Bi_Layout(**options)


class BiLayoutConfigTest(unittest.TestCase):
    def test_optional_pyramid_keeps_legacy_outputs_and_state_keys(self):
        for second in (False, True):
            model = build_test_model(two_conv_out=second).eval()
            image = torch.rand(1, 3, 8, 16, requires_grad=True)
            keys = list(model.state_dict())
            old = model(image, return_features=True)
            new = model(image, return_features=True, return_feature_pyramid=True)
            self.assertNotIn('feature_2d_pyramid', old)
            self.assertEqual(keys, list(model.state_dict()))
            for name in old:
                torch.testing.assert_close(old[name], new[name])
            spatial = new['feature_2d_pyramid'][0]
            torch.testing.assert_close(spatial, image * 2)
            spatial.sum().backward()
            torch.testing.assert_close(image.grad, torch.full_like(image, 2))

    def test_extractor_returns_actual_precompression_spatial_tensor(self):
        extractor = HorizonNetFeatureExtractor.__new__(HorizonNetFeatureExtractor)
        nn.Module.__init__(extractor)
        extractor.second = False
        extractor.step_cols = 4
        extractor.feature_extractor = DummyEncoder()
        extractor.reduce_height_module = ConstantHeightStage(7)
        extractor._prepare_x = lambda x: x
        image = torch.rand(1, 3, 4, 16)
        features, pyramid = extractor(image, return_pyramid=True)
        self.assertIs(pyramid[0], image)
        self.assertEqual(features.shape, (1, 1, 4))
        self.assertEqual(pyramid[0].shape, (1, 3, 4, 16))

    def test_same_head_reuses_both_depth_and_ratio_heads(self):
        model = build_test_model(use_same_head=True)
        original = torch.randn(2, model.patch_num, model.patch_dim)
        extended = torch.randn(2, model.patch_num, model.patch_dim)

        output = model.bi_layout_outputs(original, extended)

        self.assertFalse(hasattr(model, "linear_depth_output_2"))
        expected_depth = model.linear_depth_output(extended).view(2, model.patch_num)
        torch.testing.assert_close(output["new_depth"], expected_depth)
        self.assertEqual(output["ratio"].shape, (2, 1))

    def test_share_tf_false_uses_an_independent_transformer(self):
        model = build_test_model(share_TF=False)
        self.assertIsNot(model.transformer, model.transformer_2)
        model.transformer.bias.data.fill_(1.0)
        model.transformer_2.bias.data.fill_(2.0)

        output = model(torch.zeros(1, 3, 8, 16), return_features=True)

        torch.testing.assert_close(
            output["enc_feature"], torch.ones_like(output["enc_feature"])
        )
        torch.testing.assert_close(
            output["ext_feature"], torch.full_like(output["ext_feature"], 2.0)
        )

    def test_two_conv_out_uses_independent_height_compression_features(self):
        model = build_test_model(two_conv_out=True)

        output = model(torch.zeros(1, 3, 8, 16), return_features=True)

        self.assertTrue(model.feature_extractor.second)
        torch.testing.assert_close(
            output["layout_feature"], torch.zeros_like(output["layout_feature"])
        )
        torch.testing.assert_close(
            output["new_layout_feature"],
            torch.ones_like(output["new_layout_feature"]),
        )

    def test_horizon_extractor_calls_second_height_stage(self):
        extractor = HorizonNetFeatureExtractor.__new__(HorizonNetFeatureExtractor)
        nn.Module.__init__(extractor)
        extractor.second = True
        extractor.step_cols = 4
        extractor.feature_extractor = DummyEncoder()
        extractor.reduce_height_module = ConstantHeightStage(1.0)
        extractor.reduce_height_module2 = ConstantHeightStage(2.0)
        extractor._prepare_x = lambda image: image

        primary, secondary = extractor(torch.zeros(1, 3, 2, 8), second=True)

        self.assertEqual(primary.unique().item(), 1.0)
        self.assertEqual(secondary.unique().item(), 2.0)

    def test_invalid_flag_combinations_fail_early(self):
        with self.assertRaisesRegex(ValueError, "output_number"):
            build_test_model(output_number=3)
        with self.assertRaisesRegex(ValueError, "output_number=2"):
            build_test_model(output_number=1, two_conv_out=True)
        with self.assertRaisesRegex(ValueError, "convolutional backbone"):
            Bi_Layout(backbone="patch", two_conv_out=True)


if __name__ == "__main__":
    unittest.main()
