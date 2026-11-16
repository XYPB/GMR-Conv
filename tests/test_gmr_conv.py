"""CPU regressions: python -m unittest discover -s tests -v."""

import importlib.util
import math
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


# Load the layer directly so these tests need no optional network/transform deps.
spec = importlib.util.spec_from_file_location(
    "gmr_conv", Path(__file__).resolve().parents[1] / "src/GMR_Conv/gmr_conv.py"
)
gmr_conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gmr_conv)

CONVOLUTIONS = (
    gmr_conv.GMR_Conv1d,
    gmr_conv.GMR_Conv2d,
    gmr_conv.GMR_Conv3d,
)
ALL_CONVOLUTIONS = CONVOLUTIONS + (gmr_conv.GMR_ConvTranspose2d,)


class GMRConstructorTests(unittest.TestCase):
    def test_flags_are_independent(self):
        for cls in ALL_CONVOLUTIONS:
            for dw_fwd in (False, True):
                for clip_sigma in (False, True):
                    with self.subTest(cls=cls.__name__, dw_fwd=dw_fwd,
                                      clip_sigma=clip_sigma):
                        layer = cls(2, 4, 3, dw_fwd=dw_fwd, clip_sigma=clip_sigma)
                        self.assertIs(layer.dw_fwd, dw_fwd)
                        self.assertIs(layer.clip_sigma, clip_sigma)

    def test_defaults_and_single_flag_overrides(self):
        for cls in ALL_CONVOLUTIONS:
            for kwargs, expected in (
                ({}, (True, True)),
                ({"dw_fwd": False}, (False, True)),
                ({"clip_sigma": False}, (True, False)),
            ):
                with self.subTest(cls=cls.__name__, kwargs=kwargs):
                    layer = cls(2, 4, 3, **kwargs)
                    self.assertEqual((layer.dw_fwd, layer.clip_sigma), expected)

    def test_existing_positional_arguments_keep_their_meaning(self):
        for cls in CONVOLUTIONS:
            with self.subTest(cls=cls.__name__):
                # 3D historically puts sigma_no_weight_decay before train_gaussian_sigma.
                sigma_flags = (True, False) if cls is gmr_conv.GMR_Conv3d else (False, True)
                layer = cls(
                    2, 4, 3, 2, 1, 1, False, 1, False, "zeros", 2,
                    True, False, False, True, *sigma_flags, 3.0,
                    False, torch.device("cpu"), torch.float64,
                )
                self.assertFalse(layer.dw_fwd)
                self.assertTrue(layer.clip_sigma)
                self.assertFalse(layer.train_gaussian_sigma)
                self.assertFalse(layer.inference_accelerate)
                self.assertFalse(layer.force_circular)
                self.assertTrue(layer.train_index_mat)
                self.assertIsNone(layer.bias)
                self.assertEqual(layer.weight.dtype, torch.float64)
                self.assertEqual(layer.weight.device, torch.device("cpu"))
                self.assertNotIn("log_sigmas", dict(layer.named_parameters()))
                torch.testing.assert_close(
                    layer.log_sigmas.exp(), torch.full((2,), 0.5, dtype=torch.float64)
                )

    def test_sigma_weight_decay_option_reaches_parent(self):
        for cls in CONVOLUTIONS:
            with self.subTest(cls=cls.__name__):
                layer = cls(2, 4, 3, train_gaussian_sigma=True, sigma_no_weight_decay=True)
                self.assertTrue(layer.log_sigmas.requires_grad)
                self.assertTrue(layer.log_sigmas._no_weight_decay)

    def test_clip_sigma_controls_actual_clipping(self):
        for cls in ALL_CONVOLUTIONS:
            with self.subTest(cls=cls.__name__):
                clipped = cls(2, 4, 3, clip_sigma=True)
                unclipped = cls(2, 4, 3, clip_sigma=False)
                with torch.no_grad():
                    clipped.log_sigmas.fill_(math.log(10 * clipped.num_rings))
                    unclipped.log_sigmas.copy_(clipped.log_sigmas)
                clipped_matrix = clipped._get_index_matrix()
                self.assertFalse(torch.allclose(clipped_matrix, unclipped._get_index_matrix()))
                with torch.no_grad():
                    unclipped.log_sigmas.fill_(math.log(2 * clipped.num_rings))
                torch.testing.assert_close(clipped_matrix, unclipped._get_index_matrix())


class GMRForwardTests(unittest.TestCase):
    def test_each_dimension_forward_and_backward(self):
        for dimensions, cls in enumerate(CONVOLUTIONS, start=1):
            with self.subTest(dimensions=dimensions):
                layer = cls(2, 4, 3, padding=1, dw_fwd=False)
                inputs = torch.randn((2, 2) + (5,) * dimensions, requires_grad=True)
                outputs = layer(inputs)
                self.assertEqual(outputs.shape, (2, 4) + (5,) * dimensions)
                outputs.square().mean().backward()
                for grad in (inputs.grad, layer.weight.grad, layer.log_sigmas.grad):
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)

    def test_disabled_dw_forward_supports_grouped_reflect_padding(self):
        layer = gmr_conv.GMR_Conv2d(
            4, 6, 3, stride=2, padding=1, groups=2,
            padding_mode="reflect", dw_fwd=False,
        )
        inputs = torch.randn(2, 4, 7, 7)
        expected = F.conv2d(
            F.pad(inputs, (1, 1, 1, 1), mode="reflect"),
            layer._make_weight_matrix(layer.weight), layer.bias, stride=2, groups=2,
        )
        for training in (True, False):
            with self.subTest(training=training):
                layer.train(training)
                torch.testing.assert_close(layer(inputs), expected)

    def test_dw_and_dense_forward_have_matching_outputs_and_gradients(self):
        depthwise = gmr_conv.GMR_Conv2d(2, 4, 3, padding=1)
        dense = gmr_conv.GMR_Conv2d(2, 4, 3, padding=1, dw_fwd=False)
        dense.load_state_dict(depthwise.state_dict(), strict=True)
        inputs = torch.randn(2, 2, 5, 5, requires_grad=True)
        dense_inputs = inputs.detach().clone().requires_grad_(True)
        output = depthwise(inputs)
        dense_output = dense(dense_inputs)
        torch.testing.assert_close(output, dense_output)
        output.square().mean().backward()
        dense_output.square().mean().backward()
        torch.testing.assert_close(inputs.grad, dense_inputs.grad)
        for name, param in depthwise.named_parameters():
            torch.testing.assert_close(param.grad, dict(dense.named_parameters())[name].grad)

    def test_transpose_default_constructor_and_forward(self):
        layer = gmr_conv.GMR_ConvTranspose2d(2, 4, 3)
        self.assertTrue(layer.transposed)
        self.assertEqual(layer.weight.shape, (2, 4, 2))
        inputs = torch.randn(2, 2, 5, 5, requires_grad=True)
        expected = F.conv_transpose2d(
            inputs, layer._make_weight_matrix(layer.weight), layer.bias,
        )
        output = layer(inputs)
        torch.testing.assert_close(output, expected)
        self.assertEqual(output.shape, (2, 4, 7, 7))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertTrue(torch.isfinite(layer.log_sigmas.grad).all())

    def test_transpose_existing_positional_arguments_and_output_size(self):
        layer = gmr_conv.GMR_ConvTranspose2d(
            4, 6, 3, 2, 1, 1, 2, False, "zeros", 2, True, False, False,
            False, torch.device("cpu"), torch.float32, clip_sigma=False,
        )
        self.assertTrue(layer.transposed)
        self.assertEqual(layer.groups, 2)
        self.assertEqual(layer.weight.shape, (4, 3, 2))
        self.assertIsNone(layer.bias)
        self.assertTrue(layer.train_index_mat)
        self.assertFalse(layer.inference_accelerate)
        self.assertFalse(layer.force_circular)
        self.assertFalse(layer.dw_fwd)
        self.assertFalse(layer.clip_sigma)
        self.assertEqual(layer.weight.device, torch.device("cpu"))
        self.assertEqual(layer.weight.dtype, torch.float32)
        inputs = torch.randn(2, 4, 5, 5)
        expected = F.conv_transpose2d(
            inputs, layer._make_weight_matrix(layer.weight),
            stride=2, padding=1, output_padding=0, groups=2,
        )
        torch.testing.assert_close(layer(inputs, output_size=(9, 9)), expected)

    def test_transpose_padding_does_not_set_output_padding(self):
        for stride in (1, 2):
            with self.subTest(stride=stride):
                layer = gmr_conv.GMR_ConvTranspose2d(2, 4, 3, padding=1, stride=stride)
                self.assertEqual(layer.output_padding, (0, 0))
                inputs = torch.randn(2, 2, 5, 5)
                expected = F.conv_transpose2d(
                    inputs, layer._make_weight_matrix(layer.weight), layer.bias,
                    stride=stride, padding=1, output_padding=0,
                )
                torch.testing.assert_close(layer(inputs), expected)


if __name__ == "__main__":
    unittest.main()
