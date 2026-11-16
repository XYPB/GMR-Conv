"""Quarter-turn invariance of the full, randomly initialized GMR-ResNets."""

import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

from test_gmr_conv import gmr_conv


# Import the real network without importing optional torchvision/torchio transforms.
source_dir = Path(__file__).resolve().parents[1] / "src/GMR_Conv"
package = types.ModuleType("_gmr_rotation_tests")
package.__path__ = [str(source_dir)]
spec = importlib.util.spec_from_file_location(
    "_gmr_rotation_tests.gmr_resnet", source_dir / "gmr_resnet.py"
)
gmr_resnet = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {
    "_gmr_rotation_tests": package,
    "_gmr_rotation_tests.gmr_conv": gmr_conv,
}):
    spec.loader.exec_module(gmr_resnet)


class GMRResNetRotationTests(unittest.TestCase):
    def check_rotations(self, size, dtype, training):
        # Default pooling divides the spatial size by 16. Use aligned square
        # inputs so the pooling windows themselves commute with quarter turns.
        for builder in (gmr_resnet.gmr_resnet18, gmr_resnet.gmr_resnet50):
            for seed in (0, 1):
                for dw_fwd in (True, False):
                    for clip_sigma in (True, False):
                        config = dict(
                            model=builder.__name__, seed=seed, size=size,
                            dtype=str(dtype), training=training,
                            dw_fwd=dw_fwd, clip_sigma=clip_sigma,
                        )
                        with self.subTest(**config):
                            # The network does not expose these layer options.
                            # Set them through the actual convolution constructor,
                            # retaining the class type used by weight initialization.
                            class ConfiguredConv(gmr_conv.GMR_Conv2d):
                                def __init__(self, *args, **kwargs):
                                    super().__init__(
                                        *args, dw_fwd=dw_fwd,
                                        clip_sigma=clip_sigma, **kwargs,
                                    )

                            torch.manual_seed(seed)
                            with patch.object(gmr_resnet, "GMR_Conv2d", ConfiguredConv):
                                model = builder().to(dtype=dtype)
                            convs = [m for m in model.modules()
                                     if isinstance(m, gmr_conv.GMR_Conv2d)]
                            self.assertEqual(len(convs), 17)
                            for conv in convs:
                                self.assertIs(conv.dw_fwd, dw_fwd)
                                self.assertIs(conv.clip_sigma, clip_sigma)
                            model.train(training)
                            # Keep the input samples identical across precisions.
                            inputs = torch.randn(2, 3, size, size).to(dtype=dtype)
                            with torch.no_grad():
                                expected = model(inputs)
                                self.assertTrue(torch.isfinite(expected).all())
                                # Ensure the tested classifier is input-dependent.
                                self.assertGreater(
                                    (expected[0] - expected[1]).abs().max().item(), 1e-6
                                )
                                errors = {}
                                relative_errors = {}
                                for turns in (1, 2, 3):
                                    actual = model(torch.rot90(inputs, turns, (-2, -1)))
                                    errors[str(90 * turns)] = (
                                        actual - expected
                                    ).abs().max().item()
                                    # Normalize per image: randomly initialized
                                    # logits can be large and individual logits
                                    # near zero are ill-conditioned for relative error.
                                    scales = expected.abs().amax(dim=1).clamp_min(
                                        torch.finfo(dtype).eps
                                    )
                                    relative_errors[str(90 * turns)] = (
                                        (actual - expected).abs().amax(dim=1) / scales
                                    ).max().item()
                                    if dtype == torch.float64:
                                        torch.testing.assert_close(
                                            actual, expected, atol=1e-10, rtol=1e-10,
                                        )
                                    else:
                                        self.assertLessEqual(
                                            relative_errors[str(90 * turns)], 1e-5,
                                        )
                            print(json.dumps(dict(
                                config, max_abs_logit_error=errors,
                                max_relative_linf_error=relative_errors,
                            )), flush=True)

    def test_eval_float32_224(self):
        self.check_rotations(224, torch.float32, False)

    def test_train_float64_32(self):
        self.check_rotations(32, torch.float64, True)

    def test_eval_float64_224(self):
        self.check_rotations(224, torch.float64, False)


if __name__ == "__main__":
    unittest.main()
