"""Numerical color checks; real-DNG acceptance is recorded in color_audit.json."""

import json
import unittest
from unittest.mock import patch

import colour
import numpy as np

from src.color_pipeline import (
    COMMON_RAW_PARAMS, D50, LIBRAW_SRGB_TO_PROPHOTO, LIBRAW_SRGB_TO_XYZ_D50,
    PIPELINE_CONFIG, PROPHOTO_PARAMS, PROPHOTO_TO_XYZ_D50, SRGB_PARAMS,
    develop_dng_outputs, prophoto16_to_lab_d50,
)


class ColorPipelineTests(unittest.TestCase):
    def test_matrix_neutrals_and_linear_transfer(self):
        # Independent source-matrix relationship and published ProPhoto matrix.
        np.testing.assert_allclose(
            PROPHOTO_TO_XYZ_D50 @ LIBRAW_SRGB_TO_PROPHOTO,
            LIBRAW_SRGB_TO_XYZ_D50, rtol=0, atol=1e-12,
        )
        np.testing.assert_allclose(
            PROPHOTO_TO_XYZ_D50,
            colour.RGB_COLOURSPACES["ProPhoto RGB"].matrix_RGB_to_XYZ,
            rtol=0, atol=1.3e-4,
        )
        rgb = np.array([[0, 0, 0], [1, 1, 1], [.18, .18, .18], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        actual = prophoto16_to_lab_d50(rgb)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual[0], 0, rtol=0, atol=1e-6)
        self.assertAlmostEqual(float(actual[1, 0]), 100, places=3)
        self.assertAlmostEqual(float(actual[2, 0]), 116 * .18 ** (1 / 3) - 16, places=3)
        np.testing.assert_allclose(actual[:3, 1:], 0, rtol=0, atol=.02)
        expected = colour.XYZ_to_Lab(rgb @ PROPHOTO_TO_XYZ_D50.T, illuminant=D50)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)
        codes = np.array([[0, 32768, 65535]], dtype=np.uint16)
        np.testing.assert_allclose(prophoto16_to_lab_d50(codes), prophoto16_to_lab_d50(codes.astype(np.float32) / 65535), rtol=0, atol=1e-6)
        json.dumps(PIPELINE_CONFIG, allow_nan=False)
        rounded_white = np.full((2, 3), np.nextafter(np.float32(1), np.float32(2)), dtype=np.float32)
        np.testing.assert_allclose(prophoto16_to_lab_d50(rounded_white), prophoto16_to_lab_d50(np.ones((2, 3), dtype=np.float32)))

    def test_invalid_inputs_and_missing_camera_wb(self):
        for image in (np.zeros((0, 3)), np.zeros((2, 4)), np.array([[np.nan, 0., 0.]]), np.array([[1.1, 0., 0.]]), np.array([[-.1, 0., 0.]])):
            with self.assertRaises(ValueError):
                prophoto16_to_lab_d50(image)
        with self.assertRaises(TypeError):
            prophoto16_to_lab_d50(np.zeros((2, 3), dtype=np.uint8))
        with patch("src.color_pipeline.rawpy.imread") as imread:
            raw = imread.return_value.__enter__.return_value
            raw.camera_whitebalance = [0, 1, 1, 0]
            with self.assertRaisesRegex(ValueError, "white balance"):
                develop_dng_outputs("missing-wb.dng")
            raw.postprocess.assert_not_called()

    def test_single_read_and_explicit_two_outputs(self):
        with patch("src.color_pipeline.rawpy.imread") as imread:
            raw = imread.return_value.__enter__.return_value
            raw.camera_whitebalance = [2, 1, 1.5, 0]
            raw.postprocess.side_effect = [np.zeros((4, 6, 3), dtype=np.uint16), np.zeros((4, 6, 3), dtype=np.uint8)]
            physical, semantic = develop_dng_outputs("sample.dng")
            imread.assert_called_once_with("sample.dng")
            self.assertEqual(physical.shape, semantic.shape)
            calls = raw.postprocess.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0].kwargs, {**COMMON_RAW_PARAMS, **PROPHOTO_PARAMS})
            self.assertEqual(calls[1].kwargs, {**COMMON_RAW_PARAMS, **SRGB_PARAMS})


if __name__ == "__main__":
    unittest.main()
