"""Shared linear RAW and display-sRGB development, with explicit D50 conversion."""

from enum import Enum

import colour
import numpy as np
import rawpy


# LibRaw 0.22.1 colorconst.cpp and convert_to_rgb() ICC matrix construction:
# https://github.com/LibRaw/LibRaw/blob/0.22.1/src/tables/colorconst.cpp
# https://github.com/LibRaw/LibRaw/blob/0.22.1/src/postprocessing/postprocessing_utils_dcrdefs.cpp
# LibRaw calls the output "ProPhoto D65", but this matrix already includes its
# D50 adaptation. Applying another D65 -> D50 adaptation would be incorrect.
LIBRAW_SRGB_TO_PROPHOTO = np.array([
    [0.529317, 0.330092, 0.140588],
    [0.098368, 0.873465, 0.028169],
    [0.016879, 0.117663, 0.865457],
])
LIBRAW_SRGB_TO_XYZ_D50 = np.array([
    [0.436083, 0.385083, 0.143055],
    [0.222507, 0.716888, 0.060608],
    [0.013930, 0.097097, 0.714022],
])
PROPHOTO_TO_XYZ_D50 = LIBRAW_SRGB_TO_XYZ_D50 @ np.linalg.inv(LIBRAW_SRGB_TO_PROPHOTO)
D50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
LIBRAW_PROPHOTO = colour.RGB_Colourspace(
    "LibRaw ProPhoto with D50 ICC matrix",
    primaries=colour.RGB_COLOURSPACES["ProPhoto RGB"].primaries,
    whitepoint=D50,
    whitepoint_name="D50",
    matrix_RGB_to_XYZ=PROPHOTO_TO_XYZ_D50,
    matrix_XYZ_to_RGB=np.linalg.inv(PROPHOTO_TO_XYZ_D50),
)

# None explicitly preserves camera metadata/default geometry; never expert WB.
COMMON_RAW_PARAMS = {
    "demosaic_algorithm": rawpy.DemosaicAlgorithm.AHD,
    "half_size": False,
    "four_color_rgb": False,
    "dcb_iterations": 0,
    "dcb_enhance": False,
    "fbdd_noise_reduction": rawpy.FBDDNoiseReductionMode.Off,
    "noise_thr": None,
    "median_filter_passes": 0,
    "use_camera_wb": True,
    "use_auto_wb": False,
    "user_wb": None,
    "user_flip": None,
    "user_black": None,
    "user_cblack": None,
    "user_sat": None,
    "no_auto_bright": True,
    "auto_bright_thr": 0.01,
    "adjust_maximum_thr": 0.0,
    "bright": 1.0,
    "highlight_mode": rawpy.HighlightMode.Clip,
    "exp_shift": None,
    "exp_preserve_highlights": 0.0,
    "no_auto_scale": False,
    "chromatic_aberration": (1.0, 1.0),
    "bad_pixels_path": None,
}
PROPHOTO_PARAMS = {"output_color": rawpy.ColorSpace.ProPhoto, "output_bps": 16, "gamma": (1.0, 1.0)}
SRGB_PARAMS = {"output_color": rawpy.ColorSpace.sRGB, "output_bps": 8, "gamma": (2.4, 12.92)}

PIPELINE_CONFIG = {
    "version": "linear-libraw-d50-v1",
    "dependencies": {
        "rawpy": rawpy.__version__,
        "libraw": list(rawpy.libraw_version),
        "colour-science": colour.__version__,
        "numpy": np.__version__,
    },
    "common_raw_params": {k: v.name if isinstance(v, Enum) else v for k, v in COMMON_RAW_PARAMS.items()},
    "outputs": {
        name: {k: v.name if isinstance(v, Enum) else v for k, v in params.items()}
        for name, params in (("prophoto16", PROPHOTO_PARAMS), ("srgb8", SRGB_PARAMS))
    },
    "matrix_source_libraw": "0.22.1",
    "prophoto_to_xyz_d50": PROPHOTO_TO_XYZ_D50.tolist(),
    "lab_illuminant_xy": D50.tolist(),
    "apply_cctf_decoding": False,
    "chromatic_adaptation_transform": None,
    "missing_camera_wb": "reject",
}


def develop_dng_outputs(dng_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read once, return full-resolution linear uint16 ProPhoto and uint8 sRGB.

    Both RGB arrays use the same camera orientation and RAW visible area. The
    caller owns resizing and JPEG encoding. LibRaw can silently select auto WB
    when camera WB is absent even with use_auto_wb=False, so reject that case.
    """
    with rawpy.imread(str(dng_path)) as raw:
        camera_wb = np.asarray(raw.camera_whitebalance[:3], dtype=np.float64)
        if camera_wb.shape != (3,) or not np.isfinite(camera_wb).all() or np.any(camera_wb <= 0.00001):
            raise ValueError("Missing or invalid camera white balance; automatic fallback is disabled")
        prophoto16 = raw.postprocess(**COMMON_RAW_PARAMS, **PROPHOTO_PARAMS)
        srgb8 = raw.postprocess(**COMMON_RAW_PARAMS, **SRGB_PARAMS)
    if prophoto16.shape != srgb8.shape or prophoto16.ndim != 3 or prophoto16.shape[-1] != 3:
        raise ValueError("RAW outputs must have matching H x W x 3 geometry")
    if prophoto16.dtype != np.uint16 or srgb8.dtype != np.uint8:
        raise TypeError("Unexpected RAW output bit depth")
    return prophoto16, srgb8


def prophoto16_to_lab_d50(image: np.ndarray) -> np.ndarray:
    """Convert linear uint16 [0,65535] or linear float [0,1] RGB to float32 Lab.

    Float input supports area resizing after normalization without 16-bit
    requantization. No transfer-function decoding or additional whitepoint
    adaptation is applied. The small (<0.02) neutral a/b residual comes from
    LibRaw's rounded ICC matrix, not a second chromatic adaptation.
    """
    image = np.asarray(image)
    if image.ndim < 2 or image.shape[-1] != 3 or image.size == 0:
        raise ValueError("Expected nonempty RGB array with last dimension 3")
    if image.dtype == np.uint16:
        linear = image.astype(np.float32) / 65535.0
    elif np.issubdtype(image.dtype, np.floating):
        linear = image
    else:
        raise TypeError("Expected uint16 or normalized linear floating-point RGB")
    # INTER_AREA's float32 weights can put a white pixel one ULP above 1.
    # Reject actual out-of-domain inputs, allowing only arithmetic roundoff.
    tolerance = 8 * np.finfo(linear.dtype).eps
    if not np.isfinite(linear).all() or np.any(linear < -tolerance) or np.any(linear > 1 + tolerance):
        raise ValueError("Linear RGB must be finite and within [0,1]")
    linear = np.clip(linear, 0, 1)
    xyz = colour.RGB_to_XYZ(
        linear,
        colourspace=LIBRAW_PROPHOTO,
        illuminant=D50,
        chromatic_adaptation_transform=None,
        apply_cctf_decoding=False,
    )
    return colour.XYZ_to_Lab(xyz, illuminant=D50).astype(np.float32)
