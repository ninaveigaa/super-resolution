"""
src/preprocessing.py

Generic, chainable data preparation utilities.

Every transformation below:
    1. Accepts EITHER a file path (str/Path) OR an already-loaded numpy
       array as input -- so calls can be chained directly, e.g.:

           result = center_crop(
               cubic_interpolation(
                   convert_dtype(convert_ftype(path, "npy"), np.uint8),
                   scale_factor=1.5,
               ),
               size=(150, 150, 225),
           )

    2. ALWAYS saves the transformed array to disk (for traceability --
       every intermediate step leaves a file behind), using a default
       output path if none is given (only available when the input was a
       path; an explicit output_path is required when chaining from an
       in-memory array, since there is no original filename to derive
       a default from).
    3. ALWAYS returns the transformed array itself (not the output path),
       so it can be passed directly into the next function in a chain.
"""

from pathlib import Path

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Internal: file format registry (extensible -- add new formats here)
# ---------------------------------------------------------------------------
_LOADERS = {
    ".npy": lambda p: np.load(p),
    ".tif": lambda p: tifffile.imread(p),
    ".tiff": lambda p: tifffile.imread(p),
}

_SAVERS = {
    ".npy": lambda p, arr: np.save(p, arr),
    ".tif": lambda p, arr: tifffile.imwrite(p, arr),
    ".tiff": lambda p, arr: tifffile.imwrite(p, arr),
}


def _load_any(path) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(f"Unsupported input file type '{ext}'. Supported: {list(_LOADERS)}")
    return _LOADERS[ext](str(path))


def _save_any(path, array: np.ndarray):
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _SAVERS:
        raise ValueError(f"Unsupported output file type '{ext}'. Supported: {list(_SAVERS)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _SAVERS[ext](str(path), array)


class TrackedArray(np.ndarray):
    """A numpy array subclass that carries a `source_path` attribute,
    referencing the file it was last saved to.

    This is what makes chaining work smoothly: even though each function
    below returns an in-memory array (not a path), that array "remembers"
    where it was saved, so the NEXT function in the chain can still
    auto-derive a sensible default output filename -- without ever
    needing an explicit output_path at every step.

    Behaves exactly like a normal ndarray for all other purposes (slicing,
    math, etc.) -- the extra attribute just rides along.
    """

    def __new__(cls, input_array, source_path=None):
        obj = np.asarray(input_array).view(cls)
        obj.source_path = source_path
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.source_path = getattr(obj, "source_path", None)


def _resolve_input(input, required_ext: str = None):
    """Accepts either a file path or an already-loaded ndarray (including
    a TrackedArray, whose `source_path` is used to keep the chain's
    naming context alive).

    Returns (array, source_path_or_None).
    """
    if isinstance(input, np.ndarray):
        source_path = getattr(input, "source_path", None)
        return input, source_path

    input_path = Path(input)
    if required_ext is not None and input_path.suffix.lower() != required_ext:
        raise ValueError(
            f"Expected a '{required_ext}' input file, got '{input_path.suffix}'."
        )
    return _load_any(input_path), input_path


def _resolve_output_path(output_path, input_path, tag: str) -> Path:
    """Determines the output path to save to. If `output_path` is not
    given, derives a default from `input_path` (fails if input_path is
    None -- i.e., chaining from an in-memory array requires an explicit
    output_path)."""
    if output_path is not None:
        return Path(output_path)
    if input_path is None:
        raise ValueError(
            "output_path must be provided explicitly when chaining from an "
            "in-memory array (there is no original filename to derive a "
            "default output path from)."
        )
    return input_path.with_name(f"{input_path.stem}_{tag}{input_path.suffix}")


# ---------------------------------------------------------------------------
# Chainable transformations
# ---------------------------------------------------------------------------
def convert_dtype(input, dtype, output_path=None, min_val: float = None,
                   max_val: float = None) -> np.ndarray:
    """Converts an array (or the array stored in a .npy file) to `dtype`,
    via linear contrast stretching. The current dtype is detected
    automatically; `dtype` is the desired OUTPUT dtype (e.g. np.uint8,
    np.uint16), with its valid range derived via `np.iinfo`.

    `min_val`/`max_val` define the INPUT range mapped onto the output
    range; defaults to the array's own min/max if not given.

    Always saves the result to disk (as .npy) and returns the converted
    array, so this can be chained into further transformations.
    """
    volume, input_path = _resolve_input(input, required_ext=".npy")
    target_dtype = np.dtype(dtype)

    if not np.issubdtype(target_dtype, np.unsignedinteger):
        raise ValueError(
            f"dtype must be an unsigned integer type (e.g., uint8, uint16), got {target_dtype}."
        )

    out_info = np.iinfo(target_dtype)
    out_min, out_max = out_info.min, out_info.max

    if min_val is None:
        min_val = volume.min()
    if max_val is None:
        max_val = volume.max()
    if max_val <= min_val:
        raise ValueError(f"max_val ({max_val}) must be greater than min_val ({min_val}).")

    clipped = np.clip(volume, min_val, max_val)
    scaled = (clipped.astype("float64") - min_val) / (max_val - min_val) * (out_max - out_min) + out_min
    converted = scaled.astype(target_dtype)

    out_path = _resolve_output_path(output_path, input_path, str(target_dtype))
    _save_any(out_path, converted)
    return TrackedArray(converted, source_path=out_path)


def convert_ftype(input, ftype, output_path=None) -> np.ndarray:
    """Converts a file from its current format to `ftype` (e.g. 'npy',
    'tiff'), auto-detecting the input format from its extension (only
    applicable when `input` is a path; if an in-memory array is passed,
    this simply saves it in the requested format).

    Always saves the result to disk and returns the array, so this can be
    chained into further transformations.
    """
    array, input_path = _resolve_input(input)

    ftype = ftype if ftype.startswith(".") else f".{ftype}"
    ftype = ftype.lower()

    if output_path is None:
        if input_path is None:
            raise ValueError(
                "output_path must be provided explicitly when chaining from "
                "an in-memory array."
            )
        output_path = input_path.with_suffix(ftype)

    _save_any(output_path, array)
    return TrackedArray(array, source_path=Path(output_path))


def center_crop(input, size: tuple, output_path=None) -> np.ndarray:
    """Crops an array (or the array stored in a .npy file) to `size`,
    symmetrically around the center of each axis (i.e., excluding the
    outer edges).

    `size` must have the same number of dimensions as the input array
    (e.g. a 2D input requires a 2-tuple, a 3D input requires a 3-tuple).

    Always saves the result to disk (as .npy) and returns the cropped
    array, so this can be chained into further transformations.
    """
    volume, input_path = _resolve_input(input, required_ext=".npy")

    if len(size) != volume.ndim:
        raise ValueError(
            f"size has {len(size)} dimensions, but the input array has "
            f"{volume.ndim} dimensions. They must match (e.g., a 2D input "
            f"requires a 2-tuple, a 3D input requires a 3-tuple)."
        )

    slices = []
    for dim_size, target_size in zip(volume.shape, size):
        if target_size > dim_size:
            raise ValueError(
                f"Cannot crop: target size {target_size} is larger than "
                f"array dimension {dim_size}."
            )
        start = (dim_size - target_size) // 2
        slices.append(slice(start, start + target_size))

    cropped = volume[tuple(slices)]

    out_path = _resolve_output_path(output_path, input_path, "cropped")
    _save_any(out_path, cropped)
    return TrackedArray(cropped, source_path=out_path)


def cubic_interpolation(input, scale_factor=None, target_shape=None, output_path=None) -> np.ndarray:
    """Resizes an array (or the array stored in a .npy file) using
    tricubic interpolation (scipy.ndimage.zoom, order=3). Exactly one of
    `scale_factor` or `target_shape` must be provided.

    - `scale_factor`: a single number (applied to every axis) or a tuple
      with one factor per axis.
    - `target_shape`: the desired output shape; per-axis scale factors are
      derived automatically as target_shape[i] / volume.shape[i].

    Works for both upsampling (factor > 1) and downsampling (factor < 1).

    Always saves the result to disk (as .npy) and returns the resized
    array, so this can be chained into further transformations.
    """
    from scipy.ndimage import zoom

    volume, input_path = _resolve_input(input, required_ext=".npy")

    if (scale_factor is None) == (target_shape is None):
        raise ValueError(
            "Exactly one of scale_factor or target_shape must be provided (not both, not neither)."
        )

    if target_shape is not None:
        if len(target_shape) != volume.ndim:
            raise ValueError(
                f"target_shape has {len(target_shape)} dims, but volume has {volume.ndim} dims."
            )
        zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    else:
        zoom_factors = scale_factor

    resized = zoom(volume, zoom=zoom_factors, order=3)

    out_path = _resolve_output_path(output_path, input_path, "interpolated")
    _save_any(out_path, resized)
    return TrackedArray(resized, source_path=out_path)


def splitting(dataset, size: float, axis: int = -1, output_dir=None,
              base_name: str = None) -> dict:
    """Splits an array (or the array stored in a .npy file) into a
    training and a validation subset, by slicing contiguously along `axis`
    (the last axis by default).

    `size` is the FRACTION of the axis assigned to training (e.g. 0.8
    means 80% training / 20% validation).

    Always saves both resulting arrays to disk -- as
    `{output_dir}/{base_name}_train.npy` and `{base_name}_validation.npy`
    -- and returns {"train": array, "validation": array}, so each can be
    chained into further transformations independently.
    """
    volume, input_path = _resolve_input(dataset, required_ext=".npy")

    if not (0 < size < 1):
        raise ValueError(f"size (train fraction) must be between 0 and 1, got {size}.")

    if output_dir is None or base_name is None:
        if input_path is None:
            raise ValueError(
                "output_dir/base_name must be provided explicitly when "
                "chaining from an in-memory array with no tracked source path."
            )
        output_dir = output_dir if output_dir is not None else input_path.parent
        base_name = base_name if base_name is not None else input_path.stem
    output_dir = Path(output_dir)

    n = volume.shape[axis]
    split_idx = int(round(n * size))
    train_slices = [slice(None)] * volume.ndim
    val_slices = [slice(None)] * volume.ndim
    train_slices[axis] = slice(0, split_idx)
    val_slices[axis] = slice(split_idx, n)

    train_arr = volume[tuple(train_slices)]
    val_arr = volume[tuple(val_slices)]

    train_path = output_dir / f"{base_name}_train.npy"
    val_path = output_dir / f"{base_name}_validation.npy"
    _save_any(train_path, train_arr)
    _save_any(val_path, val_arr)

    return {
        "train": TrackedArray(train_arr, source_path=train_path),
        "validation": TrackedArray(val_arr, source_path=val_path),
    }