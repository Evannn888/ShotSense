"""Validated cached data, fixed parameter encoding and persistent photo splits."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.image_io import decode_semantic_image
from src.parameters import PARAM_MIN, PARAM_MAX, validate_parameters


class ParameterNormalizer:
    """Map absolute parameters to [-1, 1] without clipping or fitting data."""

    @staticmethod
    def _tensor(values, normalized=False):
        if values.ndim < 1 or values.shape[-1] != 6:
            raise ValueError("Parameters must have a final dimension of 6")
        if values.is_complex() or values.dtype == torch.bool:
            raise ValueError("Parameters must be real numbers")
        if not values.is_floating_point():
            values = values.to(torch.float32)
        low = torch.as_tensor(PARAM_MIN, dtype=values.dtype, device=values.device)
        high = torch.as_tensor(PARAM_MAX, dtype=values.dtype, device=values.device)
        minimum, maximum = (-1, 1) if normalized else (low, high)
        if not bool(torch.isfinite(values).all()) or bool(((values < minimum) | (values > maximum)).any()):
            raise ValueError("Parameters contain non-finite or out-of-range values")
        return values, low, high

    @classmethod
    def normalize(cls, values):
        if torch.is_tensor(values):
            values, low, high = cls._tensor(values)
            return 2 * (values - low) / (high - low) - 1
        values = validate_parameters(values)
        return 2 * (values - PARAM_MIN) / (PARAM_MAX - PARAM_MIN) - 1

    @classmethod
    def denormalize(cls, values):
        if torch.is_tensor(values):
            values, low, high = cls._tensor(values, normalized=True)
            return (values + 1) / 2 * (high - low) + low
        values = np.asarray(values)
        if values.dtype.kind not in "fiu" or values.ndim < 1 or values.shape[-1] != 6:
            raise ValueError("Normalized parameters must be numeric with a final dimension of 6")
        values = values.astype(np.float64)
        if not np.isfinite(values).all() or np.any((values < -1) | (values > 1)):
            raise ValueError("Normalized parameters must be finite and within [-1, 1]")
        return (values + 1) / 2 * (PARAM_MAX - PARAM_MIN) + PARAM_MIN


def _validate_ids(ids):
    if ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) == 0:
        raise ValueError("ids must be a nonempty one-dimensional Unicode array")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate photo IDs in metadata")
    for photo_id in ids.tolist():
        if (photo_id != photo_id.lower() or not photo_id.endswith(".dng")
                or not photo_id[:-4] or "/" in photo_id or "\\" in photo_id
                or any(ord(character) < 32 or ord(character) == 127 for character in photo_id)):
            raise ValueError("Unsafe or non-canonical photo ID: {!r}".format(photo_id))


class ShotSenseDataset(Dataset):
    """Return (ImageNet RGB image, unstandardized 132D features, normalized 6D label)."""

    def __init__(self, metadata_path, images_dir=None):
        self.metadata_path = Path(metadata_path)
        self.images_dir = Path(images_dir or self.metadata_path.parent / "images").resolve()
        with np.load(self.metadata_path, allow_pickle=False) as metadata:
            required = {"X", "Y", "Y_std", "ids"}
            if not required.issubset(metadata.files):
                raise ValueError("Metadata is missing: {}".format(sorted(required - set(metadata.files))))
            self.X, self.Y, self.Y_std, self.ids = [metadata[key].copy() for key in ("X", "Y", "Y_std", "ids")]
            self.data_version = None
            if "config_hash" in metadata.files:
                config_hash = metadata["config_hash"]
                if config_hash.shape != () or config_hash.dtype.kind != "U" or not config_hash.item():
                    raise ValueError("config_hash must be a nonempty Unicode scalar")
                self.data_version = config_hash.item()
        _validate_ids(self.ids)
        for name, values, width in (("X", self.X, 132), ("Y", self.Y, 6), ("Y_std", self.Y_std, 6)):
            if values.dtype != np.float32 or values.shape != (len(self.ids), width):
                raise ValueError("{} must be float32 with shape ({}, {})".format(name, len(self.ids), width))
            if not np.isfinite(values).all():
                raise ValueError("{} contains non-finite values".format(name))
        validate_parameters(self.Y)
        if np.any(self.Y_std < 0):
            raise ValueError("Y_std must be nonnegative")
        self.data_hash = hashlib.sha256(self.metadata_path.read_bytes()).hexdigest()
        self.scope = None
        manifest_path = self.metadata_path.parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict) or manifest.get("status") != "complete":
                raise ValueError("Only a complete preprocessing manifest is trainable")
            if manifest.get("metadata_sha256") != self.data_hash:
                raise ValueError("Metadata SHA256 does not match the preprocessing manifest")
            if 'support_audit_sha256' in manifest:
                support_path=self.metadata_path.parent/'input_support_audit.json'
                if hashlib.sha256(support_path.read_bytes()).hexdigest()!=manifest['support_audit_sha256']:
                    raise ValueError('Input support audit SHA256 does not match the manifest')
                support=json.loads(support_path.read_text())
                if set(support['unsupported_images']) & set(self.ids.tolist()):
                    raise ValueError('Unsupported RAW input leaked into training metadata')
            self.scope = manifest.get("scope")
            if self.scope not in ("pilot", "full"):
                raise ValueError("Manifest scope must be pilot or full")
            manifest_version = manifest.get("config_hash")
            if not isinstance(manifest_version, str) or not manifest_version:
                raise ValueError("manifest config_hash must be a nonempty string")
            if self.data_version is not None and manifest_version != self.data_version:
                raise ValueError("Metadata and manifest config_hash disagree")
            self.data_version = manifest_version
        self.labels = torch.from_numpy(ParameterNormalizer.normalize(self.Y).astype(np.float32))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_path = (self.images_dir / (Path(self.ids[index]).stem + ".jpg")).resolve()
        if not image_path.is_relative_to(self.images_dir):
            raise ValueError("Image resolves outside the configured images directory")
        image = torch.from_numpy(decode_semantic_image(image_path))
        return image, torch.from_numpy(self.X[index]), self.labels[index]


def _validate_splits(splits, ids):
    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        raise ValueError("Splits must contain train, val and test")
    if any(not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values)
           for values in splits.values()):
        raise ValueError("Every split must contain a nonempty list of photo IDs")
    flattened = [photo_id for values in splits.values() for photo_id in values]
    if len(set(flattened)) != len(flattened) or set(flattened) != set(ids):
        raise ValueError("Splits must partition all photo IDs exactly once")


def load_or_create_splits(dataset, output_path=None, *, data_version=None, groups=None, pilot=False, seed=42):
    """Persist approximately 80/10/10 splits; refuse stale or incompatible files.

    ``groups`` optionally maps every photo ID to its burst/near-duplicate group.
    Pilot runs use splits.pilot.json and cannot write the formal splits.json.
    """
    output_path = Path(output_path or dataset.metadata_path.parent / ("splits.pilot.json" if pilot else "splits.json"))
    if dataset.scope == "pilot" and not pilot:
        raise ValueError("Pilot metadata requires pilot=True and cannot define a formal split")
    if pilot and output_path.name == "splits.json":
        raise ValueError("Pilot splits must use a separate file such as splits.pilot.json")
    data_version = data_version or dataset.data_version
    if not isinstance(data_version, str) or not data_version:
        raise ValueError("A processing data_version/config_hash is required")
    if dataset.data_version is not None and data_version != dataset.data_version:
        raise ValueError("Explicit data_version disagrees with metadata config_hash")
    ids = dataset.ids.tolist()
    if groups is None:
        groups = dict(zip(ids, ids))
        grouping = "photo_id_only; near-duplicate grouping unavailable"
    else:
        if (not isinstance(groups, dict) or set(groups) != set(ids)
                or any(not isinstance(group, str) or not group for group in groups.values())):
            raise ValueError("groups must map every photo ID to a nonempty group string")
        grouping = "provided_groups"
    grouped_ids = {}
    for photo_id in sorted(ids):
        grouped_ids.setdefault(groups[photo_id], []).append(photo_id)
    if len(grouped_ids) < 3:
        raise ValueError("At least three distinct photo groups are needed for train/val/test")
    group_hash = hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest()
    expected = {"schema_version": 1, "scope": "pilot" if pilot else "full", "seed": seed,
                "ratios": [0.8, 0.1, 0.1], "data_version": data_version,
                "data_hash": dataset.data_hash, "groups_hash": group_hash, "grouping": grouping}
    if output_path.exists():
        saved = json.loads(output_path.read_text())
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing splits are stale or use a different configuration; use a new split file")
        _validate_splits(saved.get("splits"), ids)
        membership = {photo_id: split for split, members in saved["splits"].items() for photo_id in members}
        if any(len({membership[photo_id] for photo_id in members}) != 1 for members in grouped_ids.values()):
            raise ValueError("Saved splits leak a photo group across sets")
        return saved["splits"]
    rng = np.random.default_rng(seed)
    group_order = rng.permutation(sorted(grouped_ids)).tolist()
    group_order.sort(key=lambda group: -len(grouped_ids[group]))
    names = ("train", "val", "test")
    splits = {name: [] for name in names}
    counts = np.zeros(3, dtype=np.int64)
    targets = len(ids) * np.array([0.8, 0.1, 0.1])
    # ponytail: greedy whole-group allocation; optimize ratios only if large bursts make them unusable.
    for position, group in enumerate(group_order):
        empty = np.flatnonzero(counts == 0)
        selected = int(empty[0]) if len(group_order) - position == len(empty) else int(np.argmax(targets - counts))
        splits[names[selected]].extend(grouped_ids[group])
        counts[selected] += len(grouped_ids[group])
    splits = {name: sorted(values) for name, values in splits.items()}
    _validate_splits(splits, ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output_path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(dict(expected, splits=splits), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic exclusive creation prevents two runs from silently replacing each other's split.
        os.link(temporary, output_path)
    except FileExistsError:
        return load_or_create_splits(dataset, output_path, data_version=data_version, groups=None if grouping.startswith("photo_id") else groups, pilot=pilot, seed=seed)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return splits


def create_dataloaders(dataset, splits, batch_size=32, num_workers=0, seed=42):
    """Build loaders over the same validated image-level partition."""
    _validate_splits(splits, dataset.ids.tolist())
    index_by_id = {photo_id: index for index, photo_id in enumerate(dataset.ids.tolist())}
    return {
        name: DataLoader(Subset(dataset, [index_by_id[photo_id] for photo_id in ids]),
                         batch_size=batch_size, shuffle=name == "train", num_workers=num_workers,
                         generator=torch.Generator().manual_seed(seed))
        for name, ids in splits.items()
    }
