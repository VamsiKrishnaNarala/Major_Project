"""Model loading, image validation and result normalisation for the Streamlit app.

This module has no Streamlit dependency so it can be unit-tested on its own.

Design notes
------------
* The classifier is the official ``ChestXrayClassifier`` from the model repository's
  ``inference.py``. Nothing is re-implemented here: the official code does the
  preprocessing (RGB, 224x224 bilinear resize, divide by 255), the out-of-distribution
  (OOD) gate, classification and the 0.75 confidence check.
* ``predict()`` in the official ``inference.py`` accepts a file path, a PIL image or a
  uint8 NumPy array. This app passes a validated RGB ``PIL.Image``.
* Only four files are downloaded from the Hugging Face Hub: ``inference.py``,
  ``metadata.json``, ``lung_ood_detector.joblib`` and ``DenseNet121_FT_best.keras``.
  The other six checkpoints in the repository are never fetched.

Trust assumption
----------------
``inference.py`` is executed and ``lung_ood_detector.joblib`` is unpickled. Both come from
the model author's repository, so you are trusting that repository (and Hugging Face).
Set the ``MODEL_REVISION`` environment variable to a commit hash to pin the exact
files that are downloaded instead of following the ``main`` branch.
"""

from __future__ import annotations

import importlib.util
import io
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------------------------------- constants

REPO_ID = "FarhanKO/lung-disease-prediction"
REPO_URL = f"https://huggingface.co/{REPO_ID}"
DEFAULT_REVISION = "main"

# Files the official inference pipeline needs (DenseNet121_FT is both the classifier and
# the embedding model used by the OOD gate, per the model card).
REQUIRED_FILES = (
    "inference.py",
    "metadata.json",
    "lung_ood_detector.joblib",
    "DenseNet121_FT_best.keras",
)

# Exact order used by the model.
CLASS_NAMES = (
    "Covid-19",
    "Emphysema",
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral",
    "Tuberculosis",
)

CONFIDENCE_THRESHOLD = 0.75
VALID_STATUSES = ("accepted", "review", "rejected")

ALLOWED_FORMATS = ("JPEG", "PNG")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # keep in sync with server.maxUploadSize in config.toml
MAX_PIXELS = 40_000_000  # guards against decompression bombs
MIN_SIDE = 32

_PREDICT_LOCK = threading.Lock()  # one inference at a time keeps memory use predictable


# --------------------------------------------------------------------------- exceptions


class ClassifierError(Exception):
    """Base class for errors that are safe to show to the user."""


class InvalidImageError(ClassifierError):
    """The uploaded file is not a usable JPEG/PNG image."""


class ModelLoadError(ClassifierError):
    """The model files could not be downloaded or initialised."""


class InferenceError(ClassifierError):
    """The classifier raised an exception while predicting."""


class MalformedResultError(ClassifierError):
    """The classifier returned something this app does not understand."""


# --------------------------------------------------------------------------- images


def load_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an 8-bit RGB ``PIL.Image``.

    Raises ``InvalidImageError`` with a user-friendly message for empty, oversized,
    corrupt, unsupported, too-small or too-large images.
    """
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise InvalidImageError(f"The file is larger than the {limit_mb} MB limit.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
        if fmt not in ALLOWED_FORMATS:
            raise InvalidImageError(
                f"Unsupported image format ({fmt or 'unknown'}). Please upload a JPG, JPEG or PNG file."
            )
        if min(width, height) < MIN_SIDE:
            raise InvalidImageError(
                f"The image is too small ({width}x{height}px); each side must be at least {MIN_SIDE}px."
            )
        if width * height > MAX_PIXELS:
            raise InvalidImageError(
                f"The image is too large ({width}x{height}px); the limit is {MAX_PIXELS:,} pixels."
            )

        # Re-open for a full decode: this also surfaces truncated or corrupt files.
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return _to_rgb(img)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("The file could not be read as an image.") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise InvalidImageError(f"The image appears to be corrupt or truncated ({exc}).") from exc


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert any supported PIL mode to 8-bit RGB.

    Plain ``convert("RGB")`` saturates 16-bit grayscale PNGs (a common X-ray export), so
    those are rescaled to 8 bits first.
    """
    if img.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.asarray(img).astype(np.float32)
        peak = float(arr.max()) if arr.size else 0.0
        if peak > 255.0:
            arr = arr / 257.0 if peak <= 65535.0 else arr * (255.0 / peak)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    elif img.mode == "F":
        arr = np.asarray(img).astype(np.float32)
        peak = float(arr.max()) if arr.size else 0.0
        if peak > 0:
            arr = arr * (255.0 / peak) if peak > 1.0 else arr * 255.0
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    return img.convert("RGB")


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class ClassificationResult:
    """Validated, UI-ready view of the classifier output."""

    status: str  # "accepted" | "review" | "rejected"
    label: str | None  # None when rejected
    confidence: float | None  # raw softmax top-class score, None when rejected
    p_abnormal: float | None  # 1 - P(Normal) as returned by the model, if present
    top_3: tuple[tuple[str, float], ...]
    probabilities: Mapping[str, float]
    message: str | None = None  # extra explanation (rejection reason, downgrade note)

    @property
    def is_rejected(self) -> bool:
        return self.status == "rejected"


def _as_probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise MalformedResultError(f"'{name}' is not a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise MalformedResultError(f"'{name}' must be between 0 and 1, got {number!r}")
    return number


def _parse_probabilities(raw: Any) -> dict[str, float]:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, Mapping):
        raise MalformedResultError("'probabilities' must be a mapping of class name to score.")
    parsed: dict[str, float] = {}
    for name, value in raw.items():
        if name not in CLASS_NAMES:
            raise MalformedResultError(f"Unknown class in 'probabilities': {name!r}")
        parsed[name] = _as_probability(value, f"probabilities[{name}]")
    return parsed


def _parse_top_3(raw: Any, probabilities: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    if raw in (None, [], ()):
        # Derive from the full distribution when the model omitted top_3.
        ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return tuple(ranked[:3])
    if not isinstance(raw, (list, tuple)):
        raise MalformedResultError("'top_3' must be a list of (class, score) pairs.")
    parsed = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise MalformedResultError(f"Invalid 'top_3' entry: {item!r}")
        name, score = item
        if name not in CLASS_NAMES:
            raise MalformedResultError(f"Unknown class in 'top_3': {name!r}")
        parsed.append((name, _as_probability(score, f"top_3[{name}]")))
    return tuple(parsed[:3])


def normalize_result(raw: Any) -> ClassificationResult:
    """Validate the raw ``predict()`` output and return a ``ClassificationResult``.

    * ``rejected`` results never carry a label, confidence or probabilities, even if the
      raw dict happens to contain some: a class is never invented for a rejected image.
    * An ``accepted`` result whose confidence is below the threshold is downgraded to
      ``review`` (with an explanatory message) rather than trusted blindly.
    * Anything missing or malformed raises ``MalformedResultError``.
    """
    if not isinstance(raw, Mapping):
        raise MalformedResultError(f"Expected a dict from the classifier, got {type(raw).__name__}.")

    status = raw.get("status")
    if status not in VALID_STATUSES:
        raise MalformedResultError(f"Unknown or missing status: {status!r}")

    if status == "rejected":
        message = raw.get("message")
        return ClassificationResult(
            status="rejected",
            label=None,
            confidence=None,
            p_abnormal=None,
            top_3=(),
            probabilities={},
            message=message if isinstance(message, str) and message else None,
        )

    label = raw.get("label")
    if label not in CLASS_NAMES:
        raise MalformedResultError(f"Unknown or missing label: {label!r}")
    confidence = _as_probability(raw.get("confidence"), "confidence")

    p_abnormal_raw = raw.get("p_abnormal")
    p_abnormal = None if p_abnormal_raw is None else _as_probability(p_abnormal_raw, "p_abnormal")

    probabilities = _parse_probabilities(raw.get("probabilities"))
    top_3 = _parse_top_3(raw.get("top_3"), probabilities)

    message = None
    if status == "accepted" and confidence < CONFIDENCE_THRESHOLD:
        status = "review"
        message = (
            f"The classifier reported 'accepted' but the score ({confidence:.1%}) is below the "
            f"{CONFIDENCE_THRESHOLD:.0%} threshold, so this result is treated as needing review."
        )

    return ClassificationResult(
        status=status,
        label=label,
        confidence=confidence,
        p_abnormal=p_abnormal,
        top_3=top_3,
        probabilities=probabilities,
        message=message,
    )


@dataclass(frozen=True)
class StatusInfo:
    title: str
    severity: str  # "success" | "warning" | "error" (maps to st.success / st.warning / st.error)
    explanation: str


def status_info(status: str) -> StatusInfo:
    """Human-readable title and explanation for each status."""
    if status == "accepted":
        return StatusInfo(
            "Accepted",
            "success",
            f"The top-class score is at least {CONFIDENCE_THRESHOLD:.0%}. This is a model output "
            "for research purposes only, not a diagnosis.",
        )
    if status == "review":
        return StatusInfo(
            "Needs review",
            "warning",
            f"The top-class score is below {CONFIDENCE_THRESHOLD:.0%}. The class shown is only the "
            "model's best guess and is not confirmed. On the model's own test set, flagged images "
            "were correct only about 63% of the time.",
        )
    if status == "rejected":
        return StatusInfo(
            "Rejected",
            "error",
            "The out-of-distribution check flagged this image as unsuitable, so it was not "
            "classified. It may not be a frontal chest X-ray.",
        )
    raise ValueError(f"Unknown status: {status!r}")


def format_percent(value: float | None, digits: int = 1) -> str:
    """Format a 0-1 score as a percentage string (``"n/a"`` for ``None``)."""
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def probability_rows(result: ClassificationResult) -> list[tuple[str, float]]:
    """All class scores sorted from highest to lowest (empty if none were returned)."""
    return sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------- model


def get_revision() -> str:
    """Hub revision to download: ``MODEL_REVISION`` if set, otherwise ``main``."""
    return os.environ.get("MODEL_REVISION", "").strip() or DEFAULT_REVISION


def load_classifier(repo_id: str = REPO_ID, revision: str | None = None):
    """Download the needed artifacts and return the official ``ChestXrayClassifier``.

    Slow on a cold start (downloads ~40 MB, imports Keras + PyTorch); callers should cache
    the returned object (the Streamlit app uses ``st.cache_resource``).
    """
    revision = revision or get_revision()
    # Must be set before keras is imported (the official inference.py also defaults to
    # torch when it is installed). An explicit KERAS_BACKEND value is respected.
    os.environ.setdefault("KERAS_BACKEND", "torch")

    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(REQUIRED_FILES),
        )
    except Exception as exc:  # network, 404, rate limit, disk full ...
        raise ModelLoadError(
            f"Could not download the model files from Hugging Face ({type(exc).__name__}: {exc}). "
            "Check your internet connection and try again."
        ) from exc

    try:
        script = os.path.join(local_dir, "inference.py")
        spec = importlib.util.spec_from_file_location("chest_xray_inference", script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # A local folder as `source` makes the official class read the files we downloaded
        # instead of fetching anything else from the Hub.
        return module.ChestXrayClassifier(local_dir)
    except Exception as exc:  # missing dependency, incompatible version, corrupt file ...
        raise ModelLoadError(
            f"The model was downloaded but could not be initialised ({type(exc).__name__}: {exc}). "
            "This is usually a dependency-version mismatch; see the README troubleshooting section."
        ) from exc


def classify(classifier: Any, image: Image.Image) -> ClassificationResult:
    """Run the official ``predict()`` on a PIL image and validate its output."""
    try:
        with _PREDICT_LOCK:
            raw = classifier.predict(image)
    except Exception as exc:
        raise InferenceError(f"Inference failed ({type(exc).__name__}: {exc}).") from exc
    return normalize_result(raw)
