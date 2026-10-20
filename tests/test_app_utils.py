"""Unit and UI tests that need neither the real model nor network access.

Run from the project folder with:  python -m pytest -v

The UI tests drive the real ``app.py`` with Streamlit's ``AppTest`` and a *mock classifier*
that returns dictionaries in the same shape as the official ``inference.py``. These are test
doubles for exercising the interface; they are not model predictions.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_utils as mu  # noqa: E402

# --------------------------------------------------------------------------- helpers


def make_image_bytes(fmt: str = "PNG", mode: str = "RGB", size=(64, 64)) -> bytes:
    img = Image.new(mode, size)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


PROBS = {
    "Covid-19": 0.627,
    "Emphysema": 0.134,
    "Normal": 0.066,
    "Pneumonia-Bacterial": 0.08,
    "Pneumonia-Viral": 0.05,
    "Tuberculosis": 0.043,
}

REVIEW_RAW = {
    "status": "review",
    "label": "Covid-19",
    "confidence": 0.627,
    "p_abnormal": 0.934,
    "top_3": [("Covid-19", 0.627), ("Emphysema", 0.134), ("Normal", 0.066)],
    "probabilities": PROBS,
}
ACCEPTED_RAW = {
    "status": "accepted",
    "label": "Tuberculosis",
    "confidence": 0.99,
    "p_abnormal": 0.99,
    "top_3": [("Tuberculosis", 0.99), ("Normal", 0.006), ("Emphysema", 0.004)],
    "probabilities": {
        "Covid-19": 0.0,
        "Emphysema": 0.004,
        "Normal": 0.006,
        "Pneumonia-Bacterial": 0.0,
        "Pneumonia-Viral": 0.0,
        "Tuberculosis": 0.99,
    },
}
REJECTED_RAW = {"status": "rejected", "message": "Not a chest X-ray (OOD gate)"}


class FakeClassifier:
    """Mock with the same ``predict`` interface as the official ChestXrayClassifier."""

    def __init__(self, result=None, error: Exception | None = None):
        self.result, self.error, self.received = result, error, []

    def predict(self, image):
        self.received.append(image)
        if self.error:
            raise self.error
        return self.result


# --------------------------------------------------------------------------- constants


def test_class_order_matches_model_metadata():
    assert mu.CLASS_NAMES == (
        "Covid-19",
        "Emphysema",
        "Normal",
        "Pneumonia-Bacterial",
        "Pneumonia-Viral",
        "Tuberculosis",
    )
    assert mu.CONFIDENCE_THRESHOLD == 0.75


def test_only_needed_files_are_requested():
    assert set(mu.REQUIRED_FILES) == {
        "inference.py",
        "metadata.json",
        "lung_ood_detector.joblib",
        "DenseNet121_FT_best.keras",
    }


# --------------------------------------------------------------------------- load_image


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_load_image_returns_rgb(fmt):
    img = mu.load_image(make_image_bytes(fmt=fmt, mode="RGB"))
    assert img.mode == "RGB" and img.size == (64, 64)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "LA"])
def test_load_image_converts_other_png_modes_to_rgb(mode):
    img = mu.load_image(make_image_bytes(fmt="PNG", mode=mode))
    assert img.mode == "RGB"
    assert np.asarray(img).shape == (64, 64, 3)


def test_load_image_grayscale_jpeg_becomes_three_channels():
    img = mu.load_image(make_image_bytes(fmt="JPEG", mode="L"))
    assert np.asarray(img).shape == (64, 64, 3)


def test_load_image_rescales_16_bit_png_instead_of_saturating():
    arr = np.tile(np.linspace(0, 65535, 64, dtype=np.uint16), (64, 1))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")  # 16-bit grayscale PNG
    img = mu.load_image(buf.getvalue())
    out = np.asarray(img)[..., 0]
    assert out.min() == 0
    assert 250 <= out.max() <= 255
    assert 100 <= out[:, 32].mean() <= 140  # mid-gray stays mid-gray (plain convert would clip to 255)


def test_load_image_rejects_empty():
    with pytest.raises(mu.InvalidImageError, match="empty"):
        mu.load_image(b"")


def test_load_image_rejects_garbage():
    with pytest.raises(mu.InvalidImageError, match="could not be read"):
        mu.load_image(b"this is definitely not an image")


def test_load_image_rejects_unsupported_format():
    with pytest.raises(mu.InvalidImageError, match="Unsupported"):
        mu.load_image(make_image_bytes(fmt="GIF", mode="P"))


def test_load_image_rejects_truncated_file():
    data = make_image_bytes(fmt="PNG", size=(256, 256))
    with pytest.raises(mu.InvalidImageError, match="corrupt or truncated"):
        mu.load_image(data[: len(data) // 2])


def test_load_image_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(mu, "MAX_UPLOAD_BYTES", 100)
    with pytest.raises(mu.InvalidImageError, match="larger than"):
        mu.load_image(make_image_bytes(size=(128, 128)))


def test_load_image_rejects_tiny_image():
    with pytest.raises(mu.InvalidImageError, match="too small"):
        mu.load_image(make_image_bytes(size=(8, 8)))


def test_load_image_rejects_huge_pixel_count(monkeypatch):
    monkeypatch.setattr(mu, "MAX_PIXELS", 1000)
    with pytest.raises(mu.InvalidImageError, match="too large"):
        mu.load_image(make_image_bytes(size=(64, 64)))


# --------------------------------------------------------------------------- normalize_result


def test_normalize_review():
    r = mu.normalize_result(REVIEW_RAW)
    assert r.status == "review" and r.label == "Covid-19"
    assert r.confidence == pytest.approx(0.627)
    assert r.p_abnormal == pytest.approx(0.934)
    assert r.top_3[0] == ("Covid-19", pytest.approx(0.627))
    assert len(r.top_3) == 3 and len(r.probabilities) == 6
    assert not r.is_rejected


def test_normalize_accepted():
    r = mu.normalize_result(ACCEPTED_RAW)
    assert r.status == "accepted" and r.label == "Tuberculosis" and r.message is None


def test_normalize_rejected_has_no_class():
    r = mu.normalize_result(REJECTED_RAW)
    assert r.is_rejected
    assert r.label is None and r.confidence is None and r.p_abnormal is None
    assert r.top_3 == () and dict(r.probabilities) == {}
    assert r.message == "Not a chest X-ray (OOD gate)"


def test_rejected_never_leaks_a_class_even_if_present():
    r = mu.normalize_result({**REVIEW_RAW, "status": "rejected"})
    assert r.label is None and r.confidence is None and r.top_3 == ()


def test_accepted_below_threshold_is_downgraded_to_review():
    r = mu.normalize_result({**REVIEW_RAW, "status": "accepted"})
    assert r.status == "review"
    assert "below" in r.message


def test_top_3_is_derived_from_probabilities_when_missing():
    raw = {k: v for k, v in REVIEW_RAW.items() if k != "top_3"}
    r = mu.normalize_result(raw)
    assert [name for name, _ in r.top_3] == ["Covid-19", "Emphysema", "Pneumonia-Bacterial"]


def test_optional_fields_may_be_absent():
    r = mu.normalize_result({"status": "review", "label": "Normal", "confidence": 0.5})
    assert r.p_abnormal is None and r.top_3 == () and dict(r.probabilities) == {}


def test_numpy_floats_are_accepted():
    raw = {**REVIEW_RAW, "confidence": np.float32(0.6), "p_abnormal": np.float64(0.9)}
    assert mu.normalize_result(raw).confidence == pytest.approx(0.6)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "review",
        [],
        {},
        {"status": "maybe", "label": "Normal", "confidence": 0.9},
        {"status": "review", "confidence": 0.5},
        {"status": "review", "label": "Flu", "confidence": 0.5},
        {"status": "review", "label": "Normal"},
        {"status": "review", "label": "Normal", "confidence": 1.5},
        {"status": "review", "label": "Normal", "confidence": -0.1},
        {"status": "review", "label": "Normal", "confidence": float("nan")},
        {"status": "review", "label": "Normal", "confidence": "0.5"},
        {"status": "review", "label": "Normal", "confidence": True},
        {"status": "review", "label": "Normal", "confidence": 0.5, "p_abnormal": 2},
        {"status": "review", "label": "Normal", "confidence": 0.5, "probabilities": [1, 2]},
        {"status": "review", "label": "Normal", "confidence": 0.5, "probabilities": {"Flu": 0.5}},
        {"status": "review", "label": "Normal", "confidence": 0.5, "top_3": "abc"},
        {"status": "review", "label": "Normal", "confidence": 0.5, "top_3": [("Normal",)]},
        {"status": "review", "label": "Normal", "confidence": 0.5, "top_3": [("Flu", 0.5)]},
    ],
)
def test_malformed_results_raise(raw):
    with pytest.raises(mu.MalformedResultError):
        mu.normalize_result(raw)


# --------------------------------------------------------------------------- formatting helpers


@pytest.mark.parametrize(
    "status,severity",
    [("accepted", "success"), ("review", "warning"), ("rejected", "error")],
)
def test_status_info_severity(status, severity):
    info = mu.status_info(status)
    assert info.severity == severity and info.explanation


def test_status_info_unknown_raises():
    with pytest.raises(ValueError):
        mu.status_info("nope")


def test_review_explanation_does_not_claim_confirmation():
    text = mu.status_info("review").explanation.lower()
    assert "not confirmed" in text


def test_format_percent():
    assert mu.format_percent(0.627) == "62.7%"
    assert mu.format_percent(1.0, digits=0) == "100%"
    assert mu.format_percent(None) == "n/a"


def test_probability_rows_sorted_descending():
    rows = mu.probability_rows(mu.normalize_result(REVIEW_RAW))
    scores = [p for _, p in rows]
    assert scores == sorted(scores, reverse=True) and rows[0][0] == "Covid-19"
    assert mu.probability_rows(mu.normalize_result(REJECTED_RAW)) == []


# --------------------------------------------------------------------------- classify


def test_classify_passes_a_pil_image_and_returns_normalized_result():
    clf = FakeClassifier(REVIEW_RAW)
    img = mu.load_image(make_image_bytes())
    result = mu.classify(clf, img)
    assert isinstance(clf.received[0], Image.Image) and clf.received[0].mode == "RGB"
    assert result.label == "Covid-19"


def test_classify_wraps_inference_errors():
    clf = FakeClassifier(error=RuntimeError("boom"))
    with pytest.raises(mu.InferenceError, match="boom"):
        mu.classify(clf, mu.load_image(make_image_bytes()))


def test_classify_flags_malformed_output():
    clf = FakeClassifier({"status": "accepted"})
    with pytest.raises(mu.MalformedResultError):
        mu.classify(clf, mu.load_image(make_image_bytes()))


# --------------------------------------------------------------------------- load_classifier


FAKE_INFERENCE = '''
class ChestXrayClassifier:
    def __init__(self, source):
        self.source = source
'''


def test_load_classifier_downloads_only_required_files_and_uses_local_folder(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    (tmp_path / "inference.py").write_text(FAKE_INFERENCE)
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.delenv("MODEL_REVISION", raising=False)

    clf = mu.load_classifier()
    assert clf.source == str(tmp_path)
    assert calls["repo_id"] == mu.REPO_ID
    assert calls["revision"] == "main"
    assert set(calls["allow_patterns"]) == set(mu.REQUIRED_FILES)


def test_load_classifier_honours_model_revision(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    (tmp_path / "inference.py").write_text(FAKE_INFERENCE)
    seen = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kw: seen.update(kw) or str(tmp_path)
    )
    monkeypatch.setenv("MODEL_REVISION", "abc123")
    mu.load_classifier()
    assert seen["revision"] == "abc123"


def test_load_classifier_wraps_download_errors(monkeypatch):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    def boom(**kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    with pytest.raises(mu.ModelLoadError, match="download"):
        mu.load_classifier()


def test_load_classifier_wraps_initialisation_errors(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    (tmp_path / "inference.py").write_text(
        "class ChestXrayClassifier:\n    def __init__(self, source):\n        raise ImportError('keras missing')\n"
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: str(tmp_path))
    with pytest.raises(mu.ModelLoadError, match="keras missing"):
        mu.load_classifier()


# --------------------------------------------------------------------------- UI (AppTest)

from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def run_app(monkeypatch):
    """Return a function that launches app.py with a mock classifier and optional upload."""
    import streamlit as st

    def _run(classifier=None, upload: tuple | None = None, load_error: Exception | None = None):
        st.cache_resource.clear()

        def fake_loader(*args, **kwargs):
            if load_error:
                raise load_error
            return classifier

        monkeypatch.setattr(mu, "load_classifier", fake_loader)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at.run()
        assert not at.exception
        if upload is not None:
            at.file_uploader[0].set_value(upload)
            at.run()
        return at

    return _run


def all_text(at) -> str:
    parts = []
    for group in (at.markdown, at.success, at.warning, at.error, at.info, at.caption, at.metric, at.table):
        for el in group:
            parts.append(str(getattr(el, "value", "")))
    return "\n".join(parts)


PNG = ("xray.png", make_image_bytes(), "image/png")


def test_ui_initial_state_shows_title_disclaimer_and_prompt(run_app):
    at = run_app()
    assert not at.exception
    text = all_text(at)
    assert "Chest X-Ray Classification" in text
    assert "not a medical device" in text
    assert "Upload a JPG, JPEG or PNG" in text


def test_ui_classify_without_upload_shows_warning(run_app):
    at = run_app(FakeClassifier(REVIEW_RAW))
    at.button[0].click().run()
    assert any("upload a chest X-ray" in w.value for w in at.warning)


def test_ui_invalid_file_shows_error_and_does_not_classify(run_app):
    clf = FakeClassifier(REVIEW_RAW)
    at = run_app(clf, upload=("bad.png", b"not an image", "image/png"))
    assert any("Invalid image" in e.value for e in at.error)
    at.button[0].click().run()
    assert clf.received == []


def test_ui_accepted_result(run_app):
    clf = FakeClassifier(ACCEPTED_RAW)
    at = run_app(clf, upload=PNG)
    at.button[0].click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Predicted class"] == "Tuberculosis"
    assert metrics["Model confidence"] == "99.0%"
    assert any("Accepted" in s.value for s in at.success)
    assert "Top 3 predictions" in all_text(at) and len(at.table) == 1


def test_ui_review_result_is_not_presented_as_confirmed(run_app):
    at = run_app(FakeClassifier(REVIEW_RAW), upload=PNG)
    at.button[0].click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert "Predicted class" not in metrics
    assert metrics["Best guess (unconfirmed)"] == "Covid-19"
    assert any("Needs review" in w.value for w in at.warning)


def test_ui_rejected_result_shows_no_class(run_app):
    at = run_app(FakeClassifier(REJECTED_RAW), upload=PNG)
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) == 0
    assert any("Rejected" in e.value for e in at.error)
    assert "could not classify" in all_text(at)


def test_ui_model_load_failure_is_reported_and_retryable(run_app, monkeypatch):
    at = run_app(upload=PNG, load_error=mu.ModelLoadError("download failed"))
    at.button[0].click().run()
    assert not at.exception
    assert any("Model unavailable" in e.value and "download failed" in e.value for e in at.error)


def test_ui_inference_failure_is_reported(run_app):
    at = run_app(FakeClassifier(error=RuntimeError("kaboom")), upload=PNG)
    at.button[0].click().run()
    assert not at.exception
    assert any("Classification failed" in e.value and "kaboom" in e.value for e in at.error)


def test_ui_malformed_model_output_is_reported(run_app):
    at = run_app(FakeClassifier({"status": "review", "label": "Flu", "confidence": 0.4}), upload=PNG)
    at.button[0].click().run()
    assert not at.exception
    assert any("Unexpected model output" in e.value for e in at.error)


def test_ui_model_is_loaded_once_across_predictions(run_app, monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    loads = []

    def counting_loader(*args, **kwargs):
        loads.append(1)
        return FakeClassifier(REVIEW_RAW)

    monkeypatch.setattr(mu, "load_classifier", counting_loader)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.file_uploader[0].set_value(PNG)
    at.run()
    at.button[0].click().run()
    at.button[0].click().run()
    assert len(loads) == 1


def test_ui_result_is_cleared_when_a_new_file_is_uploaded(run_app):
    at = run_app(FakeClassifier(REVIEW_RAW), upload=PNG)
    at.button[0].click().run()
    assert len(at.metric) == 2
    at.file_uploader[0].set_value(("other.png", make_image_bytes(size=(80, 80)), "image/png"))
    at.run()
    assert len(at.metric) == 0
