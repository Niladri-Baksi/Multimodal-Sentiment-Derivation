"""Regression tests for Phase 2 annotation validation and media inspection."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.data import preprocessing
from sentiment_derivation.data.preprocessing import inspect_media_file, load_annotations


class _FakeVideoCapture:
    """Small controllable decoder fake for synthetic inspection tests."""

    def __init__(
        self,
        *,
        opened: bool = True,
        properties: dict[int, float] | None = None,
        read_result: tuple[bool, object] | None = None,
        read_error: Exception | None = None,
        open_error: Exception | None = None,
    ) -> None:
        self.opened = opened
        self.properties = properties or {1: 640, 2: 480, 3: 25.0, 4: 10}
        self.read_result = read_result
        self.read_error = read_error
        self.open_error = open_error
        self.released = False

    def isOpened(self) -> bool:
        if self.open_error:
            raise self.open_error
        return self.opened

    def get(self, property_id: int) -> float:
        return self.properties[property_id]

    def read(self) -> tuple[bool, object]:
        if self.read_error:
            raise self.read_error
        if self.read_result is not None:
            return self.read_result
        import numpy as np

        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _install_fake_cv2(monkeypatch: pytest.MonkeyPatch, capture: _FakeVideoCapture) -> None:
    monkeypatch.setattr(
        preprocessing,
        "cv2",
        SimpleNamespace(
            VideoCapture=lambda _path: capture,
            CAP_PROP_FRAME_WIDTH=1,
            CAP_PROP_FRAME_HEIGHT=2,
            CAP_PROP_FPS=3,
            CAP_PROP_FRAME_COUNT=4,
        ),
    )


@pytest.mark.parametrize("valence", ["nan", "inf", "-inf", "0", "8"])
def test_load_annotations_rejects_invalid_valence(tmp_path: Path, valence: str) -> None:
    path = tmp_path / "labels.text"
    path.write_text(f"0,0,0,0,0,0,0,0,{valence}\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_annotations(path)

    message = str(exc_info.value)
    assert "valence" in message.lower()
    assert "[1, 7]" in message
    assert "line 1" in message.lower()


def test_inspect_media_rejects_corrupt_video(tmp_path: Path) -> None:
    if preprocessing.cv2 is None:
        pytest.skip("OpenCV is unavailable; video decoding cannot be tested.")

    path = tmp_path / "corrupt.mp4"
    path.write_bytes(b"not a decodable video")
    result = inspect_media_file(path)
    assert result["valid"] is False
    assert result["error"]
    assert "unavailable" not in result["error"].lower()


def test_inspect_media_rejects_decoder_that_does_not_open_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unopenable.mp4"
    path.write_bytes(b"synthetic nonempty media")
    capture = _FakeVideoCapture(opened=False)
    _install_fake_cv2(monkeypatch, capture)

    result = inspect_media_file(path)

    assert result["valid"] is False
    assert "open" in result["error"].lower()
    assert capture.released


def test_inspect_media_rejects_positive_metadata_when_no_frame_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unreadable_frames.mp4"
    path.write_bytes(b"synthetic nonempty media")
    capture = _FakeVideoCapture(read_result=(False, None))
    _install_fake_cv2(monkeypatch, capture)

    result = inspect_media_file(path)

    assert result["valid"] is False
    assert "frame" in result["error"].lower()
    assert capture.released


@pytest.mark.parametrize(
    "property_id,value",
    [(1, 0), (2, -1), (3, 0.0), (3, float("nan")), (4, 0)],
)
def test_inspect_media_rejects_invalid_decoder_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    property_id: int,
    value: float,
) -> None:
    path = tmp_path / "invalid_metadata.mp4"
    path.write_bytes(b"synthetic nonempty media")
    properties = {1: 640, 2: 480, 3: 25.0, 4: 10}
    properties[property_id] = value
    capture = _FakeVideoCapture(properties=properties)
    _install_fake_cv2(monkeypatch, capture)

    result = inspect_media_file(path)

    assert result["valid"] is False
    assert result["error"]
    assert capture.released


def test_inspect_media_contains_decoder_initialization_and_read_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "decoder_error.mp4"
    path.write_bytes(b"synthetic nonempty media")
    capture = _FakeVideoCapture(read_error=RuntimeError("decoder read failed"))
    _install_fake_cv2(monkeypatch, capture)

    result = inspect_media_file(path)

    assert result["valid"] is False
    assert "decoder read failed" in result["error"]
    assert capture.released


def test_manifest_marks_unreadable_video_unavailable_and_reports_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_path = video_dir / "clip.mp4"
    video_path.write_bytes(b"synthetic nonempty media")
    labels_path = tmp_path / "labels.text"
    labels_path.write_text("0,0,0,0,0,0,0,0,4.0\n", encoding="utf-8")
    capture = _FakeVideoCapture(read_result=(False, None))
    _install_fake_cv2(monkeypatch, capture)

    manifest = preprocessing.build_processed_manifest(
        dataset_root=tmp_path,
        annotation_files={"train": labels_path},
        video_dirs={"train": video_dir},
    )

    sample = manifest["samples"]["train"][0]
    assert sample["modality_availability"]["video"] is False
    assert any(
        issue["sample_id"] == "clip" and "frame" in issue["issue"].lower()
        for issue in manifest["corrupted_or_missing"]
    )
    assert capture.released
