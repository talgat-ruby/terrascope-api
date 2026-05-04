"""Unit tests for YoloSahiProcess weight-path resolution.

Covers the local-path / Ultralytics-alias / HF-Hub branches of
`_resolve_weights` without ever loading SAHI or hitting the HF API.
"""

from pathlib import Path
from unittest.mock import patch

from core import ProcessSpec
from core.processes.yolo_sahi import (
    YoloSahiProcess,
    _looks_like_hf_repo,
)


def test_local_path_passthrough(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    proc = YoloSahiProcess.from_spec(
        ProcessSpec(name="custom", kwargs={"weights": str(weights)})
    )
    assert proc._resolve_weights() == str(weights)


def test_ultralytics_alias_passthrough(tmp_path: Path):
    proc = YoloSahiProcess.from_spec(
        ProcessSpec(
            name="yolov8n-sahi",
            kwargs={"weights": "yolov8n.pt", "weights_dir": str(tmp_path)},
        )
    )
    # No slash → not an HF repo, no download attempt.
    assert proc._resolve_weights() == "yolov8n.pt"


def test_hf_repo_downloads_into_weights_dir(tmp_path: Path):
    repo = "keremberke/yolov8m-satellite-vehicle-detection"
    proc = YoloSahiProcess.from_spec(
        ProcessSpec(
            name="yolov8-satellite-vehicle",
            kwargs={"weights": repo, "weights_dir": str(tmp_path)},
        )
    )

    fake_local = tmp_path / "fake_best.pt"
    fake_local.write_bytes(b"")

    class _FakeApi:
        def list_repo_files(self, _repo_id: str) -> list[str]:
            return ["README.md", "best.pt", "config.json"]

    with patch(
        "huggingface_hub.HfApi", return_value=_FakeApi()
    ), patch(
        "huggingface_hub.hf_hub_download", return_value=str(fake_local)
    ) as dl:
        out = proc._resolve_weights()

    assert out == str(fake_local)
    dl.assert_called_once()
    kwargs = dl.call_args.kwargs
    assert kwargs["repo_id"] == repo
    assert kwargs["filename"] == "best.pt"
    assert kwargs["local_dir"].endswith(repo.replace("/", "__"))


def test_hf_repo_explicit_filename(tmp_path: Path):
    proc = YoloSahiProcess.from_spec(
        ProcessSpec(
            name="custom",
            kwargs={
                "weights": "user/repo",
                "weights_dir": str(tmp_path),
                "weights_filename": "checkpoint.pt",
            },
        )
    )
    fake_local = tmp_path / "checkpoint.pt"
    fake_local.write_bytes(b"")

    with patch(
        "huggingface_hub.hf_hub_download", return_value=str(fake_local)
    ) as dl:
        out = proc._resolve_weights()

    assert out == str(fake_local)
    assert dl.call_args.kwargs["filename"] == "checkpoint.pt"


def test_hf_repo_with_no_pt_files_raises(tmp_path: Path):
    proc = YoloSahiProcess.from_spec(
        ProcessSpec(
            name="custom",
            kwargs={"weights": "user/no-pt-repo", "weights_dir": str(tmp_path)},
        )
    )

    class _FakeApi:
        def list_repo_files(self, _repo_id: str) -> list[str]:
            return ["README.md", "config.json"]

    with patch("huggingface_hub.HfApi", return_value=_FakeApi()):
        try:
            proc._resolve_weights()
        except FileNotFoundError as e:
            assert "No .pt files" in str(e)
        else:
            raise AssertionError("expected FileNotFoundError")


def test_default_weights_dir_is_local_tmp():
    proc = YoloSahiProcess.from_spec(ProcessSpec(name="yolov8n-sahi"))
    assert str(proc.weights_dir) == "tmp/weights"


def test_looks_like_hf_repo_truthtable():
    assert _looks_like_hf_repo("user/repo") is True
    assert _looks_like_hf_repo("yolov8n.pt") is False
    assert _looks_like_hf_repo("./local/best.pt") is False
    assert _looks_like_hf_repo("/abs/path/best.pt") is False
    assert _looks_like_hf_repo("user/repo/extra") is False
