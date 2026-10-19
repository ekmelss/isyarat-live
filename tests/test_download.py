"""Unit tests for tools.download_data."""

from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.download_data import (
    DATASET_CONFIG,
    WORDS,
    TRAIN_INDICES,
    TEST_INDICES,
    download_file,
    extract_archive,
    find_dataset_root,
    main,
    resolve_download_info,
    run_download_pipeline,
    verify_archive,
    verify_extracted_structure,
)


def create_synthetic_dataset_zip(
    zip_path: Path,
    folder_prefix: str = "MyWSL2023 RAW DATA",
    num_train: int = 300,
    num_test: int = 50,
    words: tuple[str, ...] = WORDS,
) -> Path:
    """Helper to create a synthetic dataset zip file for testing."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for word in words:
            for idx in range(1, num_train + 1):
                entry_name = f"{folder_prefix}/train/{word}/{word} ({idx}).jpg"
                zf.writestr(entry_name, b"fake_jpeg_bytes")
            for idx in range(301, 301 + num_test):
                entry_name = f"{folder_prefix}/test/{word}/{word} ({idx}).jpg"
                zf.writestr(entry_name, b"fake_jpeg_bytes")
    return zip_path


def create_synthetic_extracted_dataset(
    root_dir: Path,
    num_train: int = 300,
    num_test: int = 50,
    words: tuple[str, ...] = WORDS,
    empty_file: bool = False,
) -> Path:
    """Helper to create a synthetic extracted dataset on disk for testing."""
    for word in words:
        train_dir = root_dir / "train" / word
        test_dir = root_dir / "test" / word
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        content = b"" if empty_file else b"fake_jpeg_bytes"

        for idx in range(1, num_train + 1):
            (train_dir / f"{word} ({idx}).jpg").write_bytes(content)
        for idx in range(301, 301 + num_test):
            (test_dir / f"{word} ({idx}).jpg").write_bytes(content)

    return root_dir


# ==============================================================================
# URL Resolution Tests
# ==============================================================================


def test_resolve_download_info_invalid_dataset():
    with pytest.raises(ValueError, match="Unknown dataset type"):
        resolve_download_info("invalid_type")


def test_resolve_download_info_api_success():
    mock_folders = [{"id": "folder-123", "name": "MyWSL2023"}]
    mock_files = [
        {
            "id": "7f11b8a0-24e4-45df-af3d-e861f41435ea",
            "name": "MyWSL2023 RAW DATA.zip",
            "size": 123456,
            "content_details": {"download_url": "https://custom.download/url"},
        }
    ]

    def mock_urlopen(req, timeout=10.0):
        url = req.full_url if hasattr(req, "full_url") else req
        resp = MagicMock()
        if "folders" in url:
            resp.read.return_value = json.dumps(mock_folders).encode("utf-8")
        elif "files" in url:
            resp.read.return_value = json.dumps(mock_files).encode("utf-8")
        resp.__enter__.return_value = resp
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        url, name, size = resolve_download_info("raw")
        assert url == "https://custom.download/url"
        assert name == "MyWSL2023 RAW DATA.zip"
        assert size == 123456


def test_resolve_download_info_fallback_on_network_error():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network unreachable")):
        url, name, size = resolve_download_info("crop")
        assert url == DATASET_CONFIG["crop"]["direct_url"]
        assert name == "MyWSL2023 CROP DATA.zip"
        assert size == DATASET_CONFIG["crop"]["expected_size"]


# ==============================================================================
# Download Tests
# ==============================================================================


def test_download_file_new(tmp_path: Path):
    dest = tmp_path / "data.zip"
    content = b"sample_binary_content" * 100

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.headers = {"Content-Length": str(len(content))}
    mock_resp.read.side_effect = [content[:50], content[50:], b""]

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res_path = download_file("http://example.com/test.zip", dest, expected_size=len(content))
        assert res_path == dest
        assert dest.exists()
        assert dest.read_bytes() == content


def test_download_file_resume(tmp_path: Path):
    dest = tmp_path / "data.zip"
    part_path = dest.with_suffix(dest.suffix + ".part")
    initial_bytes = b"initial_content_"
    part_path.write_bytes(initial_bytes)

    remaining_bytes = b"remaining_content"

    mock_resp = MagicMock()
    mock_resp.status = 206  # Partial Content
    mock_resp.headers = {"Content-Length": str(len(remaining_bytes))}
    mock_resp.read.side_effect = [remaining_bytes, b""]

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        download_file("http://example.com/test.zip", dest, resume=True)
        # Check that Range header was passed
        req_sent = mock_urlopen.call_args[0][0]
        assert req_sent.headers.get("Range") == f"bytes={len(initial_bytes)}-"
        assert dest.read_bytes() == initial_bytes + remaining_bytes


def test_download_file_range_not_satisfiable(tmp_path: Path):
    dest = tmp_path / "data.zip"
    part_path = dest.with_suffix(dest.suffix + ".part")
    part_path.write_bytes(b"corrupted_part")

    full_content = b"fresh_full_download"

    err_416 = urllib.error.HTTPError(
        "http://example.com/test.zip", 416, "Range Not Satisfiable", {}, None
    )

    mock_resp_fresh = MagicMock()
    mock_resp_fresh.status = 200
    mock_resp_fresh.headers = {"Content-Length": str(len(full_content))}
    mock_resp_fresh.read.side_effect = [full_content, b""]

    with patch("urllib.request.urlopen", side_effect=[err_416, mock_resp_fresh]):
        download_file("http://example.com/test.zip", dest, resume=True)
        assert dest.read_bytes() == full_content


# ==============================================================================
# Archive Verification Tests
# ==============================================================================


def test_verify_archive_not_found(tmp_path: Path):
    ok, msg = verify_archive(tmp_path / "nonexistent.zip")
    assert not ok
    assert "does not exist" in msg


def test_verify_archive_empty_file(tmp_path: Path):
    p = tmp_path / "empty.zip"
    p.touch()
    ok, msg = verify_archive(p)
    assert not ok
    assert "empty" in msg


def test_verify_archive_corrupted_zip(tmp_path: Path):
    p = tmp_path / "corrupt.zip"
    p.write_bytes(b"PK\x03\x04not_a_valid_zip_payload")
    ok, msg = verify_archive(p)
    assert not ok
    assert "Invalid zip archive" in msg or "corrupted" in msg


def test_verify_archive_valid(tmp_path: Path):
    zip_path = tmp_path / "valid.zip"
    create_synthetic_dataset_zip(zip_path)
    ok, msg = verify_archive(zip_path)
    assert ok
    assert "verified successfully" in msg


def test_verify_archive_missing_entries(tmp_path: Path):
    zip_path = tmp_path / "incomplete.zip"
    # Only 10 train images instead of 300
    create_synthetic_dataset_zip(zip_path, num_train=10, num_test=50)
    ok, msg = verify_archive(zip_path)
    assert not ok
    assert "missing expected image entries" in msg


# ==============================================================================
# Extraction & Directory Structure Tests
# ==============================================================================


def test_extract_archive_success(tmp_path: Path):
    zip_path = tmp_path / "dataset.zip"
    create_synthetic_dataset_zip(zip_path, folder_prefix="MyWSL2023 RAW DATA")

    extract_dest = tmp_path / "extracted"
    extract_archive(zip_path, extract_dest)

    root = find_dataset_root(extract_dest, "MyWSL2023 RAW DATA")
    assert root is not None
    assert (root / "train" / "air" / "air (1).jpg").exists()


def test_extract_archive_prevents_unsafe_path(tmp_path: Path):
    zip_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../evil.txt", b"evil")

    extract_dest = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe zip entry"):
        extract_archive(zip_path, extract_dest)


def test_verify_extracted_structure_valid(tmp_path: Path):
    dataset_root = tmp_path / "MyWSL2023 RAW DATA"
    create_synthetic_extracted_dataset(dataset_root, num_train=300, num_test=50)

    ok, counts, errors = verify_extracted_structure(dataset_root)
    assert ok
    assert len(errors) == 0
    assert sum(counts.values()) == 3500
    for w in WORDS:
        assert counts[w] == 350


def test_verify_extracted_structure_missing_train_or_test(tmp_path: Path):
    dataset_root = tmp_path / "incomplete"
    dataset_root.mkdir()
    ok, counts, errors = verify_extracted_structure(dataset_root)
    assert not ok
    assert any("Missing 'train'" in e for e in errors)
    assert any("Missing 'test'" in e for e in errors)


def test_verify_extracted_structure_empty_images(tmp_path: Path):
    dataset_root = tmp_path / "empty_dataset"
    create_synthetic_extracted_dataset(dataset_root, num_train=300, num_test=50, empty_file=True)

    ok, counts, errors = verify_extracted_structure(dataset_root)
    assert not ok
    assert any("Empty image file" in e for e in errors)


# ==============================================================================
# Pipeline & CLI Tests
# ==============================================================================


def test_run_download_pipeline_check_only_fail(tmp_path: Path):
    # Nothing in tmp_path
    ret = run_download_pipeline(dataset_type="raw", dest_dir=tmp_path, check_only=True)
    assert ret == 1


def test_run_download_pipeline_check_only_success(tmp_path: Path):
    # Pre-populate valid extracted dataset
    extracted = tmp_path / "MyWSL2023 RAW DATA"
    create_synthetic_extracted_dataset(extracted, num_train=300, num_test=50)

    ret = run_download_pipeline(dataset_type="raw", dest_dir=tmp_path, check_only=True)
    assert ret == 0


def test_run_download_pipeline_end_to_end_mocked(tmp_path: Path):
    zip_path = tmp_path / "MyWSL2023 CROP DATA.zip"
    create_synthetic_dataset_zip(zip_path, folder_prefix="MyWSL2023 CROP DATA")

    # Mock download_file to do nothing since zip already exists
    with patch("tools.download_data.download_file") as mock_dl:
        ret = run_download_pipeline(
            dataset_type="crop",
            dest_dir=tmp_path,
            check_only=False,
            no_extract=False,
        )
        assert ret == 0
        mock_dl.assert_not_called()

        # Check that files were extracted
        extracted = tmp_path / "MyWSL2023 CROP DATA"
        assert extracted.exists()
        assert (extracted / "train" / "air" / "air (1).jpg").exists()


def test_cli_main_check(tmp_path: Path):
    extracted = tmp_path / "MyWSL2023 RAW DATA"
    create_synthetic_extracted_dataset(extracted, num_train=300, num_test=50)

    ret = main(["--check", "--dest", str(tmp_path), "--dataset", "raw"])
    assert ret == 0
