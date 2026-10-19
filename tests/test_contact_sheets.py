"""Unit tests for tools.generate_contact_sheets."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from tools.generate_contact_sheets import (
    WORDS,
    ImageItem,
    batch_generate_thumbnails,
    discover_word_images,
    extract_index_from_filename,
    find_dataset_root,
    generate_thumbnail,
    load_annotations,
    main,
    render_index_sheet,
    render_signer_sheet,
    render_word_sheet,
    run_pipeline,
)


def create_mock_image(dest_path: Path, size: tuple[int, int] = (64, 48), color: tuple[int, int, int] = (200, 50, 50)) -> Path:
    """Create a small synthetic JPEG image for testing."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(dest_path, "JPEG")
    return dest_path


def create_mock_dataset(
    root_dir: Path,
    words: tuple[str, ...] = ("air", "demam"),
    train_count: int = 10,
    test_count: int = 5,
) -> Path:
    """Create a minimal synthetic MyWSL2023 dataset directory structure."""
    for word in words:
        train_dir = root_dir / "train" / word
        test_dir = root_dir / "test" / word

        for idx in range(1, train_count + 1):
            create_mock_image(train_dir / f"{word} ({idx}).jpg", color=(idx * 20 % 255, 50, 100))

        for idx in range(1, test_count + 1):
            seq = 300 + idx
            create_mock_image(test_dir / f"{word} ({seq}).jpg", color=(100, idx * 30 % 255, 200))

    return root_dir


class TestDiscoveryAndParsing:
    def test_find_dataset_root_direct(self, tmp_path: Path) -> None:
        dataset = create_mock_dataset(tmp_path / "dataset")
        found = find_dataset_root(dataset)
        assert found == dataset

    def test_find_dataset_root_raw_folder(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "MyWSL2023 RAW DATA"
        create_mock_dataset(raw_dir)
        found = find_dataset_root(tmp_path)
        assert found == raw_dir

    def test_find_dataset_root_crop_folder(self, tmp_path: Path) -> None:
        crop_dir = tmp_path / "MyWSL2023 CROP DATA"
        create_mock_dataset(crop_dir)
        found = find_dataset_root(tmp_path)
        assert found == crop_dir

    def test_find_dataset_root_nonexistent(self, tmp_path: Path) -> None:
        assert find_dataset_root(tmp_path / "does_not_exist") is None

    def test_extract_index_from_filename(self) -> None:
        assert extract_index_from_filename("air (1).jpg") == 1
        assert extract_index_from_filename("demam (350).jpg") == 350
        assert extract_index_from_filename("air_0042.jpg") == 42
        assert extract_index_from_filename("makan-7.jpeg") == 7
        assert extract_index_from_filename("no_digits.png") is None

    def test_discover_word_images(self, tmp_path: Path) -> None:
        dataset = create_mock_dataset(tmp_path, words=("air",), train_count=5, test_count=3)
        items = discover_word_images(dataset, "air")

        assert len(items) == 8
        # Sequence numbers: 1, 2, 3, 4, 5, 301, 302, 303
        assert [it.index for it in items] == [1, 2, 3, 4, 5, 301, 302, 303]
        assert [it.split for it in items] == ["train"] * 5 + ["test"] * 3
        assert all(it.source_path.exists() for it in items)


class TestThumbnailGeneration:
    def test_generate_thumbnail(self, tmp_path: Path) -> None:
        src = create_mock_image(tmp_path / "src.jpg", size=(192, 108))
        dest = tmp_path / "thumb.jpg"

        thumb = generate_thumbnail(src, dest, max_size=(64, 64), quality=70)
        assert thumb.exists()
        assert thumb.stat().st_size > 0

        with Image.open(thumb) as im:
            # 16:9 aspect ratio downscaled to max 64x64 -> 64x36
            assert im.size[0] <= 64
            assert im.size[1] <= 64

    def test_thumbnail_caching(self, tmp_path: Path) -> None:
        src = create_mock_image(tmp_path / "src.jpg")
        dest = tmp_path / "thumb.jpg"

        generate_thumbnail(src, dest)
        mtime1 = dest.stat().st_mtime_ns

        # Re-generating without force should keep cached file
        generate_thumbnail(src, dest, force=False)
        mtime2 = dest.stat().st_mtime_ns
        assert mtime1 == mtime2

    def test_batch_generate_thumbnails(self, tmp_path: Path) -> None:
        dataset = create_mock_dataset(tmp_path / "data", words=("air",), train_count=4, test_count=2)
        items = discover_word_images(dataset, "air")
        thumb_base = tmp_path / "thumbs"

        batch_generate_thumbnails(items, thumb_base_dir=thumb_base, max_size=(60, 60), workers=2)

        for it in items:
            assert it.thumb_path is not None
            assert it.thumb_path.exists()
            assert it.thumb_path.stat().st_size > 0


class TestAnnotationsLoading:
    def test_load_referee_format(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "annotations.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "signer", "label"])
            writer.writerow(["air_0001", "S1", "air"])
            writer.writerow(["air_0002", "S1", "air"])
            writer.writerow(["air_0071", "S2", "air"])

        mapping = load_annotations(csv_file)
        assert mapping[("air", 1)] == "S1"
        assert mapping[("air", 2)] == "S1"
        assert mapping[("air", 71)] == "S2"

    def test_load_column_format(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "annotations.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "index", "signer"])
            writer.writerow(["demam", "5", "S3"])
            writer.writerow(["demam", "6", "S3"])

        mapping = load_annotations(csv_file)
        assert mapping[("demam", 5)] == "S3"
        assert mapping[("demam", 6)] == "S3"

    def test_load_range_format(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "annotations.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "start_idx", "end_idx", "signer"])
            writer.writerow(["air", "1", "3", "S1"])
            writer.writerow(["air", "4", "5", "S2"])

        mapping = load_annotations(csv_file)
        assert mapping[("air", 1)] == "S1"
        assert mapping[("air", 2)] == "S1"
        assert mapping[("air", 3)] == "S1"
        assert mapping[("air", 4)] == "S2"
        assert mapping[("air", 5)] == "S2"

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_annotations(tmp_path / "nonexistent.csv")

    def test_load_missing_signer_column_raises(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("word,index\nair,1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="signer"):
            load_annotations(csv_file)


class TestHtmlRendering:
    def test_render_word_sheet(self, tmp_path: Path) -> None:
        dataset = create_mock_dataset(tmp_path / "data", words=("air",), train_count=3, test_count=2)
        items = discover_word_images(dataset, "air")
        items[0].signer = "S1"
        items[1].signer = "S1"
        items[2].signer = "S2"

        thumb_dir = tmp_path / "sheets" / "thumbnails"
        batch_generate_thumbnails(items, thumb_base_dir=thumb_dir, max_size=(50, 50))

        out_html = tmp_path / "sheets" / "air.html"
        rendered = render_word_sheet(
            word="air",
            images=items,
            output_html_path=out_html,
            all_words=["air", "demam"],
            all_signers=["S1", "S2"],
        )

        assert rendered.exists()
        content = rendered.read_text(encoding="utf-8")

        # Check sequence numbers and structure
        assert "#1" in content
        assert "#2" in content
        assert "#3" in content
        assert "#301" in content
        assert "#302" in content
        assert "Word: air" in content
        assert "(water)" in content
        assert "badge-signer s1" in content
        assert "badge-signer s2" in content
        assert "modal" in content
        assert "setGridSize" in content

    def test_render_signer_sheet(self, tmp_path: Path) -> None:
        dataset = create_mock_dataset(tmp_path / "data", words=("air", "demam"), train_count=2, test_count=1)
        items = discover_word_images(dataset, "air") + discover_word_images(dataset, "demam")
        for it in items:
            it.signer = "S1"

        thumb_dir = tmp_path / "sheets" / "thumbnails"
        batch_generate_thumbnails(items, thumb_base_dir=thumb_dir, max_size=(50, 50))

        out_html = tmp_path / "sheets" / "signer_S1.html"
        rendered = render_signer_sheet(
            signer_id="S1",
            images=items,
            output_html_path=out_html,
            all_words=["air", "demam"],
            all_signers=["S1"],
        )

        assert rendered.exists()
        content = rendered.read_text(encoding="utf-8")
        assert "Signer Verification: S1" in content or "signer_S1" in content
        assert "air" in content
        assert "demam" in content
        assert "Total Assigned Frames" in content

    def test_render_index_sheet(self, tmp_path: Path) -> None:
        dataset = create_mock_dataset(tmp_path / "data", words=("air",), train_count=2, test_count=1)
        items = discover_word_images(dataset, "air")
        items[0].signer = "S1"

        thumb_dir = tmp_path / "sheets" / "thumbnails"
        batch_generate_thumbnails(items, thumb_base_dir=thumb_dir, max_size=(50, 50))

        out_html = tmp_path / "sheets" / "index.html"
        rendered = render_index_sheet(
            words=["air"],
            word_images={"air": items},
            signers=["S1"],
            signer_images={"S1": items},
            output_html_path=out_html,
        )

        assert rendered.exists()
        content = rendered.read_text(encoding="utf-8")
        assert "air.html" in content
        assert "signer_S1.html" in content
        assert "Visual Signer Verification" in content


class TestPipelineAndCli:
    def test_run_pipeline_success(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "dataset"
        create_mock_dataset(data_dir, words=("air", "demam"), train_count=4, test_count=2)
        out_dir = tmp_path / "sheets"

        code = run_pipeline(
            data_dir=data_dir,
            output_dir=out_dir,
            words=["air", "demam"],
            thumb_width=60,
            thumb_height=60,
        )

        assert code == 0
        assert (out_dir / "air.html").exists()
        assert (out_dir / "demam.html").exists()
        assert (out_dir / "index.html").exists()
        assert (out_dir / "thumbnails" / "air").is_dir()
        assert (out_dir / "thumbnails" / "demam").is_dir()

    def test_run_pipeline_by_signer(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "dataset"
        create_mock_dataset(data_dir, words=("air",), train_count=4, test_count=2)

        csv_file = tmp_path / "signers.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "signer", "label"])
            writer.writerow(["air_1", "S1", "air"])
            writer.writerow(["air_2", "S1", "air"])
            writer.writerow(["air_3", "S2", "air"])
            writer.writerow(["air_4", "S2", "air"])
            writer.writerow(["air_301", "S3", "air"])
            writer.writerow(["air_302", "S3", "air"])

        out_dir = tmp_path / "sheets"
        code = run_pipeline(
            data_dir=data_dir,
            output_dir=out_dir,
            by_signer=csv_file,
            words=["air"],
            thumb_width=60,
            thumb_height=60,
        )

        assert code == 0
        assert (out_dir / "signer_S1.html").exists()
        assert (out_dir / "signer_S2.html").exists()
        assert (out_dir / "signer_S3.html").exists()

    def test_run_pipeline_missing_dataset(self, tmp_path: Path) -> None:
        code = run_pipeline(data_dir=tmp_path / "nonexistent", output_dir=tmp_path / "out")
        assert code == 1

    def test_run_pipeline_missing_annotations(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "dataset"
        create_mock_dataset(data_dir, words=("air",))
        code = run_pipeline(
            data_dir=data_dir,
            output_dir=tmp_path / "out",
            by_signer=tmp_path / "missing.csv",
        )
        assert code == 1

    def test_main_cli(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "dataset"
        create_mock_dataset(data_dir, words=("air",), train_count=2, test_count=1)
        out_dir = tmp_path / "sheets"

        code = main([
            "--data-dir", str(data_dir),
            "--output-dir", str(out_dir),
            "--words", "air",
            "--thumb-width", "50",
            "--thumb-height", "50",
        ])

        assert code == 0
        assert (out_dir / "air.html").exists()
        assert (out_dir / "index.html").exists()


class TestGitSanity:
    def test_no_sheets_or_images_tracked(self) -> None:
        """Verify no generated sheets, images, or thumbnails in data/raw are tracked by git."""
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().splitlines()
        for line in lines:
            # First 2 chars are status codes
            filepath = line[3:].strip()
            assert not filepath.startswith("data/raw/"), f"File in data/raw/ is tracked: {filepath}"
            assert not filepath.endswith(".html"), f"HTML contact sheet is tracked: {filepath}"
            assert not filepath.endswith((".jpg", ".jpeg", ".png")), f"Image file is tracked: {filepath}"
