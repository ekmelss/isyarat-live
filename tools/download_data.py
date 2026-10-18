"""MyWSL2023 dataset download and verification helper.

Downloads and verifies the MyWSL2023 dataset from Mendeley Data
(DOI: 10.17632/zvk55p7ktd.1, CC BY 4.0).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DOI = "10.17632/zvk55p7ktd.1"
DATASET_ID = "zvk55p7ktd"
DATASET_VERSION = 1

WORDS: tuple[str, ...] = (
    "air",
    "demam",
    "dengar",
    "makan",
    "minum",
    "salah",
    "saya",
    "senyap",
    "tidur",
    "waktu",
)

TRAIN_INDICES: range = range(1, 301)  # 1 to 300
TEST_INDICES: range = range(301, 351)  # 301 to 350
EXPECTED_SAMPLE_COUNT: int = 3500

DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "raw": {
        "name": "MyWSL2023 RAW DATA.zip",
        "folder_name": "MyWSL2023 RAW DATA",
        "file_id": "7f11b8a0-24e4-45df-af3d-e861f41435ea",
        "expected_size": 3318498916,
        "direct_url": (
            f"https://data.mendeley.com/public-files/datasets/{DATASET_ID}"
            f"/files/7f11b8a0-24e4-45df-af3d-e861f41435ea/file_downloaded"
        ),
    },
    "crop": {
        "name": "MyWSL2023 CROP DATA.zip",
        "folder_name": "MyWSL2023 CROP DATA",
        "file_id": "4fcb3f1a-d1d1-447c-863c-f87e92d52411",
        "expected_size": 54375021,
        "direct_url": (
            f"https://data.mendeley.com/public-files/datasets/{DATASET_ID}"
            f"/files/4fcb3f1a-d1d1-447c-863c-f87e92d52411/file_downloaded"
        ),
    },
}


def resolve_download_info(
    dataset_type: str,
    timeout: float = 10.0,
) -> tuple[str, str, int]:
    """Resolve download URL, filename, and expected size for the dataset.

    Attempts resolution via Mendeley Public API, falling back to direct public links.
    Returns (url, filename, expected_size).
    """
    if dataset_type not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset type: {dataset_type!r}. Expected one of {list(DATASET_CONFIG.keys())}."
        )

    cfg = DATASET_CONFIG[dataset_type]
    filename = cfg["name"]
    expected_size = cfg["expected_size"]
    target_file_id = cfg["file_id"]

    # Attempt dynamic API resolution
    try:
        folders_url = (
            f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
            f"/folders/{DATASET_VERSION}"
        )
        req = urllib.request.Request(
            folders_url,
            headers={"User-Agent": "Mozilla/5.0 (Isyarat-Live Data Helper)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            folders = json.loads(resp.read().decode("utf-8"))

        for folder in folders:
            folder_id = folder.get("id")
            if not folder_id:
                continue
            files_url = (
                f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
                f"/files?folder_id={folder_id}&version={DATASET_VERSION}&$start=0&$limit=1000"
            )
            req_files = urllib.request.Request(
                files_url,
                headers={"User-Agent": "Mozilla/5.0 (Isyarat-Live Data Helper)"},
            )
            with urllib.request.urlopen(req_files, timeout=timeout) as resp_files:
                files = json.loads(resp_files.read().decode("utf-8"))

            for f in files:
                f_id = f.get("id")
                f_name = f.get("name") or f.get("filename")
                if f_id == target_file_id or f_name == filename:
                    download_url = (
                        f.get("content_details", {}).get("download_url")
                        or f.get("download_url")
                    )
                    size = f.get("size") or expected_size
                    if download_url:
                        logger.info("Resolved %s via Mendeley API.", filename)
                        return download_url, filename, size
    except Exception as exc:
        logger.debug("Mendeley API lookup failed (%s); falling back to direct URL.", exc)

    logger.info("Using direct Mendeley URL for %s.", filename)
    return cfg["direct_url"], filename, expected_size


def download_file(
    url: str,
    dest_path: Path,
    expected_size: int | None = None,
    chunk_size: int = 1024 * 1024,
    resume: bool = True,
    timeout: float = 30.0,
) -> Path:
    """Download a file with resume support and progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    initial_bytes = 0
    headers = {"User-Agent": "Mozilla/5.0 (Isyarat-Live Data Helper)"}

    if resume and part_path.exists():
        initial_bytes = part_path.stat().st_size
        if expected_size and initial_bytes >= expected_size:
            # Maybe already complete
            part_path.rename(dest_path)
            return dest_path
        if initial_bytes > 0:
            headers["Range"] = f"bytes={initial_bytes}-"
            logger.info("Resuming download from byte %d...", initial_bytes)

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as err:
        if err.code == 416:  # Range Not Satisfiable
            logger.warning("Range request not satisfiable, restarting download.")
            initial_bytes = 0
            headers.pop("Range", None)
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=timeout)
        else:
            raise

    mode = "ab" if initial_bytes > 0 and resp.status == 206 else "wb"
    if mode == "wb":
        initial_bytes = 0

    content_length = resp.headers.get("Content-Length")
    total_bytes = (
        int(content_length) + initial_bytes
        if content_length
        else expected_size
    )

    downloaded = initial_bytes
    last_print = time.time()
    logger.info("Downloading to %s...", dest_path.name)

    with open(part_path, mode) as out_f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_print >= 2.0 or (total_bytes and downloaded == total_bytes):
                last_print = now
                if total_bytes:
                    pct = (downloaded / total_bytes) * 100
                    mb_cur = downloaded / (1024 * 1024)
                    mb_tot = total_bytes / (1024 * 1024)
                    logger.info("Progress: %.1f%% (%.1f / %.1f MB)", pct, mb_cur, mb_tot)
                else:
                    mb_cur = downloaded / (1024 * 1024)
                    logger.info("Downloaded: %.1f MB", mb_cur)

    part_path.rename(dest_path)
    logger.info("Download complete: %s (%.1f MB)", dest_path.name, dest_path.stat().st_size / (1024 * 1024))
    return dest_path


def verify_archive(zip_path: Path) -> tuple[bool, str]:
    """Verify zip archive integrity and internal file structure.

    Checks:
    - File exists and is non-empty.
    - Zip integrity via ZipFile.testzip().
    - Contains all expected words in train and test sets.
    """
    if not zip_path.exists():
        return False, f"Archive does not exist: {zip_path}"

    if zip_path.stat().st_size == 0:
        return False, f"Archive file is empty: {zip_path}"

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            corrupted_file = zf.testzip()
            if corrupted_file is not None:
                return False, f"Archive is corrupted at: {corrupted_file}"

            namelist = set(zf.namelist())
            # Find root prefix if any (e.g. 'MyWSL2023 RAW DATA/')
            prefix = ""
            for name in namelist:
                parts = Path(name).parts
                if len(parts) >= 2 and parts[-2] in WORDS:
                    # e.g. parts might be ('MyWSL2023 RAW DATA', 'train', 'air', 'air (1).jpg')
                    prefix_parts = parts[:-3]
                    prefix = "/".join(prefix_parts)
                    if prefix:
                        prefix += "/"
                    break

            missing: list[str] = []
            for word in WORDS:
                for idx in TRAIN_INDICES:
                    expected_entry = f"{prefix}train/{word}/{word} ({idx}).jpg"
                    if expected_entry not in namelist:
                        missing.append(expected_entry)
                        if len(missing) >= 5:
                            break
                if len(missing) >= 5:
                    break

            if missing:
                return (
                    False,
                    f"Archive is missing expected image entries (e.g. {missing[:3]}).",
                )

    except zipfile.BadZipFile as exc:
        return False, f"Invalid zip archive: {exc}"
    except Exception as exc:
        return False, f"Error inspecting archive: {exc}"

    return True, "Archive verified successfully."


def extract_archive(zip_path: Path, dest_dir: Path) -> Path:
    """Extract archive into dest_dir safely."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s into %s...", zip_path.name, dest_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Check for path traversal vulnerabilities before extracting
        dest_resolved = dest_dir.resolve()
        for member in zf.infolist():
            member_path = (dest_dir / member.filename).resolve()
            if not member_path.is_relative_to(dest_resolved):
                raise ValueError(f"Unsafe zip entry detected: {member.filename}")
        zf.extractall(dest_dir)

    logger.info("Extraction complete.")
    return dest_dir


def find_dataset_root(base_dir: Path, expected_folder_name: str | None = None) -> Path | None:
    """Locate the directory containing train/ and test/ subdirectories."""
    # Check directly
    if (base_dir / "train").is_dir() and (base_dir / "test").is_dir():
        return base_dir

    # Check named folder if given
    if expected_folder_name and (base_dir / expected_folder_name / "train").is_dir():
        return base_dir / expected_folder_name

    # Check immediate subdirectories
    if base_dir.is_dir():
        for sub in base_dir.iterdir():
            if sub.is_dir() and (sub / "train").is_dir() and (sub / "test").is_dir():
                return sub

    return None


def verify_extracted_structure(
    dataset_root: Path,
) -> tuple[bool, dict[str, int], list[str]]:
    """Verify extracted dataset directory layout, files, and sizes.

    Returns:
        (is_valid, counts_per_word, error_messages)
    """
    errors: list[str] = []
    counts: dict[str, int] = {word: 0 for word in WORDS}

    if not dataset_root.exists():
        return False, counts, [f"Dataset root does not exist: {dataset_root}"]

    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    if not train_dir.is_dir():
        errors.append(f"Missing 'train' directory in {dataset_root}")
    if not test_dir.is_dir():
        errors.append(f"Missing 'test' directory in {dataset_root}")

    if errors:
        return False, counts, errors

    for word in WORDS:
        word_train = train_dir / word
        word_test = test_dir / word

        if not word_train.is_dir():
            errors.append(f"Missing train directory for word: {word}")
        else:
            for idx in TRAIN_INDICES:
                img_path = word_train / f"{word} ({idx}).jpg"
                if not img_path.is_file():
                    if len(errors) < 20:
                        errors.append(f"Missing image: {img_path.name}")
                elif img_path.stat().st_size == 0:
                    if len(errors) < 20:
                        errors.append(f"Empty image file: {img_path.name}")
                else:
                    counts[word] += 1

        if not word_test.is_dir():
            errors.append(f"Missing test directory for word: {word}")
        else:
            for idx in TEST_INDICES:
                img_path = word_test / f"{word} ({idx}).jpg"
                if not img_path.is_file():
                    if len(errors) < 20:
                        errors.append(f"Missing image: {img_path.name}")
                elif img_path.stat().st_size == 0:
                    if len(errors) < 20:
                        errors.append(f"Empty image file: {img_path.name}")
                else:
                    counts[word] += 1

    total_count = sum(counts.values())
    if total_count != EXPECTED_SAMPLE_COUNT:
        errors.append(
            f"Expected {EXPECTED_SAMPLE_COUNT} total images, found {total_count}."
        )

    is_valid = len(errors) == 0
    return is_valid, counts, errors


def run_download_pipeline(
    dataset_type: str = "raw",
    dest_dir: Path | str = "data/raw",
    check_only: bool = False,
    no_extract: bool = False,
    force: bool = False,
) -> int:
    """Execute download, extraction, and verification pipeline.

    Returns exit code 0 on success, non-zero on error.
    """
    dest = Path(dest_dir)
    cfg = DATASET_CONFIG.get(dataset_type)
    if not cfg:
        logger.error("Unknown dataset type: %s", dataset_type)
        return 1

    zip_path = dest / cfg["name"]
    expected_folder = cfg["folder_name"]

    if check_only:
        logger.info("Checking dataset integrity in %s (dataset: %s)...", dest, dataset_type)
        dataset_root = find_dataset_root(dest, expected_folder)

        extracted_valid = False
        if dataset_root:
            logger.info("Found extracted dataset at: %s", dataset_root)
            extracted_valid, counts, errors = verify_extracted_structure(dataset_root)
            if extracted_valid:
                logger.info(
                    "Extracted structure verified: all 10 words present (350 images each, 3,500 total)."
                )
            else:
                logger.warning("Extracted structure has issues: %s", errors[:5])

        archive_valid = False
        if zip_path.exists():
            archive_valid, msg = verify_archive(zip_path)
            if archive_valid:
                logger.info("Archive verified: %s", zip_path.name)
            else:
                logger.warning("Archive verification failed: %s", msg)

        if extracted_valid or archive_valid:
            logger.info("Check PASSED.")
            return 0

        logger.error(
            "Check FAILED: neither a valid extracted directory nor a valid archive was found in %s.",
            dest,
        )
        return 1

    # Normal run: download and extract
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Download archive if not present or forced
    if not zip_path.exists() or force:
        url, filename, expected_size = resolve_download_info(dataset_type)
        try:
            download_file(url, zip_path, expected_size=expected_size, resume=not force)
        except Exception as exc:
            logger.error("Failed to download %s: %s", filename, exc)
            return 1
    else:
        logger.info("Archive %s already exists.", zip_path.name)

    # 2. Verify archive
    archive_ok, archive_msg = verify_archive(zip_path)
    if not archive_ok:
        logger.error("Archive verification failed: %s", archive_msg)
        return 1
    logger.info("Archive integrity OK.")

    # 3. Extract if requested
    if not no_extract:
        try:
            extract_archive(zip_path, dest)
        except Exception as exc:
            logger.error("Failed to extract %s: %s", zip_path.name, exc)
            return 1

        # 4. Verify extracted structure
        dataset_root = find_dataset_root(dest, expected_folder)
        if not dataset_root:
            logger.error("Extracted dataset directory could not be located in %s.", dest)
            return 1

        struct_ok, counts, errors = verify_extracted_structure(dataset_root)
        if not struct_ok:
            logger.error("Extracted dataset verification failed: %s", errors[:5])
            return 1

        logger.info("Dataset structure verified: %d total valid images across 10 words.", sum(counts.values()))

    logger.info("Operation completed successfully.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify MyWSL2023 dataset from Mendeley Data."
    )
    parser.add_argument(
        "--dataset",
        choices=["raw", "crop"],
        default="raw",
        help="Dataset variant to download: 'raw' (3.3 GB full frames) or 'crop' (54 MB tight hand crops). Default: raw.",
    )
    parser.add_argument(
        "--dest",
        default="data/raw",
        help="Destination directory for download and extraction (default: data/raw).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check local archive and extracted dataset integrity without downloading.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download archive only without extracting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if archive already exists locally.",
    )

    args = parser.parse_args(argv)
    return run_download_pipeline(
        dataset_type=args.dataset,
        dest_dir=args.dest,
        check_only=args.check,
        no_extract=args.no_extract,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
