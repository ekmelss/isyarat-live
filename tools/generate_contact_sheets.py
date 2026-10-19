"""Contact sheet generator for visual signer verification in MyWSL2023.

Generates HTML contact sheets displaying downscaled thumbnails with image
sequence numbers to identify signer runs, and supports generating per-signer
contact sheets to verify signer consistency across words.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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

WORD_TRANSLATIONS: dict[str, str] = {
    "air": "water",
    "demam": "fever",
    "dengar": "hear",
    "makan": "eat",
    "minum": "drink",
    "salah": "wrong",
    "saya": "I",
    "senyap": "quiet",
    "tidur": "sleep",
    "waktu": "time",
}

SIGNER_COLORS: dict[str, str] = {
    "S1": "#ef4444",
    "S2": "#3b82f6",
    "S3": "#10b981",
    "S4": "#f59e0b",
    "S5": "#a855f7",
}


@dataclass
class ImageItem:
    """Represents a single frame in the dataset."""

    word: str
    index: int
    split: str  # 'train' or 'test'
    source_path: Path
    filename: str
    signer: str | None = None
    thumb_path: Path | None = None


def find_dataset_root(data_dir: Path | str) -> Path | None:
    """Locate the MyWSL2023 dataset directory under data_dir.

    Prioritizes full raw frames over tight crops to ensure clothing/sleeves/hands
    are visible for signer verification.
    """
    base = Path(data_dir)
    if not base.exists():
        return None

    # 1. Direct train and test under base
    if (base / "train").is_dir() and (base / "test").is_dir():
        return base

    # 2. Check explicitly named RAW directory
    raw_dir = base / "MyWSL2023 RAW DATA"
    if raw_dir.is_dir() and (raw_dir / "train").is_dir() and (raw_dir / "test").is_dir():
        return raw_dir

    # 3. Check any directory with 'raw' in its name
    for item in sorted(base.iterdir()):
        if item.is_dir() and "raw" in item.name.lower():
            if (item / "train").is_dir() and (item / "test").is_dir():
                return item

    # 4. Check explicitly named CROP directory
    crop_dir = base / "MyWSL2023 CROP DATA"
    if crop_dir.is_dir() and (crop_dir / "train").is_dir() and (crop_dir / "test").is_dir():
        return crop_dir

    # 5. Check any directory containing train and test
    for item in sorted(base.iterdir()):
        if item.is_dir() and (item / "train").is_dir() and (item / "test").is_dir():
            return item

    # 6. Check if word subdirectories exist directly under base
    if any((base / w).is_dir() for w in WORDS):
        return base

    return None


def extract_index_from_filename(filename: str) -> int | None:
    """Extract integer sequence number from filename like 'air (12).jpg' or 'air_0012.jpg'."""
    # Pattern 1: Parentheses like 'word (12).jpg'
    match = re.search(r"\((\d+)\)", filename)
    if match:
        return int(match.group(1))

    # Pattern 2: Trailing digits like 'air_0012.jpg' or 'air-12.jpg'
    stem = Path(filename).stem
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))

    # Pattern 3: Any digits
    match = re.search(r"(\d+)", stem)
    if match:
        return int(match.group(1))

    return None


def discover_word_images(dataset_root: Path, word: str) -> list[ImageItem]:
    """Discover and sort all images for a word across train and test splits."""
    items: list[ImageItem] = []
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Check train/ and test/ layouts
    train_dir = dataset_root / "train" / word
    test_dir = dataset_root / "test" / word

    for split_name, s_dir in [("train", train_dir), ("test", test_dir)]:
        if s_dir.is_dir():
            for f in s_dir.iterdir():
                if f.is_file() and f.suffix.lower() in valid_exts:
                    idx = extract_index_from_filename(f.name)
                    if idx is None:
                        idx = len(items) + 1
                    items.append(
                        ImageItem(
                            word=word,
                            index=idx,
                            split=split_name,
                            source_path=f,
                            filename=f.name,
                        )
                    )

    # If neither train nor test directory existed, check directly under word directory
    if not items:
        direct_dir = dataset_root / word
        if direct_dir.is_dir():
            for f in direct_dir.iterdir():
                if f.is_file() and f.suffix.lower() in valid_exts:
                    idx = extract_index_from_filename(f.name)
                    if idx is None:
                        idx = len(items) + 1
                    split = "test" if idx > 300 else "train"
                    items.append(
                        ImageItem(
                            word=word,
                            index=idx,
                            split=split,
                            source_path=f,
                            filename=f.name,
                        )
                    )

    # Sort strictly by index
    items.sort(key=lambda x: (x.index, x.filename))
    return items


def load_annotations(csv_path: Path | str) -> dict[tuple[str, int], str]:
    """Load signer annotations from a CSV file.

    Supports multiple column schemas:
    - Referee format: sample_id, signer, label (sample_id e.g. 'air_0001' or 'air_1')
    - Column format: word/label, index/seq, signer
    - Range format: word/label, start_idx, end_idx, signer
    - Filename format: filename/path, signer

    Returns a dict mapping (word, index) -> signer_id.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotations CSV file not found: {path}")

    mapping: dict[tuple[str, int], str] = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return mapping

        field_map = {col.strip().lower(): col for col in reader.fieldnames}

        signer_col = next(
            (field_map[k] for k in ["signer", "signer_id", "person"] if k in field_map),
            None,
        )
        if not signer_col:
            raise ValueError(
                f"Annotations CSV must have a 'signer' column. Found: {reader.fieldnames}"
            )

        word_col = next(
            (field_map[k] for k in ["label", "word", "class"] if k in field_map),
            None,
        )
        idx_col = next(
            (
                field_map[k]
                for k in ["index", "idx", "sequence", "seq", "frame", "image_index"]
                if k in field_map
            ),
            None,
        )
        sample_id_col = next(
            (field_map[k] for k in ["sample_id", "id"] if k in field_map),
            None,
        )
        file_col = next(
            (field_map[k] for k in ["filename", "file", "image_path", "path"] if k in field_map),
            None,
        )
        start_col = next(
            (field_map[k] for k in ["start", "start_idx", "start_index", "from"] if k in field_map),
            None,
        )
        end_col = next(
            (field_map[k] for k in ["end", "end_idx", "end_index", "to"] if k in field_map),
            None,
        )

        for row in reader:
            signer = row[signer_col].strip()
            if not signer:
                continue

            # Case A: Range definition (word, start, end, signer)
            if word_col and start_col and end_col and row.get(start_col) and row.get(end_col):
                w_val = row[word_col].strip().lower()
                try:
                    s_idx = int(row[start_col].strip())
                    e_idx = int(row[end_col].strip())
                    for i in range(s_idx, e_idx + 1):
                        mapping[(w_val, i)] = signer
                    continue
                except ValueError:
                    pass

            # Case B: Direct (word, index)
            if word_col and idx_col and row.get(word_col) and row.get(idx_col):
                w_val = row[word_col].strip().lower()
                i_val = row[idx_col].strip()
                if i_val.isdigit():
                    mapping[(w_val, int(i_val))] = signer
                    continue

            # Case C: sample_id format (e.g. 'air_0001', 'air_1', 'air (1)')
            if sample_id_col and row.get(sample_id_col):
                s_id = row[sample_id_col].strip()
                m = re.match(r"^([a-zA-Z]+)[_ \-]*\(?0*(\d+)\)?$", s_id)
                if m:
                    w_val = m.group(1).lower()
                    i_val = int(m.group(2))
                    mapping[(w_val, i_val)] = signer
                    continue

            # Case D: Filename / path (e.g. 'train/air/air (1).jpg')
            if file_col and row.get(file_col):
                f_name = Path(row[file_col].strip()).name
                idx = extract_index_from_filename(f_name)
                w_val = None
                if word_col and row.get(word_col):
                    w_val = row[word_col].strip().lower()
                else:
                    m = re.match(r"^([a-zA-Z]+)", f_name)
                    if m:
                        w_val = m.group(1).lower()
                if w_val and idx is not None:
                    mapping[(w_val, idx)] = signer

    return mapping


def generate_thumbnail(
    src_path: Path,
    dest_path: Path,
    max_size: tuple[int, int] = (240, 240),
    quality: int = 75,
    force: bool = False,
) -> Path:
    """Generate downscaled JPEG thumbnail preserving aspect ratio and face pixelation."""
    if Image is None:
        raise ImportError(
            "Pillow is required for thumbnail generation. Install with 'uv sync'."
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and dest_path.exists() and dest_path.stat().st_size > 0:
        if dest_path.stat().st_mtime >= src_path.stat().st_mtime:
            return dest_path

    with Image.open(src_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        thumb = img.copy()
        thumb.thumbnail(max_size, Image.Resampling.BILINEAR)
        thumb.save(dest_path, "JPEG", quality=quality, optimize=True)

    return dest_path


def batch_generate_thumbnails(
    items: list[ImageItem],
    thumb_base_dir: Path,
    max_size: tuple[int, int] = (240, 240),
    quality: int = 75,
    force: bool = False,
    workers: int | None = None,
) -> None:
    """Generate thumbnails concurrently using a thread pool."""
    tasks: list[tuple[ImageItem, Path]] = []
    for item in items:
        thumb_file = thumb_base_dir / item.word / f"{item.word}_{item.index}.jpg"
        item.thumb_path = thumb_file
        tasks.append((item, thumb_file))

    num_workers = workers or min(32, (os.cpu_count() or 4) + 4)
    logger.info(
        "Generating %d thumbnails (workers=%d, max_size=%s)...",
        len(tasks),
        num_workers,
        max_size,
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(
                generate_thumbnail,
                item.source_path,
                dest,
                max_size=max_size,
                quality=quality,
                force=force,
            ): item
            for item, dest in tasks
        }
        completed = 0
        total = len(tasks)
        for future in as_completed(future_map):
            try:
                future.result()
            except Exception as exc:
                item = future_map[future]
                logger.warning("Failed to generate thumbnail for %s: %s", item.filename, exc)
            completed += 1
            if completed % 500 == 0 or completed == total:
                logger.info(
                    "Thumbnails progress: %d / %d (%.1f%%)",
                    completed,
                    total,
                    (completed / total) * 100,
                )


COMMON_CSS = """
:root {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-card: #1e293b;
  --bg-card-hover: #243046;
  --border-color: #334155;
  --border-hover: #38bdf8;
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --accent: #38bdf8;
  --badge-train: #0284c7;
  --badge-test: #7c3aed;
  --s1-color: #ef4444;
  --s2-color: #3b82f6;
  --s3-color: #10b981;
  --s4-color: #f59e0b;
  --s5-color: #a855f7;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.5;
  padding-bottom: 80px;
}
header.top-bar {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(15, 23, 42, 0.94);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border-color);
  padding: 12px 24px;
}
.header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
.title-group h1 {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--text-primary);
}
.title-group h1 span.subtitle {
  font-size: 1rem;
  font-weight: 400;
  color: var(--text-secondary);
  margin-left: 8px;
}
.breadcrumbs {
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-bottom: 4px;
}
.breadcrumbs a {
  color: var(--accent);
  text-decoration: none;
}
.breadcrumbs a:hover { text-decoration: underline; }
.nav-pills {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}
.pill {
  display: inline-block;
  padding: 4px 10px;
  font-size: 0.82rem;
  font-weight: 500;
  border-radius: 6px;
  background: var(--bg-secondary);
  color: var(--text-secondary);
  border: 1px solid var(--border-color);
  text-decoration: none;
  transition: all 0.15s ease;
}
.pill:hover {
  background: #2d3b4e;
  color: var(--text-primary);
  border-color: var(--accent);
}
.pill.active {
  background: var(--accent);
  color: #0f172a;
  font-weight: 700;
  border-color: var(--accent);
}
.pill.signer-pill {
  border-left-width: 4px;
}
.pill.s1 { border-left-color: var(--s1-color); }
.pill.s2 { border-left-color: var(--s2-color); }
.pill.s3 { border-left-color: var(--s3-color); }
.pill.s4 { border-left-color: var(--s4-color); }
.pill.s5 { border-left-color: var(--s5-color); }

.info-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  padding: 12px 24px;
  background: rgba(30, 41, 59, 0.5);
  border-bottom: 1px solid var(--border-color);
  font-size: 0.88rem;
}
.stats-group {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  align-items: center;
}
.stat-item {
  color: var(--text-secondary);
}
.stat-item strong {
  color: var(--text-primary);
}
.controls-group {
  display: flex;
  gap: 10px;
  align-items: center;
}
.btn-toggle {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  padding: 4px 10px;
  font-size: 0.8rem;
  border-radius: 4px;
  cursor: pointer;
}
.btn-toggle:hover, .btn-toggle.active {
  color: var(--text-primary);
  border-color: var(--accent);
}

.jump-bar {
  padding: 8px 24px;
  background: rgba(15, 23, 42, 0.6);
  border-bottom: 1px solid var(--border-color);
  font-size: 0.82rem;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.jump-label {
  color: var(--text-muted);
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.72rem;
  margin-right: 4px;
}
.jump-link {
  color: var(--accent);
  text-decoration: none;
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(56, 189, 248, 0.08);
}
.jump-link:hover {
  background: rgba(56, 189, 248, 0.2);
}

.container {
  padding: 20px 24px;
}
.section-title {
  font-size: 1.15rem;
  font-weight: 600;
  margin: 24px 0 12px;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-title span.count {
  font-size: 0.85rem;
  font-weight: normal;
  color: var(--text-muted);
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}
.grid.size-sm { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 8px; }
.grid.size-md { grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.grid.size-lg { grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }

.card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow: hidden;
  transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
  cursor: pointer;
  display: flex;
  flex-direction: column;
}
.card:hover {
  transform: translateY(-2px);
  border-color: var(--border-hover);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.5);
  background: var(--bg-card-hover);
}
.card-img-wrap {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.card-img-wrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.badge {
  position: absolute;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 4px;
  line-height: 1.2;
}
.badge-seq {
  top: 6px;
  left: 6px;
  background: rgba(0, 0, 0, 0.75);
  color: #fff;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.badge-split {
  bottom: 6px;
  left: 6px;
  font-size: 10px;
  text-transform: uppercase;
}
.badge-split.train { background: var(--badge-train); color: #fff; }
.badge-split.test { background: var(--badge-test); color: #fff; }

.badge-signer {
  top: 6px;
  right: 6px;
  font-weight: 700;
  font-size: 11px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
}
.badge-signer.s1 { background: var(--s1-color); color: #fff; }
.badge-signer.s2 { background: var(--s2-color); color: #fff; }
.badge-signer.s3 { background: var(--s3-color); color: #fff; }
.badge-signer.s4 { background: var(--s4-color); color: #000; }
.badge-signer.s5 { background: var(--s5-color); color: #fff; }

.card-footer {
  padding: 8px 10px;
  font-size: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: var(--text-secondary);
  border-top: 1px solid var(--border-color);
  background: rgba(0, 0, 0, 0.2);
}
.card-footer .seq-text {
  font-weight: 600;
  color: var(--text-primary);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.card-footer .file-text {
  font-size: 11px;
  color: var(--text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 60%;
}

/* Lightbox Modal */
.modal {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: rgba(0, 0, 0, 0.88);
  backdrop-filter: blur(8px);
  align-items: center;
  justify-content: center;
}
.modal.open {
  display: flex;
}
.modal-content {
  position: relative;
  max-width: 90vw;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.modal-img-container {
  max-width: 85vw;
  max-height: 75vh;
  display: flex;
  align-items: center;
  justify-content: center;
}
.modal-img-container img {
  max-width: 100%;
  max-height: 75vh;
  border-radius: 8px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8);
  object-fit: contain;
}
.modal-caption {
  margin-top: 14px;
  color: var(--text-primary);
  font-size: 1rem;
  text-align: center;
}
.modal-caption .sub {
  color: var(--text-secondary);
  font-size: 0.85rem;
  margin-top: 4px;
}
.modal-btn {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  background: rgba(30, 41, 59, 0.8);
  border: 1px solid var(--border-color);
  color: #fff;
  font-size: 2rem;
  width: 52px;
  height: 52px;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s ease;
}
.modal-btn:hover {
  background: var(--accent);
  color: #0f172a;
}
.modal-btn.prev { left: -70px; }
.modal-btn.next { right: -70px; }
.modal-close {
  position: absolute;
  top: -46px;
  right: 0;
  background: none;
  border: none;
  color: #fff;
  font-size: 2rem;
  cursor: pointer;
}
.modal-close:hover { color: var(--accent); }

/* Index sheet specific */
.index-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
  margin-top: 16px;
}
.index-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  overflow: hidden;
  text-decoration: none;
  color: inherit;
  transition: all 0.2s ease;
  display: flex;
  flex-direction: column;
}
.index-card:hover {
  transform: translateY(-3px);
  border-color: var(--border-hover);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}
.index-card-img {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
  overflow: hidden;
}
.index-card-img img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.index-card-body {
  padding: 16px;
}
.index-card-title {
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.index-card-sub {
  color: var(--text-secondary);
  font-size: 0.9rem;
  margin-top: 4px;
}
.index-card-meta {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border-color);
  font-size: 0.82rem;
  color: var(--text-muted);
  display: flex;
  justify-content: space-between;
}
.anchor-box {
  background: rgba(30, 41, 59, 0.6);
  border-left: 4px solid var(--accent);
  padding: 12px 16px;
  border-radius: 0 8px 8px 0;
  margin: 16px 0;
  font-size: 0.88rem;
  color: var(--text-secondary);
}
.anchor-box strong { color: var(--text-primary); }
"""

JS_TEMPLATE = """
<script>
const imagesData = __IMAGES_JSON__;
let currentIndex = -1;

function setGridSize(size) {
  const grid = document.getElementById('imageGrid');
  if (!grid) return;
  grid.classList.remove('size-sm', 'size-md', 'size-lg');
  grid.classList.add('size-' + size);
  document.querySelectorAll('.btn-toggle').forEach(function(b) { b.classList.remove('active'); });
  const btn = document.getElementById('btn-size-' + size);
  if (btn) btn.classList.add('active');
}

function openModal(index) {
  currentIndex = index;
  updateModal();
  document.getElementById('modal').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  document.getElementById('modal').classList.remove('open');
  document.body.style.overflow = '';
}

function prevImage() {
  if (currentIndex > 0) {
    currentIndex--;
    updateModal();
  }
}

function nextImage() {
  if (currentIndex < imagesData.length - 1) {
    currentIndex++;
    updateModal();
  }
}

function updateModal() {
  if (currentIndex < 0 || currentIndex >= imagesData.length) return;
  const item = imagesData[currentIndex];
  const imgEl = document.getElementById('modal-img');
  imgEl.src = item.src;
  
  const captionEl = document.getElementById('modal-caption');
  let signerHtml = item.signer ? '<span style="color:var(--accent); font-weight:700;"> [' + item.signer + ']</span>' : '';
  captionEl.innerHTML = '<strong>' + item.word + ' #' + item.seq + '</strong> (' + item.split + ')' + signerHtml +
    '<div class="sub">' + item.filename + ' &bull; Frame ' + (currentIndex + 1) + ' of ' + imagesData.length + '</div>';
}

window.addEventListener('keydown', function(e) {
  const modal = document.getElementById('modal');
  if (!modal || !modal.classList.contains('open')) return;
  if (e.key === 'Escape') closeModal();
  if (e.key === 'ArrowLeft') prevImage();
  if (e.key === 'ArrowRight') nextImage();
});
</script>
"""


def render_word_sheet(
    word: str,
    images: list[ImageItem],
    output_html_path: Path,
    all_words: list[str],
    all_signers: list[str],
) -> Path:
    """Render a standalone HTML contact sheet for a single word."""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    html_dir = output_html_path.parent

    js_items: list[dict[str, Any]] = []
    cards_html: list[str] = []

    signer_counts: dict[str, int] = {}
    train_count = 0
    test_count = 0

    for idx, item in enumerate(images):
        if item.split == "train":
            train_count += 1
        else:
            test_count += 1

        if item.signer:
            signer_counts[item.signer] = signer_counts.get(item.signer, 0) + 1

        rel_thumb = ""
        if item.thumb_path:
            rel_thumb = os.path.relpath(item.thumb_path, html_dir).replace("\\", "/")

        js_items.append({
            "seq": item.index,
            "word": item.word,
            "split": item.split,
            "src": rel_thumb,
            "filename": item.filename,
            "signer": item.signer or "",
        })

        signer_badge_html = ""
        if item.signer:
            s_class = item.signer.lower()
            signer_badge_html = f'<span class="badge badge-signer {s_class}">{html.escape(item.signer)}</span>'

        split_class = "train" if item.split == "train" else "test"
        anchor_id = f"frame-{item.index}"

        card = f"""
        <div class="card" id="{anchor_id}" onclick="openModal({idx})">
          <div class="card-img-wrap">
            <img src="{html.escape(rel_thumb)}" alt="{html.escape(word)} #{item.index}" loading="lazy" width="240" height="135">
            <span class="badge badge-seq">#{item.index}</span>
            <span class="badge badge-split {split_class}">{item.split}</span>
            {signer_badge_html}
          </div>
          <div class="card-footer">
            <span class="seq-text">#{item.index}</span>
            <span class="file-text" title="{html.escape(item.filename)}">{html.escape(item.filename)}</span>
          </div>
        </div>
        """
        cards_html.append(card)

    word_pills: list[str] = []
    for w in all_words:
        active_cls = " active" if w == word else ""
        word_pills.append(f'<a href="{w}.html" class="pill{active_cls}">{w}</a>')

    signer_pills: list[str] = []
    for s in all_signers:
        s_cls = s.lower()
        signer_pills.append(f'<a href="signer_{s}.html" class="pill signer-pill {s_cls}">{s}</a>')

    jump_links: list[str] = []
    jump_targets = [1, 50, 100, 150, 200, 250, 300, 301]
    for target in jump_targets:
        if any(item.index == target for item in images):
            jump_links.append(f'<a href="#frame-{target}" class="jump-link">#{target}</a>')

    signer_breakdown_html = ""
    if signer_counts:
        parts = [
            f'<span style="color:var(--{s.lower()}-color, #fff)">{s}: {cnt}</span>'
            for s, cnt in sorted(signer_counts.items())
        ]
        signer_breakdown_html = f'<div class="stat-item">Signers: {" &bull; ".join(parts)}</div>'

    translation = WORD_TRANSLATIONS.get(word, "")
    translation_sub = f'<span class="subtitle">({translation})</span>' if translation else ""

    images_json = json.dumps(js_items)
    js_block = JS_TEMPLATE.replace("__IMAGES_JSON__", images_json)

    signers_pill_group = (
        f"<span style='font-size:0.8rem; color:var(--text-muted); align-self:center; margin:0 4px 0 12px;'>Signers:</span>{''.join(signer_pills)}"
        if signer_pills
        else ""
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Contact Sheet: {word} | Isyarat Live</title>
  <style>
{COMMON_CSS}
  </style>
</head>
<body>
  <header class="top-bar">
    <div class="breadcrumbs">
      <a href="index.html">&larr; Overview</a> / Words / <strong>{word}</strong>
    </div>
    <div class="header-row">
      <div class="title-group">
        <h1>Word: {word} {translation_sub}</h1>
      </div>
      <div class="controls-group">
        <span style="font-size:0.8rem; color:var(--text-muted)">Grid:</span>
        <button id="btn-size-sm" class="btn-toggle" onclick="setGridSize('sm')">Small</button>
        <button id="btn-size-md" class="btn-toggle active" onclick="setGridSize('md')">Medium</button>
        <button id="btn-size-lg" class="btn-toggle" onclick="setGridSize('lg')">Large</button>
      </div>
    </div>
    <div class="nav-pills">
      <span style="font-size:0.8rem; color:var(--text-muted); align-self:center; margin-right:4px;">Words:</span>
      {"".join(word_pills)}
      {signers_pill_group}
    </div>
  </header>

  <div class="info-bar">
    <div class="stats-group">
      <div class="stat-item">Total: <strong>{len(images)}</strong> frames</div>
      <div class="stat-item">Train (1–300): <strong>{train_count}</strong></div>
      <div class="stat-item">Test (301–350): <strong>{test_count}</strong></div>
      {signer_breakdown_html}
    </div>
  </div>

  <div class="jump-bar">
    <span class="jump-label">Quick Jump:</span>
    {"".join(jump_links)}
  </div>

  <div class="container">
    <div class="grid size-md" id="imageGrid">
      {"".join(cards_html)}
    </div>
  </div>

  <!-- Lightbox Modal -->
  <div id="modal" class="modal">
    <div class="modal-content">
      <button class="modal-close" onclick="closeModal()">&times;</button>
      <button class="modal-btn prev" onclick="prevImage()">&lsaquo;</button>
      <button class="modal-btn next" onclick="nextImage()">&rsaquo;</button>
      <div class="modal-img-container">
        <img id="modal-img" src="" alt="preview">
      </div>
      <div class="modal-caption" id="modal-caption"></div>
    </div>
  </div>

{js_block}
</body>
</html>
"""
    output_html_path.write_text(html_content, encoding="utf-8")
    return output_html_path


def render_signer_sheet(
    signer_id: str,
    images: list[ImageItem],
    output_html_path: Path,
    all_words: list[str],
    all_signers: list[str],
) -> Path:
    """Render a standalone per-signer verification sheet grouping frames by word."""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    html_dir = output_html_path.parent

    words_map: dict[str, list[ImageItem]] = {w: [] for w in all_words}
    for item in images:
        if item.word in words_map:
            words_map[item.word].append(item)
        else:
            words_map[item.word] = [item]

    js_items: list[dict[str, Any]] = []
    sections_html: list[str] = []
    global_idx = 0

    for word, w_items in words_map.items():
        if not w_items:
            continue

        w_cards: list[str] = []
        w_translation = WORD_TRANSLATIONS.get(word, "")
        trans_str = f" ({w_translation})" if w_translation else ""

        for item in w_items:
            rel_thumb = ""
            if item.thumb_path:
                rel_thumb = os.path.relpath(item.thumb_path, html_dir).replace("\\", "/")

            js_items.append({
                "seq": item.index,
                "word": item.word,
                "split": item.split,
                "src": rel_thumb,
                "filename": item.filename,
                "signer": item.signer or signer_id,
            })

            s_class = signer_id.lower()
            split_class = "train" if item.split == "train" else "test"

            card = f"""
            <div class="card" onclick="openModal({global_idx})">
              <div class="card-img-wrap">
                <img src="{html.escape(rel_thumb)}" alt="{html.escape(word)} #{item.index}" loading="lazy" width="240" height="135">
                <span class="badge badge-seq">#{item.index}</span>
                <span class="badge badge-split {split_class}">{item.split}</span>
                <span class="badge badge-signer {s_class}">{html.escape(signer_id)}</span>
              </div>
              <div class="card-footer">
                <span class="seq-text">#{item.index}</span>
                <span class="file-text" title="{html.escape(item.filename)}">{html.escape(item.filename)}</span>
              </div>
            </div>
            """
            w_cards.append(card)
            global_idx += 1

        min_idx = min(it.index for it in w_items)
        max_idx = max(it.index for it in w_items)

        section = f"""
        <div id="section-{word}">
          <h2 class="section-title">
            <a href="{word}.html" style="color:var(--text-primary); text-decoration:none;">{word}{trans_str}</a>
            <span class="count">{len(w_items)} frames (#{min_idx}&ndash;#{max_idx})</span>
          </h2>
          <div class="grid size-md">
            {"".join(w_cards)}
          </div>
        </div>
        """
        sections_html.append(section)

    word_pills = [f'<a href="{w}.html" class="pill">{w}</a>' for w in all_words]
    signer_pills: list[str] = []
    for s in all_signers:
        active_cls = " active" if s == signer_id else ""
        s_cls = s.lower()
        signer_pills.append(
            f'<a href="signer_{s}.html" class="pill signer-pill {s_cls}{active_cls}">{s}</a>'
        )

    jump_links = [
        f'<a href="#section-{w}" class="jump-link">{w}</a>'
        for w in all_words
        if words_map.get(w)
    ]

    images_json = json.dumps(js_items)
    js_block = JS_TEMPLATE.replace("__IMAGES_JSON__", images_json)
    s_color = SIGNER_COLORS.get(signer_id, "var(--accent)")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Signer Verification: {signer_id} | Isyarat Live</title>
  <style>
{COMMON_CSS}
  </style>
</head>
<body>
  <header class="top-bar">
    <div class="breadcrumbs">
      <a href="index.html">&larr; Overview</a> / Signers / <strong style="color:{s_color};">{signer_id}</strong>
    </div>
    <div class="header-row">
      <div class="title-group">
        <h1>Signer Verification: <span style="color:{s_color};">{signer_id}</span></h1>
      </div>
      <div class="controls-group">
        <span style="font-size:0.8rem; color:var(--text-muted)">Grid:</span>
        <button id="btn-size-sm" class="btn-toggle" onclick="setGridSize('sm')">Small</button>
        <button id="btn-size-md" class="btn-toggle active" onclick="setGridSize('md')">Medium</button>
        <button id="btn-size-lg" class="btn-toggle" onclick="setGridSize('lg')">Large</button>
      </div>
    </div>
    <div class="nav-pills">
      <span style="font-size:0.8rem; color:var(--text-muted); align-self:center; margin-right:4px;">Signers:</span>
      {"".join(signer_pills)}
      <span style="font-size:0.8rem; color:var(--text-muted); align-self:center; margin:0 4px 0 12px;">Words:</span>
      {"".join(word_pills)}
    </div>
  </header>

  <div class="info-bar">
    <div class="stats-group">
      <div class="stat-item">Signer ID: <strong style="color:{s_color};">{signer_id}</strong></div>
      <div class="stat-item">Total Assigned Frames: <strong>{len(images)}</strong></div>
      <div class="stat-item">Words Covered: <strong>{sum(1 for w, items in words_map.items() if items)} / {len(all_words)}</strong></div>
    </div>
  </div>

  <div class="jump-bar">
    <span class="jump-label">Jump to Word:</span>
    {"".join(jump_links)}
  </div>

  <div class="container" id="imageGrid">
    {"".join(sections_html)}
  </div>

  <!-- Lightbox Modal -->
  <div id="modal" class="modal">
    <div class="modal-content">
      <button class="modal-close" onclick="closeModal()">&times;</button>
      <button class="modal-btn prev" onclick="prevImage()">&lsaquo;</button>
      <button class="modal-btn next" onclick="nextImage()">&rsaquo;</button>
      <div class="modal-img-container">
        <img id="modal-img" src="" alt="preview">
      </div>
      <div class="modal-caption" id="modal-caption"></div>
    </div>
  </div>

{js_block}
</body>
</html>
"""
    output_html_path.write_text(html_content, encoding="utf-8")
    return output_html_path


def render_index_sheet(
    words: list[str],
    word_images: dict[str, list[ImageItem]],
    signers: list[str],
    signer_images: dict[str, list[ImageItem]],
    output_html_path: Path,
) -> Path:
    """Render the main index page linking to all word and signer sheets."""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    html_dir = output_html_path.parent

    word_cards: list[str] = []
    for w in words:
        items = word_images.get(w, [])
        first_thumb = ""
        if items and items[0].thumb_path:
            first_thumb = os.path.relpath(items[0].thumb_path, html_dir).replace("\\", "/")

        trans = WORD_TRANSLATIONS.get(w, "")
        total_cnt = len(items)
        train_cnt = sum(1 for it in items if it.split == "train")
        test_cnt = sum(1 for it in items if it.split == "test")

        img_tag = (
            f'<img src="{html.escape(first_thumb)}" alt="{w}" loading="lazy">'
            if first_thumb
            else ""
        )
        card = f"""
        <a href="{w}.html" class="index-card">
          <div class="index-card-img">
            {img_tag}
          </div>
          <div class="index-card-body">
            <div class="index-card-title">
              <span>{w}</span>
              <span style="font-size:0.8rem; font-weight:normal; color:var(--accent);">&rarr;</span>
            </div>
            <div class="index-card-sub">{trans}</div>
            <div class="index-card-meta">
              <span>{total_cnt} frames</span>
              <span>Train: {train_cnt} &bull; Test: {test_cnt}</span>
            </div>
          </div>
        </a>
        """
        word_cards.append(card)

    signer_cards: list[str] = []
    for s in signers:
        s_items = signer_images.get(s, [])
        first_thumb = ""
        if s_items and s_items[0].thumb_path:
            first_thumb = os.path.relpath(s_items[0].thumb_path, html_dir).replace("\\", "/")

        s_color = SIGNER_COLORS.get(s, "var(--accent)")
        words_count = len({it.word for it in s_items})
        img_tag = (
            f'<img src="{html.escape(first_thumb)}" alt="{s}" loading="lazy">'
            if first_thumb
            else ""
        )

        card = f"""
        <a href="signer_{s}.html" class="index-card" style="border-top: 3px solid {s_color};">
          <div class="index-card-img">
            {img_tag}
          </div>
          <div class="index-card-body">
            <div class="index-card-title">
              <span style="color:{s_color};">{s}</span>
              <span style="font-size:0.8rem; font-weight:normal; color:var(--accent);">&rarr;</span>
            </div>
            <div class="index-card-sub">{len(s_items)} total frames across {words_count} words</div>
            <div class="index-card-meta">
              <span>Signer Verification</span>
              <span style="color:{s_color};">Inspect &rarr;</span>
            </div>
          </div>
        </a>
        """
        signer_cards.append(card)

    signers_section_html = ""
    if signer_cards:
        signers_section_html = f"""
        <h2 class="section-title" style="margin-top:40px;">
          Per-Signer Verification Sheets
          <span class="count">({len(signers)} signers)</span>
        </h2>
        <p style="color:var(--text-secondary); font-size:0.9rem; margin-bottom:16px;">
          Inspect all frames assigned to a single person across all 10 words to verify that clothing,
          sleeves, and individual characteristics remain consistent.
        </p>
        <div class="index-grid">
          {"".join(signer_cards)}
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Contact Sheets Overview | Isyarat Live</title>
  <style>
{COMMON_CSS}
  </style>
</head>
<body>
  <header class="top-bar">
    <div class="breadcrumbs">Isyarat Live / Signer Verification</div>
    <div class="header-row">
      <div class="title-group">
        <h1>Visual Signer Verification Contact Sheets</h1>
      </div>
    </div>
  </header>

  <div class="container">
    <div class="anchor-box">
      <strong>Auditing Guide (BRIEF.md &sect;4):</strong>
      MyWSL2023 contains 5 signers (2 male, 3 female) across 3,500 frames with contiguous runs.
      Check clothing and sleeve colors to confirm runs:
      <br>
      &bull; <strong>air</strong>: #1 man in red polo &bull; #70&ndash;71, #140&ndash;141 man in black T-shirt &bull;
      #210&ndash;211 woman in brown hijab &bull; #280&ndash;281 woman in black hijab &bull; #301 man in red jersey.
    </div>

    <h2 class="section-title">
      Vocabulary Contact Sheets
      <span class="count">(10 words &bull; 350 frames each)</span>
    </h2>
    <div class="index-grid">
      {"".join(word_cards)}
    </div>

    {signers_section_html}
  </div>
</body>
</html>
"""
    output_html_path.write_text(html_content, encoding="utf-8")
    return output_html_path


def run_pipeline(
    data_dir: Path | str = "data/raw",
    output_dir: Path | str = "data/raw/sheets",
    by_signer: Path | str | None = None,
    words: list[str] | None = None,
    thumb_width: int = 240,
    thumb_height: int = 240,
    thumb_quality: int = 75,
    workers: int | None = None,
    force: bool = False,
) -> int:
    """Execute the full contact sheet generation pipeline.

    Returns exit code 0 on success, non-zero on failure.
    """
    dataset_root = find_dataset_root(data_dir)
    if not dataset_root:
        logger.error(
            "Could not locate MyWSL2023 dataset in '%s'. "
            "Please run 'uv run python -m tools.download_data' first.",
            data_dir,
        )
        return 1

    logger.info("Found dataset root at: %s", dataset_root)
    out_path = Path(output_dir)
    thumb_dir = out_path / "thumbnails"
    out_path.mkdir(parents=True, exist_ok=True)

    target_words = words or list(WORDS)
    logger.info("Target words (%d): %s", len(target_words), ", ".join(target_words))

    # Optional signer annotations
    annotations: dict[tuple[str, int], str] = {}
    if by_signer:
        try:
            annotations = load_annotations(by_signer)
            logger.info("Loaded %d signer annotations from %s.", len(annotations), by_signer)
        except Exception as exc:
            logger.error("Failed to load annotations from %s: %s", by_signer, exc)
            return 1

    # Discover images per word
    all_items: list[ImageItem] = []
    word_images_map: dict[str, list[ImageItem]] = {}

    for w in target_words:
        items = discover_word_images(dataset_root, w)
        if not items:
            logger.warning("No images discovered for word: %s", w)
            continue

        for item in items:
            if (item.word, item.index) in annotations:
                item.signer = annotations[(item.word, item.index)]

        word_images_map[w] = items
        all_items.extend(items)

    if not all_items:
        logger.error("No images found for any target words in %s.", dataset_root)
        return 1

    logger.info("Discovered %d total images across %d words.", len(all_items), len(word_images_map))

    # Generate thumbnails
    batch_generate_thumbnails(
        all_items,
        thumb_base_dir=thumb_dir,
        max_size=(thumb_width, thumb_height),
        quality=thumb_quality,
        force=force,
        workers=workers,
    )

    known_signers = sorted({it.signer for it in all_items if it.signer})
    if by_signer and not known_signers:
        logger.warning("No signers were matched from annotations to discovered images.")

    # Render word contact sheets
    for w, items in word_images_map.items():
        w_html = out_path / f"{w}.html"
        render_word_sheet(
            word=w,
            images=items,
            output_html_path=w_html,
            all_words=target_words,
            all_signers=known_signers,
        )

    logger.info("Generated %d word contact sheets in %s.", len(word_images_map), out_path)

    # Render per-signer verification sheets if requested
    signer_images_map: dict[str, list[ImageItem]] = {s: [] for s in known_signers}
    for item in all_items:
        if item.signer and item.signer in signer_images_map:
            signer_images_map[item.signer].append(item)

    if by_signer and known_signers:
        for s in known_signers:
            s_html = out_path / f"signer_{s}.html"
            render_signer_sheet(
                signer_id=s,
                images=signer_images_map[s],
                output_html_path=s_html,
                all_words=target_words,
                all_signers=known_signers,
            )
        logger.info("Generated %d per-signer contact sheets (signer_*.html).", len(known_signers))

    # Render index.html
    index_html = out_path / "index.html"
    render_index_sheet(
        words=target_words,
        word_images=word_images_map,
        signers=known_signers,
        signer_images=signer_images_map,
        output_html_path=index_html,
    )
    logger.info("Generated index sheet: %s", index_html)

    logger.info("Contact sheet generation complete.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate HTML contact sheets for visual signer verification in MyWSL2023."
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Base directory containing the raw or crop MyWSL2023 dataset (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/sheets",
        help="Destination directory for HTML sheets and thumbnails (default: data/raw/sheets).",
    )
    parser.add_argument(
        "--by-signer",
        nargs="?",
        const="data/annotations/signers.csv",
        default=None,
        metavar="ANNOTATIONS_CSV",
        help=(
            "Generate per-signer verification sheets (signer_S1.html .. signer_S5.html) "
            "using the given annotations CSV file (default if flag given without arg: "
            "data/annotations/signers.csv)."
        ),
    )
    parser.add_argument(
        "--words",
        default=None,
        help="Comma-separated list of words to process (e.g. 'air,demam'). Default: all 10 words.",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=240,
        help="Maximum width for generated thumbnails in pixels (default: 240).",
    )
    parser.add_argument(
        "--thumb-height",
        type=int,
        default=240,
        help="Maximum height for generated thumbnails in pixels (default: 240).",
    )
    parser.add_argument(
        "--thumb-quality",
        type=int,
        default=75,
        help="JPEG quality for generated thumbnails (1-100, default: 75).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for parallel thumbnail generation (default: auto).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of thumbnails even if they already exist on disk.",
    )

    args = parser.parse_args(argv)

    word_list = None
    if args.words:
        word_list = [w.strip() for w in args.words.split(",") if w.strip()]

    return run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        by_signer=args.by_signer,
        words=word_list,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
        thumb_quality=args.thumb_quality,
        workers=args.workers,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
