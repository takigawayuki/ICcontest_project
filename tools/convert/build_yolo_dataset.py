"""
Build the final four-class YOLO dataset for the regional contest.

Classes:
  0 plate
  1 person
  2 car
  3 traffic_light

Main training sources:
  - CCPD2019 / CCPD2020 / CRPD_multi / CRPD_double / CLPD -> plate
  - BDD100K -> person, car, traffic_light

WTS_DATASET_TEST is intentionally not converted into YOLO training data. It can
be inventoried into meta/wts_video_manifest.csv for later video-rule testing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path(r"D:\Tempcode\26IC\车牌数据集\中国车牌")
WTS_ROOT = Path(r"D:\Tempcode\26IC\车牌数据集\中国车牌\WTS_DATASET_TEST")
OUTPUT_ROOT = ROOT / "datasets" / "yolo"

SEED = 20260709
SPLITS = ("train", "val", "test")
CLASS_NAMES = {
    0: "plate",
    1: "person",
    2: "car",
    3: "traffic_light",
}

# Preferred split sizes from doc/YOLO四类数据集转换大纲.md. If the local
# source count differs, the script falls back to an 8/1/1 image-level split.
SOURCE_COUNTS = {
    "ccpd2019": (273582, 34198, 34198),
    "ccpd2020": (9420, 1178, 1178),
    "crpd_multi": (1268, 158, 159),
    "crpd_double": (4882, 610, 610),
    "clpd": (960, 120, 120),
    "bdd100k": (80000, 10000, 10000),
}


Box = tuple[int, float, float, float, float]


@dataclass(frozen=True)
class BddTrafficLightColor:
    object_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    color: str


@dataclass(frozen=True)
class LabelResult:
    boxes: list[Box]
    bdd_traffic_lights: list[BddTrafficLightColor] = field(default_factory=list)


@dataclass(frozen=True)
class Record:
    source: str
    image_path: Path
    key: str
    label_func: Callable[[], LabelResult]


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def yolo_box(
    cls: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> Box | None:
    x1 = max(0.0, min(float(img_w), float(x1)))
    x2 = max(0.0, min(float(img_w), float(x2)))
    y1 = max(0.0, min(float(img_h), float(y1)))
    y2 = max(0.0, min(float(img_h), float(y2)))
    if x2 <= x1 or y2 <= y1 or img_w <= 0 or img_h <= 0:
        return None
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
        return None
    return cls, xc, yc, bw, bh


def parse_ccpd_bbox(filename: str, img_w: int, img_h: int) -> list[Box]:
    parts = filename.split("-")
    if len(parts) < 3:
        return []
    try:
        p1, p2 = parts[2].split("_")
        x1, y1 = map(float, p1.split("&"))
        x2, y2 = map(float, p2.split("&"))
    except ValueError:
        return []
    box = yolo_box(0, x1, y1, x2, y2, img_w, img_h)
    return [box] if box else []


def parse_quad_label(label_path: Path, img_w: int, img_h: int) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.exists():
        return boxes
    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                nums = [float(v) for v in parts[:8]]
            except ValueError:
                continue
            xs = nums[0::2]
            ys = nums[1::2]
            box = yolo_box(0, min(xs), min(ys), max(xs), max(ys), img_w, img_h)
            if box:
                boxes.append(box)
    return boxes


def parse_bdd_label(label_path: Path, img_w: int, img_h: int) -> LabelResult:
    boxes: list[Box] = []
    traffic_lights: list[BddTrafficLightColor] = []
    data = json.loads(label_path.read_text(encoding="utf-8"))
    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            category = obj.get("category")
            if category == "person":
                cls = 1
            elif category == "car":
                cls = 2
            elif category == "traffic light":
                cls = 3
            else:
                continue

            b = obj.get("box2d") or {}
            box = yolo_box(cls, b.get("x1", 0), b.get("y1", 0), b.get("x2", 0), b.get("y2", 0), img_w, img_h)
            if not box:
                continue
            boxes.append(box)

            if cls == 3:
                traffic_lights.append(
                    BddTrafficLightColor(
                        object_id=str(obj.get("id", "")),
                        x1=float(b.get("x1", 0)),
                        y1=float(b.get("y1", 0)),
                        x2=float(b.get("x2", 0)),
                        y2=float(b.get("y2", 0)),
                        color=str((obj.get("attributes") or {}).get("trafficLightColor", "none")),
                    )
                )
    return LabelResult(boxes, traffic_lights)


def make_key(source: str, rel: Path) -> str:
    stem = "__".join(rel.with_suffix("").parts)
    safe = []
    for ch in stem:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return f"{source}_{''.join(safe)}"


def collect_ccpd2019() -> list[Record]:
    base = RAW_ROOT / "CCPD2019"
    records: list[Record] = []
    seen: set[Path] = set()
    for txt_name in ("train.txt", "val.txt", "test.txt"):
        txt_path = base / "splits" / txt_name
        if not txt_path.exists():
            raise FileNotFoundError(txt_path)
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                rel = Path(raw.replace("/", "\\"))
                if rel in seen:
                    continue
                seen.add(rel)
                img_path = base / rel
                if not img_path.exists():
                    continue

                def label_func(path=img_path):
                    w, h = image_size(path)
                    return LabelResult(parse_ccpd_bbox(path.name, w, h))

                records.append(Record("ccpd2019", img_path, make_key("ccpd2019", rel), label_func))
    return records


def collect_ccpd2020() -> list[Record]:
    base = RAW_ROOT / "CCPD2020" / "ccpd_green"
    records: list[Record] = []
    for old_split in SPLITS:
        for img_path in sorted((base / old_split).glob("*.jpg")):
            rel = Path(old_split) / img_path.name

            def label_func(path=img_path):
                w, h = image_size(path)
                return LabelResult(parse_ccpd_bbox(path.name, w, h))

            records.append(Record("ccpd2020", img_path, make_key("ccpd2020", rel), label_func))
    return records


def collect_crpd(source: str, dirname: str) -> list[Record]:
    base = RAW_ROOT / dirname
    records: list[Record] = []
    for old_split in SPLITS:
        img_dir = base / old_split / "images"
        label_dir = base / old_split / "labels"
        for img_path in sorted(img_dir.glob("*.jpg")):
            rel = Path(old_split) / img_path.name
            label_path = label_dir / f"{img_path.stem}.txt"

            def label_func(path=img_path, lbl=label_path):
                w, h = image_size(path)
                return LabelResult(parse_quad_label(lbl, w, h))

            records.append(Record(source, img_path, make_key(source, rel), label_func))
    return records


def collect_clpd() -> list[Record]:
    base = RAW_ROOT / "CLPD"
    csv_path = base / "CLPD.csv"
    records: list[Record] = []
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = Path(row["path"].replace("/", "\\"))
            img_path = base / rel
            if not img_path.exists():
                continue
            coords = [float(row[f"{axis}{idx}"]) for idx in range(1, 5) for axis in ("x", "y")]

            def label_func(path=img_path, nums=coords):
                w, h = image_size(path)
                xs = nums[0::2]
                ys = nums[1::2]
                box = yolo_box(0, min(xs), min(ys), max(xs), max(ys), w, h)
                return LabelResult([box] if box else [])

            records.append(Record("clpd", img_path, make_key("clpd", rel), label_func))
    return records


def collect_bdd100k() -> list[Record]:
    base = RAW_ROOT / "BDD100K"
    img_root = base / "bdd100k_images_100k" / "100k"
    label_root = base / "bdd100k_labels" / "100k"
    records: list[Record] = []
    for old_split in SPLITS:
        img_dir = img_root / old_split
        label_dir = label_root / old_split
        for img_path in sorted(img_dir.glob("*.jpg")):
            label_path = label_dir / f"{img_path.stem}.json"
            if not label_path.exists():
                continue
            rel = Path(old_split) / img_path.name

            def label_func(path=img_path, lbl=label_path):
                w, h = image_size(path)
                return parse_bdd_label(lbl, w, h)

            records.append(Record("bdd100k", img_path, make_key("bdd100k", rel), label_func))
    return records


def split_counts(source: str, count: int) -> tuple[int, int, int]:
    preferred = SOURCE_COUNTS.get(source)
    if preferred and sum(preferred) == count:
        return preferred
    train_n = int(count * 0.8)
    val_n = int(count * 0.1)
    test_n = count - train_n - val_n
    print(f"[warn] {source}: local count={count}, expected={preferred}; using 8/1/1 fallback.")
    return train_n, val_n, test_n


def split_records(source: str, records: list[Record]) -> dict[str, list[Record]]:
    train_n, val_n, _test_n = split_counts(source, len(records))
    rng = random.Random(f"{SEED}:{source}")
    shuffled = list(records)
    rng.shuffle(shuffled)
    return {
        "train": shuffled[:train_n],
        "val": shuffled[train_n : train_n + val_n],
        "test": shuffled[train_n + val_n :],
    }


def output_is_empty(output: Path) -> bool:
    if not output.exists():
        return True
    return not any(output.rglob("*"))


def ensure_output_dirs(output: Path) -> None:
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)


def write_yaml(output: Path) -> None:
    content = f"""path: {output.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: plate
  1: person
  2: car
  3: traffic_light
"""
    (output / "yolo.yaml").write_text(content, encoding="utf-8")


def write_label(path: Path, boxes: Iterable[Box]) -> None:
    lines = [f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for cls, xc, yc, w, h in boxes]
    path.write_text("".join(lines), encoding="utf-8")


def copy_image(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def convert(records_by_split: dict[str, list[Record]], output: Path, dry_run: bool, copy_mode: str) -> dict[str, dict[str, int]]:
    stats = {
        split: {name: 0 for name in ("images", "plate", "person", "car", "traffic_light", "empty", "skipped")}
        for split in SPLITS
    }
    manifest_rows: list[list[str]] = []
    traffic_light_rows: list[list[str]] = []

    if not dry_run:
        ensure_output_dirs(output)

    for split in SPLITS:
        records = records_by_split[split]
        total = len(records)
        for idx, record in enumerate(records, 1):
            try:
                result = record.label_func()
                boxes = result.boxes
                if record.source != "bdd100k" and not boxes:
                    stats[split]["skipped"] += 1
                    continue

                ext = record.image_path.suffix.lower()
                dst_name = f"{record.key}{ext}"
                label_name = f"{record.key}.txt"

                if not dry_run:
                    copy_image(record.image_path, output / "images" / split / dst_name, copy_mode)
                    write_label(output / "labels" / split / label_name, boxes)

                stats[split]["images"] += 1
                if not boxes:
                    stats[split]["empty"] += 1
                for cls, *_ in boxes:
                    stats[split][CLASS_NAMES[cls]] += 1
                manifest_rows.append([split, record.source, dst_name, label_name, str(record.image_path)])

                for tl in result.bdd_traffic_lights:
                    traffic_light_rows.append(
                        [
                            split,
                            dst_name,
                            tl.object_id,
                            f"{tl.x1:.3f}",
                            f"{tl.y1:.3f}",
                            f"{tl.x2:.3f}",
                            f"{tl.y2:.3f}",
                            tl.color,
                        ]
                    )
            except Exception as exc:
                stats[split]["skipped"] += 1
                print(f"[skip] {split} {record.source} {record.image_path}: {exc}")

            if idx % 5000 == 0 or idx == total:
                print(f"[{split}] {idx}/{total}")

    if not dry_run:
        write_yaml(output)
        meta = output / "meta"
        with (meta / "source_manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "source", "image_name", "label_name", "source_path"])
            writer.writerows(manifest_rows)
        with (meta / "bdd_traffic_light_color.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "image_name", "object_id", "x1", "y1", "x2", "y2", "color"])
            writer.writerows(traffic_light_rows)

    return stats


def write_wts_video_manifest(wts_root: Path, output: Path, dry_run: bool) -> None:
    if not wts_root.exists():
        print(f"[warn] WTS root not found: {wts_root}")
        return

    rows: list[list[str]] = []
    for video in sorted(wts_root.rglob("*.mp4")):
        lower = str(video).lower()
        if "external" in lower and "bdd_pc_5k" in lower:
            group = "external_bdd_pc_5k"
        elif "normal_trimmed" in lower and "overhead_view" in lower:
            group = "wts_normal_overhead"
        elif "normal_trimmed" in lower and "vehicle_view" in lower:
            group = "wts_normal_vehicle"
        elif "overhead_view" in lower:
            group = "wts_event_overhead"
        elif "vehicle_view" in lower:
            group = "wts_event_vehicle"
        else:
            group = "other"
        rows.append([group, str(video), str(video.relative_to(wts_root)), str(video.stat().st_size)])

    print(f"WTS videos inventoried: {len(rows)}")
    if dry_run:
        return
    meta = output / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "wts_video_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "path", "relative_path", "size_bytes"])
        writer.writerows(rows)


def print_stats(stats: dict[str, dict[str, int]]) -> None:
    print("\nSummary")
    print("split,images,plate,person,car,traffic_light,total_boxes,empty,skipped")
    totals = {name: 0 for name in ("images", "plate", "person", "car", "traffic_light", "empty", "skipped")}
    for split in SPLITS:
        row = stats[split]
        total_boxes = row["plate"] + row["person"] + row["car"] + row["traffic_light"]
        print(
            f"{split},{row['images']},{row['plate']},{row['person']},"
            f"{row['car']},{row['traffic_light']},{total_boxes},{row['empty']},{row['skipped']}"
        )
        for key in totals:
            totals[key] += row[key]
    total_boxes = totals["plate"] + totals["person"] + totals["car"] + totals["traffic_light"]
    print(
        f"total,{totals['images']},{totals['plate']},{totals['person']},"
        f"{totals['car']},{totals['traffic_light']},{total_boxes},{totals['empty']},{totals['skipped']}"
    )


def parse_sources(value: str) -> set[str]:
    if value.lower() == "all":
        return set(SOURCE_COUNTS)
    return {part.strip() for part in value.split(",") if part.strip()}


def main() -> None:
    global RAW_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--wts-root", type=Path, default=WTS_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--sources", default="all", help="Comma-separated sources or 'all'.")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-wts-manifest", action="store_true")
    parser.add_argument(
        "--only-wts-manifest",
        action="store_true",
        help="Only write meta/wts_video_manifest.csv; do not collect or convert YOLO images.",
    )
    parser.add_argument("--allow-nonempty", action="store_true", help="Allow writing into a non-empty output directory.")
    args = parser.parse_args()

    RAW_ROOT = args.raw_root
    selected_sources = parse_sources(args.sources)

    print(f"Raw root: {RAW_ROOT}")
    print(f"WTS root: {args.wts_root}")
    print(f"Output:   {args.output}")
    print(f"Sources:  {', '.join(sorted(selected_sources))}")
    print(f"Dry run:  {args.dry_run}")

    if not RAW_ROOT.exists():
        raise SystemExit(f"Raw root not found: {RAW_ROOT}")

    if args.only_wts_manifest:
        write_wts_video_manifest(args.wts_root, args.output, args.dry_run)
        return

    if not args.dry_run and not args.allow_nonempty and not output_is_empty(args.output):
        raise SystemExit(f"Output directory is not empty: {args.output}. Use --allow-nonempty if this is intentional.")

    collectors: dict[str, Callable[[], list[Record]]] = {
        "ccpd2019": collect_ccpd2019,
        "ccpd2020": collect_ccpd2020,
        "crpd_multi": lambda: collect_crpd("crpd_multi", "CRPD_multi"),
        "crpd_double": lambda: collect_crpd("crpd_double", "CRPD_double"),
        "clpd": collect_clpd,
        "bdd100k": collect_bdd100k,
    }
    unknown = selected_sources.difference(collectors)
    if unknown:
        raise SystemExit(f"Unknown sources: {', '.join(sorted(unknown))}")

    records_by_split = {split: [] for split in SPLITS}
    for source, collector in collectors.items():
        if source not in selected_sources:
            continue
        records = collector()
        split_map = split_records(source, records)
        print(
            f"{source}: total={len(records)} "
            f"train={len(split_map['train'])} val={len(split_map['val'])} test={len(split_map['test'])}"
        )
        for split in SPLITS:
            records_by_split[split].extend(split_map[split])

    for split in SPLITS:
        records_by_split[split].sort(key=lambda r: (r.source, r.key))
        print(f"merged {split}: {len(records_by_split[split])}")

    stats = convert(records_by_split, args.output, args.dry_run, args.copy_mode)
    print_stats(stats)

    if args.write_wts_manifest:
        write_wts_video_manifest(args.wts_root, args.output, args.dry_run)


if __name__ == "__main__":
    main()
