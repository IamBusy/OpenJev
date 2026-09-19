"""Licensed public images, deterministic original-ID splits, and verified labels."""

import binascii
import hashlib
import io
import json
import struct
import tarfile
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from PIL import Image

from .data import sha256, write_json

PETS_REPO = "timm/oxford-iiit-pet"
PETS_REVISION = "089695c834a7deb60505b7cc506672db1c31a6aa"
CLEVR_URL = "https://thor.robots.ox.ac.uk/clevr4/clevr_4_10k_v1.zip"


def order_key(text):
    return hashlib.sha256(("openjev-vision-v01:" + text).encode()).hexdigest()


def image_bytes(raw):
    with Image.open(io.BytesIO(raw)) as original:
        image = original.convert("RGB")
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()


def write_records(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def prepare_pets(mirror, annotations, output):
    mirror, output = Path(mirror), Path(output)
    if (output / "pets.jsonl").exists():
        raise FileExistsError("Pets records already exist")
    official = {}
    with tarfile.open(annotations, "r:gz") as archive:
        for name, split in (("trainval", "train"), ("test", "test")):
            content = archive.extractfile(f"annotations/{name}.txt").read().decode()
            for line in content.splitlines():
                if not line or line.startswith("#"):
                    continue
                identifier, breed, species, _ = line.split()
                official[identifier] = (split, int(species) - 1)
    rows, classes, selected_ids = [], None, set()
    sources = {}
    for source_split in ("train", "test"):
        path = mirror / "data" / f"{source_split}-00000-of-00001.parquet"
        sources[path.name] = sha256(path)
        parquet = pq.ParquetFile(path)
        metadata = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
        names = metadata["info"]["features"]["label"]["names"]
        if classes is None:
            classes = names
        elif classes != names:
            raise ValueError("Breed taxonomy mismatch")
        items = parquet.read().to_pylist()
        by_label = {i: [] for i in range(len(classes))}
        for item in items:
            identifier = item["image_id"].removesuffix(".jpg")
            expected_breed = identifier.rsplit("_", 1)[0].lower()
            if classes[item["label"]] != expected_breed:
                raise ValueError("Mirror breed label does not match original ID")
            if official.get(identifier) != (source_split, item["label_cat_dog"]):
                raise ValueError("Mirror record differs from official split/species annotation")
            by_label[item["label"]].append(item)
        for label, members in by_label.items():
            members.sort(key=lambda item: order_key(item["image_id"]))
            limits = (
                (("train", 40), ("dev", 10), ("calibration", 10))
                if source_split == "train"
                else (("test", 20),)
            )
            cursor = 0
            for split, size in limits:
                chosen = members[cursor : cursor + size]
                cursor += size
                if len(chosen) != size:
                    raise ValueError("Insufficient images for predeclared per-class selection")
                for item in chosen:
                    identifier = item["image_id"].removesuffix(".jpg")
                    if identifier in selected_ids:
                        raise ValueError("Original image ID crosses splits")
                    selected_ids.add(identifier)
                    raw = item["image"]["bytes"]
                    content = image_bytes(raw)
                    target = output / "images" / "pets" / f"{identifier}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    rows.append(
                        {
                            "source": "oxford-iiit-pet",
                            "source_id": identifier,
                            "group_id": f"pets:{identifier}",
                            "split": split,
                            "official_split": source_split,
                            "image": str(target.relative_to(output)),
                            "labels": {"breed": int(label), "species": int(item["label_cat_dog"])},
                            "source_sha256": hashlib.sha256(raw).hexdigest(),
                            "image_sha256": hashlib.sha256(content).hexdigest(),
                            "license": "cc-by-sa-4.0",
                        }
                    )
    rows.sort(key=lambda row: (row["split"], row["source_id"]))
    write_records(output / "pets.jsonl", rows)
    species = {}
    for row in rows:
        species[row["labels"]["breed"]] = row["labels"]["species"]
    manifest = {
        "source": "Oxford-IIIT Pet",
        "source_page": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "license": "cc-by-sa-4.0",
        "mirror": PETS_REPO,
        "mirror_revision": PETS_REVISION,
        "source_files": sources,
        "official_annotations_sha256": sha256(annotations),
        "records_sha256": sha256(output / "pets.jsonl"),
        "classes": classes,
        "species": [species[i] for i in range(len(classes))],
        "splits": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "dev", "calibration", "test")
        },
        "selection": "SHA256 ordering of original IDs within breed; official test stays test",
        "transform": "RGB conversion, longest side at most 256, Lanczos, lossless PNG",
        "audit": {"group_disjoint": True, "mirror_matches_official_annotations": True},
    }
    write_json(output / "pets_manifest.json", manifest)
    return manifest


class RangeReader(io.RawIOBase):
    """Seekable HTTP ranges; reject a server that silently sends the full archive."""

    def __init__(self, url):
        self.url = url
        self.client = httpx.Client(timeout=90, follow_redirects=True)
        response = self.client.head(url)
        response.raise_for_status()
        self.size = int(response.headers["content-length"])
        self.etag = response.headers.get("etag")
        self.position = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        )
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def range(self, start, length):
        if length <= 0:
            return b""
        end = min(start + length, self.size) - 1
        response = self.client.get(self.url, headers={"Range": f"bytes={start}-{end}"})
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != end - start + 1:
            raise ValueError("Server did not honor the requested byte range")
        if self.etag and response.headers.get("etag") != self.etag:
            raise ValueError("Remote archive changed during download")
        return response.content

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else size
        data = self.range(self.position, size)
        self.position += len(data)
        return data

    def close(self):
        self.client.close()
        super().close()

    def member(self, info):
        header = self.range(info.header_offset, 30)
        fields = struct.unpack("<4s5H3L2H", header)
        if fields[0] != b"PK\x03\x04" or fields[2] & 1:
            raise ValueError("Invalid or encrypted ZIP member")
        offset = info.header_offset + 30 + fields[-2] + fields[-1]
        raw = self.range(offset, info.compress_size)
        if info.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(raw, -15)
        elif info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Unsupported ZIP compression")
        if len(raw) != info.file_size or binascii.crc32(raw) != info.CRC:
            raise ValueError("ZIP member integrity mismatch")
        return raw


def inspect_clevr():
    with RangeReader(CLEVR_URL) as source:
        with zipfile.ZipFile(source) as archive:
            return {
                "size": source.size,
                "etag": source.etag,
                "members": archive.namelist()[:8],
                "json": {
                    name: json.loads(archive.read(name))
                    for name in archive.namelist()
                    if name.endswith(".json")
                },
            }


def prepare_clevr(metadata_path, output, workers=6):
    output = Path(output)
    if (output / "clevr4.jsonl").exists():
        raise FileExistsError("CLEVR-4 records already exist")
    metadata = json.loads(Path(metadata_path).read_text())
    annotations = metadata["json"]["clevr_4_annots.json"]
    colors = sorted({a["color"] for a in annotations.values()})
    shapes = sorted({a["shape"] for a in annotations.values()})
    selected = []
    heldout = []
    for c, color in enumerate(colors):
        for s, shape in enumerate(shapes):
            unseen = (c + s) % 5 == 0
            if unseen:
                heldout.append([color, shape])
            for official, limits in (
                ("train", () if unseen else (("train", 12), ("dev", 2), ("calibration", 2))),
                ("val", (("test", 4),)),
            ):
                members = sorted(
                    [
                        name
                        for name, a in annotations.items()
                        if a["color"] == color and a["shape"] == shape and a["split"] == official
                    ],
                    key=order_key,
                )
                cursor = 0
                for split, size in limits:
                    if len(members[cursor : cursor + size]) != size:
                        raise ValueError("Insufficient source examples for declared split")
                    selected.extend(
                        (name, split, unseen, c, s) for name in members[cursor : cursor + size]
                    )
                    cursor += size
    with RangeReader(CLEVR_URL) as source:
        if source.size != metadata["size"] or source.etag != metadata["etag"]:
            raise ValueError("CLEVR-4 archive identity changed")
        with zipfile.ZipFile(source) as archive:
            info_by_name = {item.filename: item for item in archive.infolist()}

        def download(spec):
            name, split, unseen, c, s = spec
            info = info_by_name[f"images/{name}.png"]
            target = output / "images" / "clevr4" / f"{name}.png"
            receipt = output / "receipts" / f"{name}.json"
            if target.exists() and receipt.exists():
                record = json.loads(receipt.read_text())
                if sha256(target) == record["image_sha256"] and record["source_crc32"] == info.CRC:
                    return record
            raw = source.member(info)
            content = image_bytes(raw)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            ann = annotations[name]
            record = {
                "source": "clevr4",
                "source_id": name,
                "group_id": f"clevr4:{name}",
                "split": split,
                "official_split": ann["split"],
                "image": str(target.relative_to(output)),
                "labels": {"color": c, "shape": s, "joint": c * len(shapes) + s},
                "attributes": {k: ann[k] for k in ("color", "shape", "texture", "count")},
                "unseen_composition": unseen,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "source_crc32": info.CRC,
                "image_sha256": hashlib.sha256(content).hexdigest(),
                "license": "cc-by-4.0",
            }
            write_json(receipt, record)
            return record

        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(download, selected):
                rows.append(row)
                if len(rows) % 100 == 0:
                    print(
                        json.dumps({"clevr4_downloaded": len(rows), "total": len(selected)}),
                        flush=True,
                    )
    rows.sort(key=lambda r: (r["split"], r["source_id"]))
    write_records(output / "clevr4.jsonl", rows)
    manifest = {
        "source": "CLEVR-4 10k v1",
        "source_page": "https://www.robots.ox.ac.uk/~vgg/data/clevr4/",
        "url": CLEVR_URL,
        "license": "cc-by-4.0",
        "archive_size": metadata["size"],
        "archive_etag": metadata["etag"],
        "annotation_sha256": hashlib.sha256(
            json.dumps(annotations, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "records_sha256": sha256(output / "clevr4.jsonl"),
        "colors": colors,
        "shapes": shapes,
        "heldout_pairs": heldout,
        "splits": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "dev", "calibration", "test")
        },
        "selection": "Original-ID hash within color/shape and official split; (color_index+shape_index)%5==0 held out",
        "transform": "RGB, longest side at most 256, Lanczos, PNG",
        "audit": {"group_disjoint": True, "unseen_pairs_absent_from_train_dev_calibration": True},
    }
    write_json(output / "clevr4_manifest.json", manifest)
    return manifest


def prepare_public(root):
    """Download pinned sources and reconstruct the declared public subsets."""
    from huggingface_hub import snapshot_download

    root = Path(root)
    mirror = root / "pets_mirror"
    processed = root / "processed"
    snapshot_download(
        PETS_REPO, repo_type="dataset", revision=PETS_REVISION, local_dir=mirror, token=False
    )
    annotations = root / "raw" / "annotations.tar.gz"
    expected = "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91"
    if not annotations.exists():
        annotations.parent.mkdir(parents=True, exist_ok=True)
        temporary = annotations.with_suffix(".download")
        with httpx.stream(
            "GET",
            "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
            follow_redirects=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for part in response.iter_bytes():
                    stream.write(part)
        if sha256(temporary) != expected:
            raise ValueError("Official annotation hash mismatch")
        temporary.replace(annotations)
    if sha256(annotations) != expected:
        raise ValueError("Official annotation hash mismatch")
    if not (processed / "pets_manifest.json").exists():
        prepare_pets(mirror, annotations, processed)
    metadata = root / "clevr_metadata.json"
    if not metadata.exists():
        write_json(metadata, inspect_clevr())
    if not (processed / "clevr4_manifest.json").exists():
        prepare_clevr(metadata, processed)
    return {
        dataset: json.loads((processed / f"{dataset}_manifest.json").read_text())
        for dataset in ("pets", "clevr4")
    }
