"""Build portable image-Parquet datasets with source-specific licenses."""

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .data import load_split, sha256, write_json
from .public_model import records_for
from .query import answer_questions, evaluation_queries
from .world import describe_world


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def write_parquet(path, rows):
    features = {
        "image": {"_type": "Image"},
        **{key: {"dtype": "string", "_type": "Value"} for key in rows[0] if key != "image"},
    }
    schema = pa.schema(
        [
            (
                key,
                pa.struct([("bytes", pa.binary()), ("path", pa.string())])
                if key == "image"
                else pa.string(),
            )
            for key in rows[0]
        ],
        metadata={b"huggingface": json.dumps({"info": {"features": features}}).encode()},
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", row_group_size=128)
    reloaded = pq.ParquetFile(path)
    if reloaded.metadata.num_rows != len(rows):
        raise ValueError("Parquet round-trip row mismatch")
    return {"rows": len(rows), "sha256": sha256(path), "bytes": Path(path).stat().st_size}


def public_questions(row, manifest):
    labels = row["labels"]
    if row["source"] == "oxford-iiit-pet":
        names = manifest["classes"]
        breed = names[labels["breed"]]
        species = ("cat", "dog")[labels["species"]]
        alternative = names[(labels["breed"] + 7) % len(names)]
        return [
            {
                "id": "breed",
                "type": "choice",
                "instructions": "Which breed is shown?",
                "criteria": {name: {"breed": name} for name in names},
                "target": breed,
            },
            {
                "id": "species",
                "type": "choice",
                "instructions": "Is the pictured pet a cat or a dog?",
                "criteria": {name: {"species": name} for name in ("cat", "dog")},
                "target": species,
            },
            {
                "id": "is_cat",
                "type": "noul",
                "instructions": "Is the pet a cat?",
                "event": {"species": "cat"},
                "target": float(species == "cat"),
            },
            {
                "id": "positive_breed",
                "type": "noul",
                "instructions": f"Is the pet a {breed.replace('_', ' ')}?",
                "event": {"breed": breed},
                "target": 1.0,
            },
            {
                "id": "negative_breed",
                "type": "noul",
                "instructions": f"Is the pet a {alternative.replace('_', ' ')}?",
                "event": {"breed": alternative},
                "target": 0.0,
            },
        ]
    color, shape = row["attributes"]["color"], row["attributes"]["shape"]
    return [
        {
            "id": "color",
            "type": "choice",
            "instructions": "What color are the objects?",
            "criteria": {name: {"color": name} for name in manifest["colors"]},
            "target": color,
        },
        {
            "id": "shape",
            "type": "choice",
            "instructions": "What shape are the objects?",
            "criteria": {name: {"shape": name} for name in manifest["shapes"]},
            "target": shape,
        },
        {
            "id": "red",
            "type": "noul",
            "instructions": "Are the objects red?",
            "event": {"color": "red"},
            "target": float(color == "red"),
        },
        {
            "id": "red_cube",
            "type": "noul",
            "instructions": "Are the objects both red and cube-shaped?",
            "event": {"color": "red", "shape": "cube"},
            "target": float(color == "red" and shape == "cube"),
        },
        {
            "id": "not_red_cube",
            "type": "noul",
            "instructions": "Is it false that the objects are red cubes?",
            "event": {"not": {"color": "red", "shape": "cube"}},
            "target": float(not (color == "red" and shape == "cube")),
        },
    ]


def export_dataset(synthetic_dir, public_dir, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Dataset export directory is not empty")
    output.mkdir(parents=True)
    manifest = {"version": "OpenJev-Vision-Research-v0.1", "files": {}, "configs": {}}
    specs, _ = evaluation_queries()
    questions = {
        f"q{i}": {"type": "noul", "event": expr, "family": family}
        for i, (family, expr) in enumerate(specs)
    }
    synthetic_manifest = json.loads((Path(synthetic_dir) / "manifest.json").read_text())
    write_json(output / "synthetic-manifest.json", synthetic_manifest)
    configs = []
    synthetic_splits = list(synthetic_manifest["splits"])
    for split in synthetic_splits:
        data = load_split(synthetic_dir, split)
        rows = []
        for i in range(len(data["images"])):
            answers = answer_questions(data["targets"][i], questions)
            question_rows = [
                {"id": name, **spec, "target": answers[name]["noul"]}
                for name, spec in questions.items()
            ]
            rows.append(
                {
                    "id": f"synthetic:{split}:{i}",
                    "group_id": str(data["groups"][i]),
                    "image": {"bytes": png_bytes(data["images"][i]), "path": None},
                    "source": "openjev-original-renderer",
                    "source_id": str(data["groups"][i]),
                    "split": split,
                    "license": "apache-2.0",
                    "questions_json": json.dumps(question_rows),
                    "target_json": json.dumps(
                        {
                            "posterior": data["targets"][i].tolist(),
                            "prior": data["priors"][i].tolist(),
                            "latent_world": describe_world(data["truth"][i]),
                        }
                    ),
                    "provenance_json": json.dumps(
                        {
                            "observed": data["observed"][i].tolist(),
                            "visible_objects": int(data["visibility"][i]),
                            "render_seed": int(data["render_seeds"][i]),
                            "family": str(data["families"][i]),
                            "sensor_error": synthetic_manifest["config"]["sensor_error"],
                            "supervision": "exact posterior given observable sensor values; latent truth is separate",
                        }
                    ),
                }
            )
        filename = f"synthetic-{split}.parquet"
        manifest["files"][filename] = write_parquet(output / filename, rows)
        print(json.dumps({"exported": filename, "rows": len(rows)}), flush=True)
    configs.append(("synthetic", synthetic_splits))
    for dataset in ("pets", "clevr4"):
        source_manifest = json.loads((Path(public_dir) / f"{dataset}_manifest.json").read_text())
        write_json(output / f"{dataset}-manifest.json", source_manifest)
        records = records_for(public_dir, dataset)
        seen, source_seen = {}, {}
        for record in records:
            for key, lookup in (("image_sha256", seen), ("source_sha256", source_seen)):
                digest = record[key]
                if digest in lookup and lookup[digest] != record["split"]:
                    raise ValueError("Exact duplicate public image across splits")
                lookup[digest] = record["split"]
        for split in ("train", "dev", "calibration", "test"):
            rows = []
            for record in records:
                if record["split"] != split:
                    continue
                image = Path(public_dir) / record["image"]
                if sha256(image) != record["image_sha256"]:
                    raise ValueError("Public image integrity mismatch")
                rows.append(
                    {
                        "id": record["group_id"],
                        "group_id": record["group_id"],
                        "image": {"bytes": image.read_bytes(), "path": None},
                        "source": record["source"],
                        "source_id": record["source_id"],
                        "split": split,
                        "license": record["license"],
                        "questions_json": json.dumps(public_questions(record, source_manifest)),
                        "target_json": json.dumps(
                            {"labels": record["labels"], "attributes": record.get("attributes", {})}
                        ),
                        "provenance_json": json.dumps(
                            {k: v for k, v in record.items() if k not in {"image", "labels"}}
                        ),
                    }
                )
            filename = f"{dataset}-{split}.parquet"
            manifest["files"][filename] = write_parquet(output / filename, rows)
            print(json.dumps({"exported": filename, "rows": len(rows)}), flush=True)
        configs.append((dataset, ["train", "dev", "calibration", "test"]))
    manifest["configs"] = {name: splits for name, splits in configs}
    write_json(output / "EXPORT_MANIFEST.json", manifest)
    return manifest
