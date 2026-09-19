"""Frozen text encoder + shared residual candidate scorer; no output decoding."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sentence_transformers import SentenceTransformer
from torch import nn

from .io import digest, read_records, sha256, write_json
from .schema import Request, query_text

TYPE_INDEX = {"choice": 0, "noul": 1, "score": 2}
INPUT_VERSION = "question-newline-state-v1"


def select_device(device="auto"):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_encoder(config, device="auto", local_path=None):
    model = SentenceTransformer(
        str(local_path) if local_path else config["model_name"],
        revision=None if local_path else config["model_revision"],
        device=select_device(device),
    )
    model.max_seq_length = config["max_seq_length"]
    model.eval()
    return model


class DecisionHead(nn.Module):
    def __init__(self, dim, hidden_dim=128, dropout=0.1, base_scale=20.0):
        super().__init__()
        self.type_embedding = nn.Embedding(3, 8)
        self.residual = nn.Sequential(
            nn.Linear(dim * 4 + 8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.log_scale = nn.Parameter(torch.full((3,), math.log(base_scale)))

    def forward(self, query, options, mask, types, baseline=False):
        cosine = torch.einsum("bd,bkd->bk", query, options)
        if baseline:
            score = cosine * 20.0
        else:
            q = query[:, None, :].expand_as(options)
            type_features = self.type_embedding(types)[:, None, :].expand(-1, options.shape[1], -1)
            features = torch.cat(
                (q, options, torch.abs(q - options), q * options, type_features), dim=-1
            )
            score = cosine * self.log_scale[types].clamp(-2, 5).exp()[:, None]
            score = score + self.residual(features).squeeze(-1)
        return score.masked_fill(~mask, -1e9)


def new_head(config, dim=384):
    return DecisionHead(dim, config["hidden_dim"], config["dropout"], config["base_scale"])


def cache_features(paths, config, cache_dir: Path, device="auto"):
    signature = {
        "files": {str(p): sha256(p) for p in paths},
        "model_name": config["model_name"],
        "model_revision": config["model_revision"],
        "max_seq_length": config["max_seq_length"],
        "input_version": INPUT_VERSION,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = digest(json.dumps(signature, sort_keys=True))[:24]
    path = cache_dir / f"{key}.npz"
    records = [r for p in paths for r in read_records(p)]
    if path.exists():
        return dict(np.load(path, allow_pickle=False)), records, path
    encoder = load_encoder(config, device)
    texts, lookup = [], {}

    def index(text):
        if text not in lookup:
            lookup[text] = len(texts)
            texts.append(text)
        return lookup[text]

    queries, candidates = [], []
    max_options = max(len(r.target) for r in records)
    targets = np.zeros((len(records), max_options), dtype=np.float32)
    option_indices = np.zeros((len(records), max_options), dtype=np.int64)
    masks = np.zeros_like(targets, dtype=bool)
    for i, r in enumerate(records):
        queries.append(index(query_text(r.state, r.question)))
        indices = [index(t) for t in r.question.options()[1]]
        candidates.append(indices)
        targets[i, : len(indices)] = r.target
        option_indices[i, : len(indices)] = indices
        masks[i, : len(indices)] = True
    started = time.perf_counter()
    print(
        f"Encoding {len(texts)} unique texts for {len(records)} records on {encoder.device}",
        flush=True,
    )
    vectors = encoder.encode(
        texts,
        batch_size=config["encoder_batch_size"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    arrays = {
        "embeddings": vectors.astype(np.float32),
        "query_index": np.array(queries, dtype=np.int64),
        "option_index": option_indices,
        "targets": targets,
        "mask": masks,
        "types": np.array([TYPE_INDEX[r.question.type] for r in records], dtype=np.int64),
    }
    np.savez_compressed(path, **arrays)
    write_json(
        path.with_suffix(".json"),
        {
            **signature,
            "records": len(records),
            "unique_texts": len(texts),
            "encoding_seconds": time.perf_counter() - started,
            "device": str(encoder.device),
            "embedding_dimension": int(vectors.shape[1]),
            "cache_sha256": sha256(path),
        },
    )
    print(f"Saved feature cache {path.name}", flush=True)
    return arrays, records, path


def tensors(arrays):
    return {k: torch.from_numpy(v) for k, v in arrays.items()}


def batch(features, indices):
    query = features["embeddings"][features["query_index"][indices]]
    options = features["embeddings"][features["option_index"][indices]]
    return query, options, features["mask"][indices], features["types"][indices]


def predict_logits(head, arrays, batch_size=128, baseline=False):
    head.eval()
    features = tensors(arrays)
    result = []
    with torch.inference_mode():
        for start in range(0, len(arrays["targets"]), batch_size):
            idx = slice(start, start + batch_size)
            result.append(head(*batch(features, idx), baseline=baseline).numpy())
    return np.concatenate(result)


def save_head(path: Path, head, config, **metadata):
    path.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()},
        str(path / "head.safetensors"),
    )
    write_json(
        path / "config.json",
        {
            **config,
            "architecture": "FrozenEncoderResidualScorer",
            "input_version": INPUT_VERSION,
            **metadata,
        },
    )


class OpenJev:
    def __init__(self, checkpoint, device="auto"):
        self.checkpoint = Path(checkpoint)
        self.config = json.loads((self.checkpoint / "config.json").read_text())
        self.head = new_head(self.config, self.config.get("embedding_dimension", 384))
        self.head.load_state_dict(load_file(str(self.checkpoint / "head.safetensors")))
        self.head.eval()
        local = self.checkpoint / "backbone"
        self.encoder = load_encoder(self.config, device, local if local.exists() else None)
        calibration = self.checkpoint / "calibration.json"
        self.calibration = json.loads(calibration.read_text()) if calibration.exists() else None

    def predict(self, state, questions, *, calibrated=True, baseline=False):
        request = Request(state=state, questions=questions)
        started = time.perf_counter()
        names = list(request.questions)
        qs = list(request.questions.values())
        texts = []
        ids = {}

        def index(text):
            if text not in ids:
                ids[text] = len(texts)
                texts.append(text)
            return ids[text]

        query_indices = [index(query_text(request.state, q)) for q in qs]
        option_lists = [[index(x) for x in q.options()[1]] for q in qs]
        # Truncation is explicitly disclosed, never silently represented as a
        # full-context decision. Tokenization is included in latency.
        lengths = [len(self.encoder.tokenizer(x, truncation=False)["input_ids"]) for x in texts]
        truncated = sum(n > self.config["max_seq_length"] for n in lengths)
        vectors = self.encoder.encode(
            texts,
            batch_size=self.config["encoder_batch_size"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = torch.from_numpy(vectors)
        k = max(map(len, option_lists))
        option_tensor = torch.zeros(len(qs), k, vectors.shape[1])
        mask = torch.zeros(len(qs), k, dtype=torch.bool)
        for i, option_ids in enumerate(option_lists):
            option_tensor[i, : len(option_ids)] = vectors[option_ids]
            mask[i, : len(option_ids)] = True
        types = torch.tensor([TYPE_INDEX[q.type] for q in qs])
        with torch.inference_mode():
            logits = self.head(
                vectors[query_indices], option_tensor, mask, types, baseline=baseline
            )
            temperatures = torch.ones(len(qs))
            if calibrated and self.calibration is not None:
                key = "baseline" if baseline else "trained"
                temperatures = torch.tensor(
                    [self.calibration[key][q.type]["temperature"] for q in qs]
                )
            p = torch.softmax(logits / temperatures[:, None], dim=-1).numpy()
        answers = {}
        for i, (name, q) in enumerate(zip(names, qs, strict=True)):
            option_names, descriptions = q.options()
            probabilities = p[i, : len(option_names)]
            if q.type == "noul":
                answer = {"noul": float(probabilities[1])}
            else:
                answer = {
                    "probabilities": dict(
                        zip(option_names, map(float, probabilities), strict=True)
                    ),
                    "confidence": float(probabilities.max()),
                }
                if q.type == "choice":
                    answer["choice"] = option_names[int(probabilities.argmax())]
                else:
                    answer["score"] = float(probabilities @ np.arange(len(probabilities)))
                    answer["legend"] = descriptions
            answers[name] = answer
        return {
            "model": self.config.get("release_name", "OpenJev-v0.1"),
            "answers": answers,
            "calibration": {
                "applied": bool(calibrated and self.calibration is not None),
                "scope": "Temperature fitted on the recorded calibration split; validity on new distributions is not guaranteed.",
                "confidence_definition": "maximum candidate probability",
            },
            "usage": {
                "questions": len(qs),
                "unique_encoded_texts": len(texts),
                "input_tokens": sum(lengths),
                "truncated_inputs": truncated,
                "output_decoding_tokens": 0,
                "latency_seconds": time.perf_counter() - started,
            },
        }
