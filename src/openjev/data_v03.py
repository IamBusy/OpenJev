"""Multi-question software states with executable gold rules and fresh holdouts."""

import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .branch_model import SYSTEM
from .data import make_record, normalize, onehot, rank
from .data_v02 import balanced, clean
from .io import digest, read_records, sha256, write_json, write_records
from .qwen import prompt as v02_prompt
from .synthesize import call_teacher, credentials, parse_result

SEED = 43
GENERATOR = (
    "Render the supplied observed facts as a short natural narrative in the requested "
    "language. Include every field precisely. null means unknown and must remain unknown. "
    "Use only generic people/services; no names, phone numbers, email or addresses. "
    "Describe observations and explicit customer requests only. Do not give a verdict, "
    "eligibility, priority, severity, answer, recommendation or probability. "
    'Return JSON {"text":"..."} with 40–100 words or comparable Chinese length.'
)
VERIFY = (
    "Extract the supplied facts from the narrative without guessing. Return every field "
    "in the given schema, preserving its JSON type. Use null for unknown. Do not infer "
    'an action or verdict. Return JSON {"facts":{...}} only.'
)


def world_facts(family, rng):
    if family == "refund":
        return {
            "days_since_purchase": rng.randint(0, 50),
            "return_window_days": rng.choice([7, 14, 21, 30]),
            "opened": rng.choice([True, False, None]),
            "damaged": rng.choice([True, False]),
            "refund_requested": rng.choice([True, False]),
        }
    if family == "incident":
        return {
            "affected_users": rng.choice([0, rng.randint(1, 200)]),
            "workaround": rng.choice([True, False]),
            "service": rng.choice(["login", "payments", "export"]),
        }
    if family == "access":
        return {
            "account_active": rng.choice([True, False]),
            "clearance_rank": rng.randint(0, 3),
            "required_rank": rng.randint(0, 3),
            "resource": "dataset-" + str(rng.randint(1, 50)),
        }
    return {
        "shipped": rng.choice([True, False]),
        "damaged": rng.choice([True, False]),
        "days_late": rng.randint(0, 20),
        "trace_threshold": rng.randint(1, 5),
    }


def template_text(world):
    f = world["facts"]
    family = world["family"]
    zh = world["language"] == "zh"
    if family == "refund":
        opened = {True: "opened", False: "unopened", None: "not stated"}[f["opened"]]
        if zh:
            opened = {True: "已经开封", False: "尚未开封", None: "是否开封未说明"}[f["opened"]]
            return (
                f"购买距今{f['days_since_purchase']}天，退货窗口为{f['return_window_days']}天。"
                f"包装{opened}，商品{'有' if f['damaged'] else '没有'}损坏。"
                + (
                    "客户明确请求退款。"
                    if f["refund_requested"]
                    else "客户仅咨询商品信息，并未请求退款。"
                )
            )
        return (
            f"The purchase was {f['days_since_purchase']} days ago and the return window is "
            f"{f['return_window_days']} days. Package status is {opened}. "
            f"The item is {'damaged' if f['damaged'] else 'undamaged'}. "
            + (
                "The customer explicitly requests a refund."
                if f["refund_requested"]
                else "The customer requests information only and does not request a refund."
            )
        )
    if family == "incident":
        if zh:
            return f"{f['service']}服务出现问题，{f['affected_users']}名用户无法完成工作。{'存在' if f['workaround'] else '不存在'}替代方案。"
        return f"The {f['service']} service has {f['affected_users']} users blocked. A workaround is {'available' if f['workaround'] else 'unavailable'}."
    if family == "access":
        if zh:
            return f"账号{'处于有效状态' if f['account_active'] else '已停用'}，权限等级为{f['clearance_rank']}。资源{f['resource']}要求等级{f['required_rank']}。"
        return f"The account is {'active' if f['account_active'] else 'inactive'}, with clearance rank {f['clearance_rank']}. Resource {f['resource']} requires rank {f['required_rank']}."
    if zh:
        return f"包裹{'已经' if f['shipped'] else '尚未'}发货，{'有' if f['damaged'] else '没有'}损坏，延迟{f['days_late']}天。追踪阈值为{f['trace_threshold']}天。"
    return f"The parcel has {'been shipped' if f['shipped'] else 'not been shipped'}, is {'damaged' if f['damaged'] else 'undamaged'}, and is {f['days_late']} days late. The trace threshold is {f['trace_threshold']} days."


def build_worlds():
    worlds = []
    seen = set()
    for split, count in [("train", 24), ("dev", 6), ("calibration", 6), ("test_id_new", 12)]:
        for family in ["refund", "incident"]:
            rng = random.Random(rank(SEED, split + family))
            for i in range(count):
                while True:
                    facts = world_facts(family, rng)
                    signature = digest(family + json.dumps(facts, sort_keys=True))
                    if signature not in seen:
                        break
                seen.add(signature)
                worlds.append(
                    {
                        "id": signature[:20],
                        "group_id": signature,
                        "split": split,
                        "family": family,
                        "language": "zh" if i % 2 else "en",
                        "facts": facts,
                    }
                )
    for family in ["access", "shipping"]:
        rng = random.Random(rank(SEED, family))
        for i in range(12):
            while True:
                facts = world_facts(family, rng)
                signature = digest(family + json.dumps(facts, sort_keys=True))
                if signature not in seen:
                    break
            seen.add(signature)
            worlds.append(
                {
                    "id": signature[:20],
                    "group_id": signature,
                    "split": "test_family_new",
                    "family": family,
                    "language": "zh" if i % 2 else "en",
                    "facts": facts,
                }
            )
    return worlds


def balance_worlds(worlds):
    """Fill primary outcomes before final training; never use model predictions."""

    def label(w):
        f = w["facts"]
        family = w["family"]
        if family == "refund":
            outcomes = [
                f["days_since_purchase"] <= f["return_window_days"]
                and not opened
                and not f["damaged"]
                for opened in ([False, True] if f["opened"] is None else [f["opened"]])
            ]
            return "yes" if all(outcomes) else "no" if not any(outcomes) else "unknown"
        if family == "incident":
            return "0" if not f["affected_users"] else "1" if f["workaround"] else "2"
        if family == "access":
            return (
                "deny"
                if not f["account_active"]
                else "approval"
                if f["clearance_rank"] < f["required_rank"]
                else "allow"
            )
        return (
            "replace"
            if f["damaged"]
            else "warehouse"
            if not f["shipped"]
            else "trace"
            if f["days_late"] > f["trace_threshold"]
            else "wait"
        )

    expected = {
        "refund": ["yes", "no", "unknown"],
        "incident": ["0", "1", "2"],
        "access": ["allow", "deny", "approval"],
        "shipping": ["replace", "warehouse", "trace", "wait"],
    }

    def context(rng):
        return {
            "channel": rng.choice(["web", "app", "desk"]),
            "site": rng.choice(["quartz", "cedar", "slate", "delta"]),
            "reported_hour": rng.randrange(24),
        }

    # Every scenario, including existing teacher-rendered worlds, receives
    # independent structured context. Context presence is not a class cue.
    for w in worlds:
        w["context"] = context(random.Random(rank(SEED + 7, w["id"])))
    buckets = defaultdict(list)
    for w in worlds:
        buckets[(w["split"], w["family"])].append(w)
    additions = []
    for (split, family), items in buckets.items():
        counts = Counter(label(w) for w in items)
        quota = max(counts.values())
        rng = random.Random(rank(SEED + 9, split + family))
        for target in expected[family]:
            while counts[target] < quota:
                facts = world_facts(family, rng)
                candidate = {"family": family, "facts": facts}
                if label(candidate) != target:
                    continue
                ctx = context(rng)
                identity = digest(
                    f"balanced:{split}:{family}:"
                    + json.dumps([facts, ctx], sort_keys=True)
                    + str(len(additions))
                )
                w = {
                    "id": identity[:20],
                    "group_id": identity,
                    "split": split,
                    "family": family,
                    "language": "zh" if counts[target] % 2 else "en",
                    "facts": facts,
                    "context": ctx,
                    "render_provenance": {
                        "method": "balanced deterministic rule template",
                        "selection": "primary-outcome coverage; no model predictions",
                    },
                }
                w["text"] = template_text(w)
                additions.append(w)
                counts[target] += 1
    return worlds + additions


def render_worlds(root, worlds, replay=False):
    selected = []
    limits = {"train": 24, "dev": 6, "calibration": 6, "test_id_new": 6, "test_family_new": 6}
    for split, n in limits.items():
        selected += sorted(
            [w for w in worlds if w["split"] == split], key=lambda w: rank(SEED, w["id"])
        )[:n]
    assert len(selected) == 48
    key, base = ("", "https://api.deepseek.com") if replay else credentials(root)
    folder = root / "artifacts/v03-render"
    folder.mkdir(exist_ok=True)

    def render(w):
        gen = folder / (w["id"] + ".generation.json")
        review = folder / (w["id"] + ".review.json")
        if gen.exists():
            g = json.loads(gen.read_text())
        else:
            if replay:
                raise FileNotFoundError("Rendering replay requires the saved generation response")
            g = call_teacher(
                key,
                base,
                "deepseek-v4-pro",
                GENERATOR,
                {"facts": w["facts"], "language": w["language"]},
            )
            write_json(gen, g)
        text = parse_result(g).get("text")
        if not isinstance(text, str) or not 20 <= len(text) <= 1600 or not clean(text):
            return w["id"], None, "text_filter"
        if re.search(
            r"\b(ineligible|eligible|severity level|should be approved|should be denied)\b|符合退款|应当批准|严重等级",
            text,
            re.I,
        ):
            return w["id"], None, "verdict_leak"
        schema = {k: type(v).__name__ for k, v in w["facts"].items()}
        if review.exists():
            v = json.loads(review.read_text())
        else:
            if replay:
                raise FileNotFoundError("Rendering replay requires the saved verification response")
            v = call_teacher(key, base, "deepseek-v4-pro", VERIFY, {"text": text, "schema": schema})
            write_json(review, v)
        extracted = parse_result(v).get("facts")
        ok = isinstance(extracted, dict) and set(extracted) == set(w["facts"])
        ok = ok and all(
            type(extracted[k]) is type(value) and extracted[k] == value
            for k, value in w["facts"].items()
        )
        if not ok:
            return w["id"], None, "fact_reconstruction_failed"
        return w["id"], text, {"generation_hash": sha256(gen), "verification_hash": sha256(review)}

    rendered = {}
    rejected = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(render, w) for w in selected]
        for i, job in enumerate(as_completed(jobs), 1):
            identity, text, proof = job.result()
            if text:
                rendered[identity] = (text, proof)
            else:
                rejected[identity] = proof
            if i % 8 == 0:
                print("Verified rendered worlds", i, "/ 48", flush=True)
    for w in worlds:
        w["text"], w["render_provenance"] = rendered.get(
            w["id"], (template_text(w), {"method": "deterministic template"})
        )
    usages = [json.loads(p.read_text())["response"].get("usage", {}) for p in folder.glob("*.json")]
    write_json(
        root / "reports/v03/rendering.json",
        {
            "selected_worlds": 48,
            "accepted": len(rendered),
            "rejected": rejected,
            "recorded_calls": len(usages),
            "maximum_calls": 96,
            "prompt_tokens": sum(x.get("prompt_tokens", 0) for x in usages),
            "completion_tokens": sum(x.get("completion_tokens", 0) for x in usages),
            "gold_source": "executable rules; teacher only renders and reconstructs facts",
            "limitation": "Same teacher family verifies rendering; canonical structured facts are also provided as evidence.",
        },
    )
    return worlds


def world_records(w):
    f = w["facts"]
    zh = w["language"] == "zh"
    family = w["family"]
    records = []
    facts = {k: v for k, v in f.items() if k != "refund_requested"}
    state = {"narrative": w["text"], "facts": facts}
    if "context" in w:
        state["context"] = w["context"]

    def add(name, kind, instructions, criteria, target_index, labels=None):
        if kind == "choice":
            items = list(criteria.items())
            random.Random(rank(SEED, w["id"] + name)).shuffle(items)
            correct = list(criteria)[target_index]
            criteria = dict(items)
            target_index = list(criteria).index(correct)
        length = len(criteria) if kind != "noul" else 2
        r = make_record(
            "software_worlds",
            w["id"],
            w["split"],
            family + "_" + name,
            json.dumps(state, ensure_ascii=False, sort_keys=True),
            {"type": kind, "instructions": instructions, "criteria": criteria},
            onehot(length, target_index),
            list(criteria)[target_index] if isinstance(criteria, dict) else str(target_index),
            family=family,
            language=w["language"],
            canonical_facts=f,
            oracle="software-worlds-v1",
            rendering=w["render_provenance"],
        )
        r.group_id = w["group_id"]
        records.append(r)

    yesno = {"false": "否。" if zh else "No.", "true": "是。" if zh else "Yes."}
    if family == "refund":
        state["policy"] = (
            "退款要求同时满足：购买天数不超过退货窗口、未开封、无损坏。"
            if zh
            else "A refund requires ALL of: days since purchase <= return window; unopened; undamaged."
        )
        possible = [
            (
                f["days_since_purchase"] <= f["return_window_days"]
                and not opened
                and not f["damaged"]
            )
            for opened in ([False, True] if f["opened"] is None else [f["opened"]])
        ]
        result = 0 if all(possible) else 1 if not any(possible) else 2
        add(
            "eligibility",
            "choice",
            "仅根据当前证据和政策，能否确定订单符合退款条件？"
            if zh
            else "Based only on the evidence and policy, can the order be determined eligible for a refund?",
            {
                "yes": "符合条件。" if zh else "It meets every condition.",
                "no": "至少一项条件确定不满足。"
                if zh
                else "At least one condition is definitely not met.",
                "unknown": "信息不足，无法确定。"
                if zh
                else "There is insufficient information to decide.",
            },
            result,
        )
        add(
            "request",
            "choice",
            "客户明确请求什么？" if zh else "What does the customer explicitly request?",
            {
                "refund": "退款。" if zh else "A refund.",
                "information": "仅咨询信息。" if zh else "Information only.",
                "password": "修改密码。" if zh else "A password change.",
                "address": "修改地址。" if zh else "An address change.",
            },
            0 if f["refund_requested"] else 1,
        )
        add(
            "opened",
            "choice",
            "证据如何描述包装开封状态？" if zh else "What is known about the package being opened?",
            {
                "opened": "已开封。" if zh else "Opened.",
                "sealed": "未开封。" if zh else "Unopened.",
                "unknown": "未说明。" if zh else "Not stated.",
            },
            2 if f["opened"] is None else 0 if f["opened"] else 1,
        )
        if f["opened"] is not None:
            failures = (
                int(f["days_since_purchase"] > f["return_window_days"])
                + int(f["opened"])
                + int(f["damaged"])
            )
            add(
                "failures",
                "score",
                "有多少项退款条件不满足？" if zh else "How many refund conditions fail?",
                [
                    "无。" if zh else "None.",
                    "恰好一项。" if zh else "Exactly one.",
                    "至少两项。" if zh else "At least two.",
                ],
                min(failures, 2),
            )
    elif family == "incident":
        teams = ["Blue", "Green", "Amber"]
        random.Random(w["id"]).shuffle(teams)
        state["routing_policy"] = dict(zip(["login", "payments", "export"], teams, strict=True))
        severity = 0 if f["affected_users"] == 0 else 1 if f["workaround"] else 2
        add(
            "severity",
            "score",
            "按照规则判定等级。" if zh else "Assign the level using this rubric.",
            [
                "无用户受阻。" if zh else "No users are blocked.",
                "有用户受阻但存在替代方案。"
                if zh
                else "Users are blocked but a workaround exists.",
                "有用户受阻且无替代方案。" if zh else "Users are blocked and no workaround exists.",
            ],
            severity,
        )
        add(
            "escalate",
            "noul",
            "仅当有用户受阻且无替代方案才升级处理。是否应升级？"
            if zh
            else "Escalate only if users are blocked and no workaround exists. Should this be escalated?",
            yesno,
            int(severity == 2),
        )
        add(
            "owner",
            "choice",
            "根据 routing_policy 应交给哪个团队？"
            if zh
            else "Which team is responsible according to routing_policy?",
            {
                team: f"{team} 团队。" if zh else f"The {team} team."
                for team in ["Blue", "Green", "Amber"]
            },
            ["Blue", "Green", "Amber"].index(state["routing_policy"][f["service"]]),
        )
    elif family == "access":
        state["policy"] = (
            "账号停用则拒绝；有效但权限等级低于要求则需要审批；否则允许。"
            if zh
            else "Inactive accounts are denied. Active accounts below required clearance need approval. Otherwise allow."
        )
        action = (
            1 if not f["account_active"] else 2 if f["clearance_rank"] < f["required_rank"] else 0
        )
        add(
            "action",
            "choice",
            "根据政策应采取什么决定？" if zh else "What decision follows from the policy?",
            {
                "allow": "允许。" if zh else "Allow.",
                "deny": "拒绝。" if zh else "Deny.",
                "approval": "需要审批。" if zh else "Request approval.",
            },
            action,
        )
        add(
            "active",
            "noul",
            "账号当前是否有效？" if zh else "Is the account currently active?",
            yesno,
            int(f["account_active"]),
        )
        gap = max(0, f["required_rank"] - f["clearance_rank"])
        add(
            "gap",
            "score",
            "权限等级与要求的差距是多少？" if zh else "How far is clearance below the requirement?",
            [
                "无差距。" if zh else "No shortfall.",
                "差一级。" if zh else "One rank below.",
                "至少差两级。" if zh else "At least two ranks below.",
            ],
            min(gap, 2),
        )
    else:
        state["policy"] = (
            "优先处理损坏并补发；未损坏但未发货则联系仓库；已发货且延迟超过阈值则追踪；否则等待。"
            if zh
            else "First replace damaged parcels. Otherwise contact the warehouse if unshipped. Otherwise trace if days late exceeds the threshold. Otherwise wait."
        )
        action = (
            0
            if f["damaged"]
            else 1
            if not f["shipped"]
            else 2
            if f["days_late"] > f["trace_threshold"]
            else 3
        )
        add(
            "action",
            "choice",
            "按照政策，下一步是什么？" if zh else "What is the next action under the policy?",
            {
                "replace": "补发。" if zh else "Replace the parcel.",
                "warehouse": "联系仓库。" if zh else "Contact the warehouse.",
                "trace": "追踪包裹。" if zh else "Trace the parcel.",
                "wait": "等待。" if zh else "Wait.",
            },
            action,
        )
        add(
            "shipped",
            "noul",
            "包裹是否已经发货？" if zh else "Has the parcel been shipped?",
            yesno,
            int(f["shipped"]),
        )
        level = 0 if f["days_late"] == 0 else 1 if f["days_late"] <= f["trace_threshold"] else 2
        add(
            "lateness",
            "score",
            "按照延迟程度评分。" if zh else "Score the lateness using this rubric.",
            [
                "未延迟。" if zh else "Not late.",
                "延迟但不超过阈值。" if zh else "Late but not beyond the threshold.",
                "延迟超过阈值。" if zh else "Late beyond the threshold.",
            ],
            level,
        )
    return records


def acquire(root):
    raw = root / "data/v03/raw"
    raw.mkdir(parents=True, exist_ok=True)
    lock = root / "data/v03/SOURCE_LOCK.json"
    previous = json.loads(lock.read_text()) if lock.exists() else []
    expected = {entry["file"]: entry["sha256"] for entry in previous}
    rows = []
    for source, revision, files, license_name in [
        (
            "google/boolq",
            "35b264d03638db9f4ce671b711558bf7ff0f80d5",
            {
                "boolq_train.parquet": "data/train-00000-of-00001.parquet",
                "boolq_validation.parquet": "data/validation-00000-of-00001.parquet",
            },
            "CC-BY-SA-3.0",
        ),
        (
            "allenai/ai2_arc",
            "210d026faf9955653af8916fad021475a3f00453",
            {
                "arc_train.parquet": "ARC-Easy/train-00000-of-00001.parquet",
                "arc_test.parquet": "ARC-Easy/test-00000-of-00001.parquet",
            },
            "CC-BY-SA-4.0",
        ),
    ]:
        for name, path in files.items():
            url = f"https://huggingface.co/datasets/{source}/resolve/{revision}/{path}"
            local = raw / name
            if not local.exists():
                r = httpx.get(url, follow_redirects=True, timeout=90)
                r.raise_for_status()
                local.write_bytes(r.content)
            if name in expected and sha256(local) != expected[name]:
                raise ValueError(f"Downloaded source differs from its lock: {name}")
            rows.append(
                {
                    "file": name,
                    "url": url,
                    "revision": revision,
                    "license": license_name,
                    "sha256": sha256(local),
                }
            )
    write_json(root / "data/v03/SOURCE_LOCK.json", rows)
    return raw


def public_records(raw, source, split):
    result = []
    if source == "boolq":
        name = "boolq_train.parquet" if split == "train" else "boolq_validation.parquet"
        for i, r in enumerate(pq.read_table(raw / name).to_pylist()):
            result.append(
                make_record(
                    "boolq",
                    f"{split}/{i}",
                    split,
                    "boolq",
                    r["passage"],
                    {
                        "type": "noul",
                        "instructions": "Answer only from the passage as written: "
                        + r["question"]
                        + "?",
                        "criteria": {"false": "No.", "true": "Yes."},
                    },
                    onehot(2, int(r["answer"])),
                    str(r["answer"]),
                    original_split=split,
                )
            )
    else:
        name = "arc_train.parquet" if split == "train" else "arc_test.parquet"
        for r in pq.read_table(raw / name).to_pylist():
            c = r["choices"]
            texts, keys = c["text"], c["label"]
            if len(texts) != 4 or len(set(texts)) != 4:
                continue
            order = list(range(4))
            random.Random(rank(SEED, r["id"])).shuffle(order)
            correct = keys.index(r["answerKey"])
            result.append(
                make_record(
                    "arc_easy",
                    r["id"],
                    split,
                    "arc_easy",
                    r["question"],
                    {
                        "type": "choice",
                        "instructions": "Choose the correct answer to the question.",
                        "criteria": {keys[i]: texts[i] for i in order},
                    },
                    onehot(4, order.index(correct)),
                    keys[correct],
                    original_split=split,
                )
            )
    return result


def prepare_v03(root, replay=False, live_rendering=False):
    out = root / "data/v03/processed"
    if (out / "manifest.json").exists():
        raise FileExistsError("V0.3 data already frozen")
    raw = acquire(root)
    tok = AutoTokenizer.from_pretrained(root / "artifacts/base/qwen3-0.6b")
    cfg = json.loads((root / "configs/branch-v03.json").read_text())
    marker = "STATE_PLACEHOLDER"
    head, tail = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ).split(marker)

    def eligible(r):
        if not clean(r.state) and r.source != "software_worlds":
            return False
        prefix = len(tok.encode(head + "State:\n" + r.state, add_special_tokens=False))
        suffixes = [
            len(
                tok.encode(
                    "\n\nQuestion:\n"
                    + r.question.instructions
                    + "\n\nCandidate answer:\n"
                    + c
                    + "\n\nIs this candidate correct? Answer Yes or No."
                    + tail,
                    add_special_tokens=False,
                )
            )
            for c in r.question.options()[1]
        ]
        old_tokens = len(tok.encode(v02_prompt(tok, r.state, r.question), add_special_tokens=False))
        if (
            prefix > cfg["max_prefix_tokens"]
            or max(suffixes) > cfg["max_branch_tokens"]
            or old_tokens > 768
        ):
            return False
        r.provenance.update(
            prefix_tokens=prefix, max_branch_tokens=max(suffixes), v02_input_tokens=old_tokens
        )
        return True

    old = read_records(root / "data/v02/processed/test.jsonl")
    used_old = {normalize(r.state) for r in old}
    outputs = {"train": [], "dev": [], "calibration": [], "test_fresh": []}
    for source in ["boolq", "arc"]:
        test = public_records(raw, source, "validation" if source == "boolq" else "test")
        all_test_groups = {r.group_id for r in test}
        fresh = [r for r in test if normalize(r.state) not in used_old and eligible(r)]
        picked = balanced(fresh, 64)
        for r in picked:
            r.split = "test_fresh"
        outputs["test_fresh"] += picked
        candidates = [
            r
            for r in public_records(raw, source, "train")
            if r.group_id not in all_test_groups and eligible(r)
        ]
        # Assign whole source states before selecting records.
        unique = {}
        for r in sorted(candidates, key=lambda r: rank(SEED, r.id)):
            unique.setdefault(r.group_id, r)
        rows = list(unique.values())
        for split, n in [("train", 128), ("dev", 32), ("calibration", 32)]:
            selected = balanced(rows, n)
            used = {r.group_id for r in selected}
            rows = [r for r in rows if r.group_id not in used]
            for r in selected:
                r.split = split
            outputs[split] += selected
    for split, n in [("train", 64), ("dev", 16), ("calibration", 16)]:
        previous = read_records(root / f"data/v02/processed/{split}.jsonl")
        for task in ["banking_choice", "sentiment_score"]:
            outputs[split] += balanced([r for r in previous if r.task == task and eligible(r)], n)
    if replay or live_rendering:
        worlds = build_worlds()
        write_json(root / "data/v03/WORLDS_PLAN.json", worlds)
        worlds = render_worlds(root, worlds, replay=replay)
        worlds = balance_worlds(worlds)
        write_json(root / "data/v03/WORLD_BALANCE_PLAN.json", worlds)
    else:
        # Public, reviewed facts/narratives are sufficient to reconstruct the
        # exact release corpus. Raw provider requests are deliberately private.
        path = root / "data/v03/WORLD_BALANCE_PLAN.json"
        frozen = json.loads((root / "reports/v03/data_manifest.json").read_text())
        if sha256(path) != frozen["balanced_worlds_sha256"]:
            raise ValueError("Frozen world snapshot differs from the release manifest")
        worlds = json.loads(path.read_text())
    for w in worlds:
        records = world_records(w)
        if not all(eligible(r) for r in records):
            # Length rejection is decided for a complete state group.
            continue
        split = w["split"] if w["split"] in outputs else "test_fresh"
        for r in records:
            r.split = split
        outputs[split] += records
    groups = defaultdict(set)
    summaries = {}
    for split, records in outputs.items():
        for r in records:
            groups[r.group_id].add(split)
        records.sort(key=lambda r: rank(SEED, r.id))
        write_records(out / f"{split}.jsonl", records)
        summaries[split] = {
            "records": len(records),
            "state_groups": len({r.group_id for r in records}),
            "tasks": dict(Counter(r.task for r in records)),
            "sha256": sha256(out / f"{split}.jsonl"),
        }
    if any(len(s) > 1 for s in groups.values()):
        raise AssertionError("State group split leakage")
    if any(normalize(r.state) in used_old for r in outputs["test_fresh"]):
        raise AssertionError("Fresh test overlaps the old regression set")
    # Bounded paired reference, fixed before model predictions.
    bytask = defaultdict(list)
    for r in outputs["test_fresh"]:
        bytask[r.task].append(r)
    reference = []
    for values in bytask.values():
        reference += sorted(values, key=lambda r: rank(SEED + 1, r.id))[:4]
    reference = sorted(reference, key=lambda r: rank(SEED + 2, r.id))[:64]
    write_records(out / "reference.jsonl", reference)
    summaries["reference"] = {
        "records": len(reference),
        "sha256": sha256(out / "reference.jsonl"),
        "relation": "intentional subset of fresh test",
    }
    manifest = {
        "version": "OpenJev-v0.3",
        "seed": SEED,
        "splits": summaries,
        "source_lock_sha256": sha256(root / "data/v03/SOURCE_LOCK.json"),
        "worlds_plan_sha256": sha256(root / "data/v03/WORLDS_PLAN.json"),
        "balanced_worlds_sha256": sha256(root / "data/v03/WORLD_BALANCE_PLAN.json"),
        "balance_protocol": "Supplement primary outcome classes using exact-rule templates before final training; independent context on all worlds",
        "old_regression_sha256": sha256(root / "data/v02/processed/test.jsonl"),
        "new_family_holdouts": ["access", "shipping"],
        "unknown_policy": "Explicit insufficient-information choice; no fabricated 0.5 targets",
        "gold_sources": "public annotations and executable rules, never teacher confidence",
        "limits": "Bounded pilot; public pretraining contamination and template specificity remain.",
    }
    write_json(out / "manifest.json", manifest)
    write_json(root / "reports/v03/data_manifest.json", manifest)
    return manifest
