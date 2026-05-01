from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from snorkel.labeling.model import LabelModel

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "outputs" / "llm_baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Point to the MBFC-linkable benchmark CSV (not included in this repo).
DATA_PATH = Path(
    os.environ.get(
        "MBFC_DATA_PATH",
        str(REPO_ROOT / "data" / "messages_with_risk_label_urls_removed_nonempty_no_linkurl_evidence.csv"),
    )
)

MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

RANDOM_STATES = list(range(10))
THRESH_GRID = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]

URL_STRIP_PATTERN = re.compile(
    r"(https?://|http://|www\.[^\s]*|t\.me/[^\s]*)", flags=re.IGNORECASE
)


def append_progress(line: str) -> None:
    with (RESULTS_DIR / "progress.log").open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def load_mbfc_linkable_subset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if "source" not in df.columns:
        df = df.rename(columns={df.columns[0]: "source"})

    df["message"] = (
        df["message"]
        .astype(str)
        .str.replace(URL_STRIP_PATTERN, " ", regex=True)
        .str.strip()
    )
    df = df[df["message"] != ""].copy()
    df = df.dropna(subset=["risk_label"]).copy()
    df["y"] = df["risk_label"].astype(int)
    df = df.dropna(subset=["normalized_domain"]).copy()
    df["message_id"] = df.index.astype(int)

    expected_n = os.environ.get("EXPECTED_N")
    if expected_n is not None and expected_n.strip():
        exp = int(expected_n)
        if len(df) != exp:
            raise RuntimeError(f"Unexpected N={len(df)} after filtering (expected {exp}).")
    return df


def iter_domain_splits(
    df: pd.DataFrame, seeds: Iterable[int] = RANDOM_STATES
) -> Iterable[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    for seed in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        trainval_idx, test_idx = next(
            gss.split(df, df["y"], groups=df["normalized_domain"])
        )
        df_trainval = df.iloc[trainval_idx].copy()
        df_test = df.iloc[test_idx].copy()
        df_train, df_val = train_test_split(
            df_trainval,
            test_size=0.125,
            random_state=100 + seed,
            stratify=df_trainval["y"],
        )
        yield seed, df_train, df_val, df_test


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(y_proba, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = indices == b
        if not np.any(mask):
            continue
        p_bin = y_proba[mask].mean()
        y_bin = y_true[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(p_bin - y_bin)
    return float(ece)


def sweep_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for t in THRESH_GRID:
        pred = (y_proba >= t).astype(int)
        cand = {
            "threshold": float(t),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
            "macro_recall": float(recall_score(y_true, pred, average="macro")),
            "recall_pos": float(recall_score(y_true, pred, pos_label=1)),
        }
        if best is None or cand["macro_f1"] > best["macro_f1"]:
            best = cand
    if best is None:
        raise RuntimeError("Threshold sweep failed.")
    return best


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "macro_recall": float(recall_score(y_true, pred, average="macro")),
        "recall_pos": float(recall_score(y_true, pred, pos_label=1)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "ece": float(expected_calibration_error(y_true, y_proba, n_bins=10)),
    }


def iter_jsonl_dir(dir_path: Path) -> Iterable[dict[str, Any]]:
    if not dir_path.exists():
        return
    for path in sorted(dir_path.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def done_ids_from_jsonl(dir_path: Path) -> set[int]:
    done: set[int] = set()
    for row in iter_jsonl_dir(dir_path):
        mid = row.get("message_id")
        if isinstance(mid, int) and row.get("error") is None:
            done.add(mid)
    return done


def done_pairs_from_jsonl(dir_path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    for row in iter_jsonl_dir(dir_path):
        mid = row.get("message_id")
        sig = row.get("signal_key")
        if isinstance(mid, int) and isinstance(sig, str) and row.get("error") is None:
            done.add((mid, sig))
    return done


@dataclass
class ShardedJsonlWriter:
    out_dir: Path
    prefix: str
    shard_size: int = 1000

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.out_dir.glob(f"{self.prefix}_shard_*.jsonl"))
        if existing:
            indices: list[int] = []
            for p in existing:
                m = re.search(r"_shard_(\d+)\.jsonl$", p.name)
                if m:
                    indices.append(int(m.group(1)))
            self._shard_index = (max(indices) + 1) if indices else 0
        else:
            self._shard_index = 0
        self._rows_in_shard = 0
        self._fp = None
        self._open_next()

    def _open_next(self) -> None:
        if self._fp is not None:
            self._fp.close()
        path = self.out_dir / f"{self.prefix}_shard_{self._shard_index:05d}.jsonl"
        self._fp = path.open("a", encoding="utf-8")
        self._shard_index += 1
        self._rows_in_shard = 0

    def write(self, obj: dict[str, Any]) -> None:
        if self._fp is None:
            raise RuntimeError("Writer not initialized")
        self._fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fp.flush()
        self._rows_in_shard += 1
        if self._rows_in_shard >= self.shard_size:
            self._open_next()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None


def require_openai_key() -> None:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    # OpenAI keys are expected to start with "sk-". This avoids burning time on retries
    # with obviously invalid clipboard/environment contents.
    if not key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY does not look like an OpenAI key (expected prefix 'sk-').")


def _safe_json_loads(s: str) -> dict[str, Any]:
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object.")
    return obj


async def call_with_retries(
    coro_factory,
    *,
    max_attempts: int = 8,
    base_delay_s: float = 1.0,
    max_delay_s: float = 60.0,
) -> Any:
    delay = base_delay_s
    last_exc: Exception | None = None
    for _ in range(max_attempts):
        try:
            return await coro_factory()
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            # Do not retry non-transient 4xx errors.
            if isinstance(
                e,
                (
                    AuthenticationError,
                    BadRequestError,
                    PermissionDeniedError,
                    NotFoundError,
                    ConflictError,
                    UnprocessableEntityError,
                ),
            ):
                raise
            last_exc = e
            await asyncio.sleep(delay)
            delay = min(max_delay_s, delay * 2)
    if last_exc is None:
        raise RuntimeError("Unknown OpenAI failure.")
    raise last_exc


ZERO_SHOT_SYSTEM_PROMPT = """You are a careful annotator.

Task: Given the text of a Telegram message with URLs removed/masked, estimate the probability that the message comes from a higher-risk / lower-credibility information source, as defined by an objective MediaBiasFactCheck (MBFC)-derived proxy label.

Important constraints:
- Do NOT browse the web or use external knowledge.
- Use ONLY the provided text.
- URLs are removed/masked; do not infer domain reputation.
- Do NOT judge factual truth; predict the dataset label (high-risk vs low-risk source) based on stylistic/rhetorical cues.

Output MUST be strict JSON and nothing else.
""".strip()

ZERO_SHOT_USER_TEMPLATE = """Telegram message (URLs removed):
{message}

Return strict JSON with keys:
- p_high_risk: number in [0,1]
- y_pred: 0 or 1 (use 1 if p_high_risk >= 0.5 else 0)
""".strip()


PASTEL_SYSTEM_PROMPT = """You are a helpful and unbiased credibility-signal extraction assistant.

You will be given the text of a Telegram message (URLs removed/masked). Answer credibility-signal questions.

Answer rules:
- Answer with exactly one of: Yes, No, Unsure.
- You are expected to answer Yes/No, but you may answer Unsure if you do not have enough information or context.
- Do NOT browse the web or use external knowledge.
- Use ONLY the provided message text.
- If a question mentions a \"title\", interpret it as the first line of the message (up to the first newline), if any.

Output MUST be strict JSON and nothing else.
""".strip()

# Questions sourced from the official PASTEL repo (external/PASTEL/data/signals.csv),
# minimally adapted from \"article\"→\"message\".
PASTEL_SIGNALS: list[dict[str, str]] = [
    {
        "key": "Evidence",
        "question": "Does the message fail to present any supporting evidence or arguments to substantiate its claims?",
    },
    {
        "key": "Explicitly Unverified Claims",
        "question": "Does the message contain claims that are explicitly unverified?",
    },
    {
        "key": "Clickbait",
        "question": "Does the message contain any form of 'clickbait' in the title?",
    },
    {
        "key": "Misleading about content",
        "question": "Does the title of the message fail to accurately reflect its content?",
    },
    {
        "key": "Personal Perspective",
        "question": "Does the message express the author’s opinion on the subject?",
    },
    {
        "key": "Emotional Valence",
        "question": "Does the message lack a neutral tone?",
    },
    {
        "key": "Polarising Language",
        "question": "Does the message make use of polarising terms or make divisions into sharply contrasting groups or sets of opinions or beliefs?",
    },
    {
        "key": "Call to Action",
        "question": "Does the message contain language that can be understood as a call to action, requesting readers to follow-through with a particular task or tells readers what to do?",
    },
    {
        "key": "Bias",
        "question": "Does the message contain explicit or implicit biases?",
    },
    {
        "key": "Inference",
        "question": "Does the message make claims about correlation and causation?",
    },
    {
        "key": "Expert Citation",
        "question": "Does the message lack citations of experts in the subject?",
    },
    {
        "key": "Document Citation",
        "question": "Does the message lack citations of studies or documents to support its claims?",
    },
    {
        "key": "Source Credibility",
        "question": "Does the message cite sources that are not considered credible?",
    },
    {
        "key": "Incorrect Spelling",
        "question": "Does the message have significant misspellings and/or grammatical errors?",
    },
    {
        "key": "Informal Tone",
        "question": "Does the message make use of all caps or consecutive exclamation or question marks?",
    },
    {
        "key": "Incivility",
        "question": "Does the message make use of stereotypes and generalisations of groups of people?",
    },
    {
        "key": "Impoliteness",
        "question": "Does the message contain insults, name-calling or profanity?",
    },
    {
        "key": "Sensationalism",
        "question": "Does the message make use of sensationalist claims?",
    },
    {
        "key": "Reported by Other Sources",
        "question": "Was the story on this message not reported by other reputable media outlets?",
    },
]


def pastel_questions_block() -> str:
    lines: list[str] = []
    for i, s in enumerate(PASTEL_SIGNALS, start=1):
        lines.append(f"{i}. {s['key']}: {s['question']}")
    return "\n".join(lines)


PASTEL_USER_TEMPLATE = """Telegram message (URLs removed):
{message}

Answer each question independently.
Questions:
{questions}

Return strict JSON with exactly these keys (string values must be: Yes, No, or Unsure):
{keys_block}
""".strip()

PASTEL_SINGLE_USER_TEMPLATE = """Telegram message (URLs removed):
{message}

Question:
{question} (Yes/No/Unsure)

Return strict JSON: {{"answer": "Yes" | "No" | "Unsure"}}
""".strip()


async def call_chat_json(
    client: AsyncOpenAI,
    *,
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int,
) -> tuple[str, dict[str, int] | None]:
    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        seed=0,
        response_format={"type": "json_object"},
        max_completion_tokens=max_completion_tokens,
        timeout=60.0,
    )
    content = resp.choices[0].message.content or ""
    usage = None
    if getattr(resp, "usage", None) is not None:
        usage = {
            "prompt_tokens": int(resp.usage.prompt_tokens or 0),
            "completion_tokens": int(resp.usage.completion_tokens or 0),
            "total_tokens": int(resp.usage.total_tokens or 0),
        }
    return content, usage


async def run_zero_shot_inference(
    df: pd.DataFrame,
    *,
    concurrency: int,
    shard_size: int,
    log_every: int,
    max_messages: int | None,
) -> Path:
    require_openai_key()
    out_dir = RESULTS_DIR / "zero_shot_raw"
    done_ids = done_ids_from_jsonl(out_dir)
    remaining = df[~df["message_id"].isin(done_ids)].copy()
    if max_messages is not None:
        remaining = remaining.head(max_messages)

    append_progress(f"[zero_shot] done={len(done_ids)} remaining={len(remaining)}")
    if len(remaining) == 0:
        return out_dir

    writer = ShardedJsonlWriter(out_dir, prefix="zero_shot", shard_size=shard_size)
    lock = asyncio.Lock()
    client = AsyncOpenAI()
    start_time = time.time()
    processed = 0

    async def handle_row(message_id: int, message: str) -> dict[str, Any]:
        user_prompt = ZERO_SHOT_USER_TEMPLATE.format(message=message)

        raw, usage = await call_with_retries(
            lambda: call_chat_json(
                client,
                system_prompt=ZERO_SHOT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_completion_tokens=128,
            )
        )
        parsed = _safe_json_loads(raw)
        p = float(parsed.get("p_high_risk"))
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p_high_risk out of range: {p}")
        y_pred = int(parsed.get("y_pred"))
        if y_pred not in (0, 1):
            raise ValueError(f"y_pred invalid: {y_pred}")
        return {"p_high_risk": p, "y_pred": y_pred, "usage": usage, "raw": raw}

    async def worker(queue: asyncio.Queue[tuple[int, str] | None]) -> None:
        nonlocal processed
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                message_id, message = item
                t0 = time.time()
                try:
                    out = await handle_row(message_id, message)
                    rec = {
                        "message_id": int(message_id),
                        "model": MODEL_NAME,
                        "p_high_risk": out["p_high_risk"],
                        "y_pred": out["y_pred"],
                        "usage": out["usage"],
                        "raw": out["raw"],
                        "elapsed_s": float(time.time() - t0),
                        "error": None,
                    }
                except Exception as e:
                    rec = {"message_id": int(message_id), "model": MODEL_NAME, "error": repr(e)}
                async with lock:
                    writer.write(rec)
                    processed += 1
                    if processed % log_every == 0:
                        elapsed = time.time() - start_time
                        rate = processed / max(elapsed, 1e-6)
                        append_progress(
                            f"[zero_shot] processed={processed} elapsed_s={elapsed:.1f} rate_msg_per_s={rate:.2f}"
                        )
            finally:
                queue.task_done()

    queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()
    for row in remaining[["message_id", "message"]].itertuples(index=False):
        queue.put_nowait((int(row.message_id), str(row.message)))
    for _ in range(concurrency):
        queue.put_nowait(None)

    workers = [asyncio.create_task(worker(queue)) for _ in range(concurrency)]
    await queue.join()
    for w in workers:
        await w
    writer.close()
    await client.close()
    append_progress(f"[zero_shot] completed processed={processed}")
    return out_dir


def compile_zero_shot_outputs(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in iter_jsonl_dir(raw_dir):
        if rec.get("error") is not None:
            continue
        if "message_id" not in rec or "p_high_risk" not in rec:
            continue
        usage = rec.get("usage") or {}
        rows.append(
            {
                "message_id": int(rec["message_id"]),
                "p_high_risk": float(rec["p_high_risk"]),
                "y_pred_llm": int(rec.get("y_pred", 0)),
                "usage_prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "usage_completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "usage_total_tokens": int(usage.get("total_tokens", 0) or 0),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("message_id").drop_duplicates("message_id", keep="last")
    return df


async def run_pastel_signals_batch(
    df: pd.DataFrame,
    *,
    concurrency: int,
    shard_size: int,
    log_every: int,
) -> Path:
    require_openai_key()
    out_dir = RESULTS_DIR / "pastel_signals_raw_batch"
    done_ids = done_ids_from_jsonl(out_dir)
    remaining = df[~df["message_id"].isin(done_ids)].copy()

    append_progress(f"[pastel_batch] done={len(done_ids)} remaining={len(remaining)}")
    if len(remaining) == 0:
        return out_dir

    writer = ShardedJsonlWriter(out_dir, prefix="pastel_batch", shard_size=shard_size)
    lock = asyncio.Lock()
    client = AsyncOpenAI()
    start_time = time.time()
    processed = 0

    keys = [s["key"] for s in PASTEL_SIGNALS]
    keys_block = "\n".join([f"- {k}" for k in keys])
    questions = pastel_questions_block()

    async def handle_row(message_id: int, message: str) -> dict[str, Any]:
        user_prompt = PASTEL_USER_TEMPLATE.format(
            message=message, questions=questions, keys_block=keys_block
        )
        raw, usage = await call_with_retries(
            lambda: call_chat_json(
                client,
                system_prompt=PASTEL_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_completion_tokens=512,
            )
        )
        parsed = _safe_json_loads(raw)
        signals: dict[str, str] = {}
        for k in keys:
            v = parsed.get(k)
            if not isinstance(v, str):
                raise ValueError(f"Missing/non-string for key {k}")
            v = v.strip()
            if v not in {"Yes", "No", "Unsure"}:
                raise ValueError(f"Invalid value for {k}: {v}")
            signals[k] = v
        return {"signals": signals, "usage": usage, "raw": raw}

    async def worker(queue: asyncio.Queue[tuple[int, str] | None]) -> None:
        nonlocal processed
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                message_id, message = item
                t0 = time.time()
                try:
                    out = await handle_row(message_id, message)
                    rec = {
                        "message_id": int(message_id),
                        "model": MODEL_NAME,
                        "signals": out["signals"],
                        "usage": out["usage"],
                        "raw": out["raw"],
                        "elapsed_s": float(time.time() - t0),
                        "error": None,
                    }
                except Exception as e:
                    rec = {"message_id": int(message_id), "model": MODEL_NAME, "error": repr(e)}
                async with lock:
                    writer.write(rec)
                    processed += 1
                    if processed % log_every == 0:
                        elapsed = time.time() - start_time
                        rate = processed / max(elapsed, 1e-6)
                        append_progress(
                            f"[pastel_batch] processed={processed} elapsed_s={elapsed:.1f} rate_msg_per_s={rate:.2f}"
                        )
            finally:
                queue.task_done()

    queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()
    for row in remaining[["message_id", "message"]].itertuples(index=False):
        queue.put_nowait((int(row.message_id), str(row.message)))
    for _ in range(concurrency):
        queue.put_nowait(None)

    workers = [asyncio.create_task(worker(queue)) for _ in range(concurrency)]
    await queue.join()
    for w in workers:
        await w
    writer.close()
    await client.close()
    append_progress(f"[pastel_batch] completed processed={processed}")
    return out_dir


async def run_pastel_signals_single(
    df: pd.DataFrame,
    *,
    concurrency: int,
    shard_size: int,
    log_every: int,
) -> Path:
    require_openai_key()
    out_dir = RESULTS_DIR / "pastel_signals_raw_single"
    done_pairs = done_pairs_from_jsonl(out_dir)

    keys = [s["key"] for s in PASTEL_SIGNALS]
    questions = {s["key"]: s["question"] for s in PASTEL_SIGNALS}

    tasks: list[tuple[int, str, str]] = []
    for row in df[["message_id", "message"]].itertuples(index=False):
        mid = int(row.message_id)
        msg = str(row.message)
        for k in keys:
            if (mid, k) in done_pairs:
                continue
            tasks.append((mid, msg, k))

    append_progress(
        f"[pastel_single] done_pairs={len(done_pairs)} remaining_calls={len(tasks)}"
    )
    if len(tasks) == 0:
        return out_dir

    writer = ShardedJsonlWriter(out_dir, prefix="pastel_single", shard_size=shard_size)
    lock = asyncio.Lock()
    client = AsyncOpenAI()
    start_time = time.time()
    processed = 0

    async def handle_call(message_id: int, message: str, signal_key: str) -> dict[str, Any]:
        user_prompt = PASTEL_SINGLE_USER_TEMPLATE.format(
            message=message,
            question=questions[signal_key],
        )
        raw, usage = await call_with_retries(
            lambda: call_chat_json(
                client,
                system_prompt=PASTEL_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_completion_tokens=64,
            )
        )
        parsed = _safe_json_loads(raw)
        ans = parsed.get("answer")
        if not isinstance(ans, str):
            raise ValueError("Missing/non-string answer")
        ans = ans.strip()
        if ans not in {"Yes", "No", "Unsure"}:
            raise ValueError(f"Invalid answer: {ans}")
        return {"answer": ans, "usage": usage, "raw": raw}

    async def worker(queue: asyncio.Queue[tuple[int, str, str] | None]) -> None:
        nonlocal processed
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                message_id, message, signal_key = item
                t0 = time.time()
                try:
                    out = await handle_call(message_id, message, signal_key)
                    rec = {
                        "message_id": int(message_id),
                        "signal_key": str(signal_key),
                        "model": MODEL_NAME,
                        "answer": out["answer"],
                        "usage": out["usage"],
                        "raw": out["raw"],
                        "elapsed_s": float(time.time() - t0),
                        "error": None,
                    }
                except Exception as e:
                    rec = {
                        "message_id": int(message_id),
                        "signal_key": str(signal_key),
                        "model": MODEL_NAME,
                        "error": repr(e),
                    }
                async with lock:
                    writer.write(rec)
                    processed += 1
                    if processed % log_every == 0:
                        elapsed = time.time() - start_time
                        rate = processed / max(elapsed, 1e-6)
                        append_progress(
                            f"[pastel_single] processed_calls={processed} elapsed_s={elapsed:.1f} rate_calls_per_s={rate:.2f}"
                        )
            finally:
                queue.task_done()

    queue: asyncio.Queue[tuple[int, str, str] | None] = asyncio.Queue()
    for t in tasks:
        queue.put_nowait(t)
    for _ in range(concurrency):
        queue.put_nowait(None)

    workers = [asyncio.create_task(worker(queue)) for _ in range(concurrency)]
    await queue.join()
    for w in workers:
        await w
    writer.close()
    await client.close()
    append_progress(f"[pastel_single] completed processed_calls={processed}")
    return out_dir


def compile_pastel_signals_batch(raw_dir: Path) -> pd.DataFrame:
    keys = [s["key"] for s in PASTEL_SIGNALS]
    rows: list[dict[str, Any]] = []
    for rec in iter_jsonl_dir(raw_dir):
        if rec.get("error") is not None:
            continue
        sigs = rec.get("signals")
        if not isinstance(sigs, dict):
            continue
        usage = rec.get("usage") or {}
        row: dict[str, Any] = {
            "message_id": int(rec["message_id"]),
            "usage_prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "usage_completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "usage_total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        ok = True
        for k in keys:
            v = sigs.get(k)
            if v not in {"Yes", "No", "Unsure"}:
                ok = False
                break
            row[k] = v
            row[k + "_code"] = 1 if v == "Yes" else 0 if v == "No" else -1
        if ok:
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values("message_id").drop_duplicates("message_id", keep="last")
    return df


def compile_pastel_signals_single(raw_dir: Path) -> pd.DataFrame:
    keys = [s["key"] for s in PASTEL_SIGNALS]
    # (message_id, signal_key) -> latest record
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for rec in iter_jsonl_dir(raw_dir):
        if rec.get("error") is not None:
            continue
        mid = rec.get("message_id")
        sig = rec.get("signal_key")
        ans = rec.get("answer")
        if not isinstance(mid, int) or not isinstance(sig, str) or not isinstance(ans, str):
            continue
        ans = ans.strip()
        if ans not in {"Yes", "No", "Unsure"}:
            continue
        latest[(mid, sig)] = rec

    # Aggregate per message_id
    usage_agg: dict[int, dict[str, int]] = {}
    answers: dict[int, dict[str, str]] = {}
    for (mid, sig), rec in latest.items():
        answers.setdefault(mid, {})[sig] = str(rec["answer"]).strip()
        usage = rec.get("usage") or {}
        agg = usage_agg.setdefault(mid, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        agg["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        agg["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        agg["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

    rows: list[dict[str, Any]] = []
    for mid, sig_map in answers.items():
        if not all(k in sig_map for k in keys):
            continue
        row: dict[str, Any] = {
            "message_id": int(mid),
            "usage_prompt_tokens": int(usage_agg.get(mid, {}).get("prompt_tokens", 0)),
            "usage_completion_tokens": int(usage_agg.get(mid, {}).get("completion_tokens", 0)),
            "usage_total_tokens": int(usage_agg.get(mid, {}).get("total_tokens", 0)),
        }
        for k in keys:
            v = sig_map[k]
            row[k] = v
            row[k + "_code"] = 1 if v == "Yes" else 0 if v == "No" else -1
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("message_id").drop_duplicates("message_id", keep="last")
    return df


def split_labels_for_seed(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    for _, df_train, df_val, df_test in iter_domain_splits(df, seeds=[seed]):
        split = pd.Series(index=df["message_id"].values, data="", dtype=str)
        split.loc[df_train["message_id"].values] = "train"
        split.loc[df_val["message_id"].values] = "val"
        split.loc[df_test["message_id"].values] = "test"
        out = df[["message_id", "y"]].copy()
        out["split"] = out["message_id"].map(split.to_dict())
        if out["split"].isna().any() or (out["split"] == "").any():
            raise RuntimeError("Failed to assign splits for all messages.")
        return out
    raise RuntimeError("Seed not found")


def evaluate_zero_shot(df: pd.DataFrame, zero_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = df[["message_id", "y"]].merge(
        zero_df[["message_id", "p_high_risk"]], on="message_id", how="left"
    )
    if merged["p_high_risk"].isna().any():
        missing = int(merged["p_high_risk"].isna().sum())
        raise RuntimeError(f"Missing zero-shot predictions for {missing} messages")

    merged = merged.set_index("message_id")
    rows: list[dict[str, Any]] = []
    for seed, _, df_val, df_test in iter_domain_splits(df):
        val = merged.loc[df_val["message_id"].values]
        test = merged.loc[df_test["message_id"].values]
        best = sweep_thresholds(val["y"].values, val["p_high_risk"].values)
        test_metrics = compute_metrics(
            test["y"].values, test["p_high_risk"].values, best["threshold"]
        )
        rows.append(
            {
                "model": "zero_shot",
                "split_seed": int(seed),
                "threshold": float(best["threshold"]),
                "val_macro_f1": float(best["macro_f1"]),
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby("model")[
            ["test_accuracy", "test_roc_auc", "test_macro_f1", "test_brier", "test_ece"]
        ]
        .agg(["mean", "std"])
        .round(6)
    )
    return per_seed, summary


def evaluate_pastel(df: pd.DataFrame, signals_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [s["key"] for s in PASTEL_SIGNALS]
    needed = ["message_id"] + [k + "_code" for k in keys]
    if not set(needed).issubset(set(signals_df.columns)):
        missing = sorted(set(needed) - set(signals_df.columns))
        raise RuntimeError(f"Missing PASTEL signal columns: {missing}")

    sig = df[["message_id", "y"]].merge(signals_df[needed], on="message_id", how="left")
    if sig[[k + "_code" for k in keys]].isna().any().any():
        missing = int(sig[[k + "_code" for k in keys]].isna().any(axis=1).sum())
        raise RuntimeError(f"Missing PASTEL signals for {missing} messages")

    sig = sig.set_index("message_id")
    rows: list[dict[str, Any]] = []
    for seed, df_train, df_val, df_test in iter_domain_splits(df):
        train = sig.loc[df_train["message_id"].values]
        val = sig.loc[df_val["message_id"].values]
        test = sig.loc[df_test["message_id"].values]

        L_train = train[[k + "_code" for k in keys]].to_numpy(dtype=int)
        L_val = val[[k + "_code" for k in keys]].to_numpy(dtype=int)
        L_test = test[[k + "_code" for k in keys]].to_numpy(dtype=int)

        label_model = LabelModel(cardinality=2, device="cpu", verbose=False)
        label_model.fit(L_train, n_epochs=500, seed=42)

        proba_val = label_model.predict_proba(L_val)[:, 1]
        proba_test = label_model.predict_proba(L_test)[:, 1]

        best = sweep_thresholds(val["y"].values, proba_val)
        test_metrics = compute_metrics(test["y"].values, proba_test, best["threshold"])
        rows.append(
            {
                "model": "pastel",
                "split_seed": int(seed),
                "threshold": float(best["threshold"]),
                "val_macro_f1": float(best["macro_f1"]),
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby("model")[
            ["test_accuracy", "test_roc_auc", "test_macro_f1", "test_brier", "test_ece"]
        ]
        .agg(["mean", "std"])
        .round(6)
    )
    return per_seed, summary


def write_zero_shot_scores_seed0(
    df: pd.DataFrame, zero_df: pd.DataFrame, per_seed: pd.DataFrame
) -> None:
    seed = 0
    threshold = float(per_seed.loc[per_seed["split_seed"] == seed, "threshold"].iloc[0])
    splits = split_labels_for_seed(df, seed)
    merged = (
        df[["message_id", "source", "msg_id", "channel", "normalized_domain", "y"]]
        .merge(splits[["message_id", "split"]], on="message_id", how="left")
        .merge(zero_df[["message_id", "p_high_risk"]], on="message_id", how="left")
        .rename(columns={"y": "y_true", "p_high_risk": "p_hat_raw"})
    )
    merged["p_hat_calibrated"] = merged["p_hat_raw"]
    merged["y_pred"] = (merged["p_hat_raw"] >= threshold).astype(int)
    if len(merged) != len(df):
        raise RuntimeError("zero_shot_scores row mismatch")
    merged[
        [
            "message_id",
            "source",
            "msg_id",
            "channel",
            "normalized_domain",
            "split",
            "y_true",
            "p_hat_raw",
            "p_hat_calibrated",
            "y_pred",
        ]
    ].to_csv(RESULTS_DIR / "zero_shot_scores.csv", index=False)


def write_pastel_scores_seed0(
    df: pd.DataFrame, signals_df: pd.DataFrame, per_seed: pd.DataFrame
) -> None:
    seed = 0
    threshold = float(per_seed.loc[per_seed["split_seed"] == seed, "threshold"].iloc[0])
    keys = [s["key"] for s in PASTEL_SIGNALS]
    needed = ["message_id"] + [k + "_code" for k in keys]
    sig = df[["message_id", "y"]].merge(signals_df[needed], on="message_id", how="left")
    splits = split_labels_for_seed(df, seed)

    # Train label model on seed0 train split only.
    for _, df_train, _, _ in iter_domain_splits(df, seeds=[seed]):
        train = sig.set_index("message_id").loc[df_train["message_id"].values]
        L_train = train[[k + "_code" for k in keys]].to_numpy(dtype=int)
        label_model = LabelModel(cardinality=2, device="cpu", verbose=False)
        label_model.fit(L_train, n_epochs=500, seed=42)
        break
    else:
        raise RuntimeError("Could not build seed0 train split")

    L_all = sig[[k + "_code" for k in keys]].to_numpy(dtype=int)
    proba_all = label_model.predict_proba(L_all)[:, 1]

    merged = (
        df[["message_id", "source", "msg_id", "channel", "normalized_domain", "y"]]
        .merge(splits[["message_id", "split"]], on="message_id", how="left")
        .assign(p_hat_raw=proba_all.astype(float))
        .rename(columns={"y": "y_true"})
    )
    merged["p_hat_calibrated"] = merged["p_hat_raw"]
    merged["y_pred"] = (merged["p_hat_raw"] >= threshold).astype(int)
    if len(merged) != len(df):
        raise RuntimeError("pastel_scores row mismatch")
    merged[
        [
            "message_id",
            "source",
            "msg_id",
            "channel",
            "normalized_domain",
            "split",
            "y_true",
            "p_hat_raw",
            "p_hat_calibrated",
            "y_pred",
        ]
    ].to_csv(RESULTS_DIR / "pastel_scores.csv", index=False)


def ensure_our_pipeline_scores() -> None:
    out_path = RESULTS_DIR / "our_pipeline_scores.csv"
    if out_path.exists():
        return

    # Reproduce the seed=0 combined (TF-IDF + Style) pipeline from
    # mbfc_url_masked_logreg_v6.executed.ipynb, using fixed lr=0.01.
    from scipy import sparse
    from scipy.special import expit
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MultiLabelBinarizer

    def _stack_features(a, b):
        if sparse.issparse(a):
            return sparse.vstack([a, b])
        return np.vstack([a, b])

    class ManualLogisticRegression:
        def __init__(
            self,
            lr: float = 0.1,
            max_iter: int = 200,
            C: float = 1.0,
            class_weight=None,
            tol: float | None = 1e-4,
        ):
            self.lr = lr
            self.max_iter = max_iter
            self.C = C
            self.class_weight = class_weight
            self.tol = tol

        def _prepare_X(self, X):
            if sparse.issparse(X):
                return X.tocsr()
            return np.asarray(X, dtype=float)

        def fit(self, X, y):
            X = self._prepare_X(X)
            y = np.asarray(y, dtype=float)
            n_samples, n_features = X.shape
            self.coef_ = np.zeros(n_features, dtype=float)
            self.intercept_ = 0.0

            if self.class_weight is None:
                sample_weights = np.ones_like(y)
            elif self.class_weight == "balanced":
                classes, counts = np.unique(y, return_counts=True)
                n_classes = len(classes)
                class_weight_values = {
                    cls: n_samples / (n_classes * count)
                    for cls, count in zip(classes, counts)
                }
                sample_weights = np.array([class_weight_values[yi] for yi in y], dtype=float)
            else:
                raise ValueError("Unsupported class_weight")

            prev_loss = None
            for i in range(self.max_iter):
                z = X.dot(self.coef_) + self.intercept_
                p = expit(z)
                residual = (p - y) * sample_weights
                grad_w = X.T.dot(residual) / n_samples
                grad_w += self.coef_ / (self.C * n_samples)
                grad_b = residual.mean()

                self.coef_ -= self.lr * grad_w
                self.intercept_ -= self.lr * grad_b

                if self.tol is not None and (i % 10 == 0 or i == self.max_iter - 1):
                    z = X.dot(self.coef_) + self.intercept_
                    p = expit(z)
                    eps = 1e-15
                    loss_vec = (
                        -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                    ) * sample_weights
                    loss = loss_vec.mean() + 0.5 * np.sum(self.coef_ ** 2) / (
                        self.C * n_samples
                    )
                    if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                        break
                    prev_loss = loss
            return self

        def predict_proba(self, X):
            X = self._prepare_X(X)
            z = X.dot(self.coef_) + self.intercept_
            p_pos = expit(z)
            return np.vstack([1 - p_pos, p_pos]).T

    # Style token normalization copied from executed notebook v6.
    DROP_LINK_URL_LABEL = True
    _LINK_URL_LABEL_NORM = "link/url"

    def tokenize_multi(value: str):
        if not isinstance(value, str):
            return []
        value = value.replace("+", ",")
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not DROP_LINK_URL_LABEL:
            return parts
        return [p for p in parts if "".join(p.lower().split()) != _LINK_URL_LABEL_NORM]

    THEME_BUCKETS = [
        "Finance/Crypto",
        "Public health & medicine",
        "Politics",
        "Lifestyle & well-being",
        "Crime & public safety",
        "Gaming/Gambling",
        "News/Information",
        "Sports",
        "Technology",
        "Conversation/Chat/Other",
        "Other theme",
    ]

    def _norm_theme(raw):
        if not isinstance(raw, str):
            return None
        t = raw.strip()
        if not t:
            return None
        tl = t.lower()
        if t in THEME_BUCKETS:
            return t
        if any(k in tl for k in ["crypto", "token", "coin", "market", "finance", "econom"]):
            return "Finance/Crypto"
        if any(k in tl for k in ["health", "covid", "vaccine", "medicine", "disease", "pandemic", "hospital"]):
            return "Public health & medicine"
        if any(k in tl for k in ["politic", "election", "government", "policy", "war", "conflict", "ukraine", "russia"]):
            return "Politics"
        if any(k in tl for k in ["crime", "terror", "shooting", "police", "fraud", "scam"]):
            return "Crime & public safety"
        if any(k in tl for k in ["gaming", "gambling", "casino", "betting", "lottery"]):
            return "Gaming/Gambling"
        if any(k in tl for k in ["sport", "football", "soccer", "basketball", "tennis", "nba", "nfl"]):
            return "Sports"
        if any(k in tl for k in ["technology", "tech", "software", "platform", "ai ", "blockchain", "internet", "science", "research", "study"]):
            return "Technology"
        if any(k in tl for k in ["lifestyle", "well-being", "culture", "entertainment", "celebrity", "society", "community"]):
            return "Lifestyle & well-being"
        if any(k in tl for k in ["news", "headline", "breaking", "coverage", "update"]):
            return "News/Information"
        if any(k in tl for k in ["comment", "conversation", "chat", "q&a", "ama"]):
            return "Conversation/Chat/Other"
        return "Other theme"

    def _dedupe(out: list[str]) -> list[str]:
        seen = set()
        res = []
        for v in out:
            if v not in seen:
                seen.add(v)
                res.append(v)
        return res

    def _norm_claim_labels(raw):
        labels = tokenize_multi(raw)
        out: list[str] = []
        for base in labels:
            low = base.lower()
            if "no substantive claim" in low:
                out.append("No substantive claim")
            elif "verifiable" in low or "factual" in low:
                out.append("Verifiable factual statement")
            elif "rumour" in low or "unverified" in low:
                out.append("Rumour / unverified report")
            elif "misleading" in low or "cherry" in low:
                out.append("Misleading context / cherry-picking")
            elif "promotional" in low or "exaggerated profit" in low:
                out.append("Promotional hype / exaggerated profit guarantee")
            elif "fear" in low or "emotional" in low:
                out.append("Emotional appeal / fear-mongering")
            elif "scarcity" in low or "fomo" in low:
                out.append("Scarcity/FOMO tactic")
            elif "statistic" in low:
                out.append("Statistics")
            elif "fake" in low or "fabricated" in low:
                out.append("Fake content")
            elif "predict" in low or "forecast" in low:
                out.append("Speculative forecast / prediction")
            elif "announcement" in low:
                out.append("Announcement")
            elif "opinion" in low or "analysis" in low or "review" in low:
                out.append("Opinion / subjective statement")
            elif "assertion only" in low:
                out.append("None / assertion only")
            else:
                out.append("Other claim type")
        return _dedupe(out)

    def _norm_cta_labels(raw):
        labels = tokenize_multi(raw)
        out: list[str] = []
        for base in labels:
            low = base.lower()
            if "no cta" in low:
                out.append("No CTA")
            elif "engage" in low or "ask" in low:
                out.append("Engage/Ask questions")
            elif "attend" in low or "event" in low or "livestream" in low:
                out.append("Attend event / livestream")
            elif "join" in low or "subscribe" in low or "follow" in low:
                out.append("Join/Subscribe")
            elif "buy" in low or "invest" in low or "donate" in low:
                out.append("Buy / invest / donate")
            elif "share" in low or "repost" in low or "like" in low:
                out.append("Share / repost / like")
            elif "visit" in low or "watch" in low or "link" in low or "website" in low:
                out.append("Visit external link / watch video")
            else:
                out.append("Other CTA")
        return _dedupe(out)

    def _norm_evidence_labels(raw):
        labels = tokenize_multi(raw)
        out: list[str] = []
        for base in labels:
            low = base.lower()
            if "link" in low or "url" in low:
                continue
            if "statistic" in low:
                out.append("Statistics")
            elif "quote" in low or "testimony" in low:
                out.append("Quotes/Testimony")
            elif "chart" in low or "graph" in low or "diagram" in low:
                out.append("Chart / price graph / TA diagram")
            elif "assertion only" in low:
                out.append("None / assertion only")
            else:
                out.append("Other (Evidence)")
        return _dedupe(out)

    def build_style_tokens(row):
        tokens: list[str] = []
        theme = _norm_theme(row.get("theme"))
        if theme is not None:
            tokens.append(f"theme={theme}")
        for label in _norm_claim_labels(row.get("claim_types")):
            tokens.append(f"claim={label}")
        for label in _norm_cta_labels(row.get("ctas")):
            tokens.append(f"cta={label}")
        for label in _norm_evidence_labels(row.get("evidence")):
            tokens.append(f"evid={label}")
        return tokens

    df = load_mbfc_linkable_subset()
    seed = 0
    for _, df_train, df_val, df_test in iter_domain_splits(df, seeds=[seed]):
        vec = TfidfVectorizer(
            ngram_range=(1, 2), max_features=20000, min_df=2, strip_accents="unicode"
        )
        X_train_text = vec.fit_transform(df_train["message"].astype(str))
        X_val_text = vec.transform(df_val["message"].astype(str))
        X_test_text = vec.transform(df_test["message"].astype(str))

        train_tokens = df_train.apply(build_style_tokens, axis=1)
        val_tokens = df_val.apply(build_style_tokens, axis=1)
        test_tokens = df_test.apply(build_style_tokens, axis=1)
        mlb = MultiLabelBinarizer()
        X_train_style = mlb.fit_transform(train_tokens)
        X_val_style = mlb.transform(val_tokens)
        X_test_style = mlb.transform(test_tokens)

        y_train = df_train["y"].values
        y_val = df_val["y"].values

        def train_base(X_train, X_val, X_test, lr: float = 0.01):
            clf_tr = ManualLogisticRegression(
                lr=lr, max_iter=1000, C=1.0, class_weight="balanced", tol=None
            )
            clf_tr.fit(X_train, y_train)
            val_proba = clf_tr.predict_proba(X_val)[:, 1]
            train_proba = clf_tr.predict_proba(X_train)[:, 1]

            X_trainval = _stack_features(X_train, X_val)
            y_trainval = np.concatenate([y_train, y_val])
            clf_tv = ManualLogisticRegression(
                lr=lr, max_iter=1000, C=1.0, class_weight="balanced", tol=None
            )
            clf_tv.fit(X_trainval, y_trainval)
            test_proba = clf_tv.predict_proba(X_test)[:, 1]
            return train_proba, val_proba, test_proba

        tr_p_text, val_p_text, te_p_text = train_base(X_train_text, X_val_text, X_test_text)
        tr_p_style, val_p_style, te_p_style = train_base(X_train_style, X_val_style, X_test_style)

        meta_X_val = np.stack([val_p_text, val_p_style, val_p_text * val_p_style], axis=1)
        meta_clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
        meta_clf.fit(meta_X_val, y_val)
        meta_val_proba = meta_clf.predict_proba(meta_X_val)[:, 1]
        thr = float(sweep_thresholds(y_val, meta_val_proba)["threshold"])

        meta_X_test = np.stack([te_p_text, te_p_style, te_p_text * te_p_style], axis=1)
        meta_test_proba = meta_clf.predict_proba(meta_X_test)[:, 1]

        meta_X_train = np.stack([tr_p_text, tr_p_style, tr_p_text * tr_p_style], axis=1)
        meta_train_proba = meta_clf.predict_proba(meta_X_train)[:, 1]

        def pack(df_part: pd.DataFrame, split: str, proba: np.ndarray) -> pd.DataFrame:
            out = df_part[["message_id", "source", "msg_id", "channel", "normalized_domain", "y"]].copy()
            out = out.rename(columns={"y": "y_true"})
            out["split"] = split
            out["p_hat_raw"] = proba.astype(float)
            out["p_hat_calibrated"] = out["p_hat_raw"]
            out["y_pred"] = (out["p_hat_raw"] >= thr).astype(int)
            return out

        scores = pd.concat(
            [
                pack(df_train, "train", meta_train_proba),
                pack(df_val, "val", meta_val_proba),
                pack(df_test, "test", meta_test_proba),
            ],
            ignore_index=True,
        )
        if len(scores) != len(df):
            raise RuntimeError("our_pipeline_scores row mismatch")
        scores.to_csv(out_path, index=False)
        return
    raise RuntimeError("Failed to compute our_pipeline_scores")


def generate_report(
    *,
    our_metrics_path: Path,
    zero_summary: pd.DataFrame,
    pastel_summary: pd.DataFrame,
    signals_df: pd.DataFrame,
    pastel_mode: str,
    message_lookup: pd.DataFrame,
) -> None:
    report_path = RESULTS_DIR / "report.md"
    prompts_path = RESULTS_DIR / "prompts.json"

    prompts_path.write_text(
        json.dumps(
            {
                "zero_shot_system": ZERO_SHOT_SYSTEM_PROMPT,
                "zero_shot_user_template": ZERO_SHOT_USER_TEMPLATE,
                "pastel_system": PASTEL_SYSTEM_PROMPT,
                "pastel_user_template": PASTEL_USER_TEMPLATE,
                "pastel_single_user_template": PASTEL_SINGLE_USER_TEMPLATE,
                "pastel_questions": pastel_questions_block(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    our = pd.read_csv(our_metrics_path)
    our_sum = (
        our.groupby("model")[
            ["test_accuracy", "test_roc_auc", "test_macro_f1", "test_brier", "test_ece"]
        ]
        .agg(["mean", "std"])
        .round(6)
    )

    def fmt(summary: pd.DataFrame, model_key: str, metric: str) -> str:
        mean = float(summary.loc[model_key, (metric, "mean")])
        std = float(summary.loc[model_key, (metric, "std")])
        return f"{mean:.3f} ± {std:.3f}"

    # Table aligned to your paper Table 2.
    rows = [
        (
            "TF-IDF",
            fmt(our_sum, "tfidf", "test_accuracy"),
            fmt(our_sum, "tfidf", "test_roc_auc"),
            fmt(our_sum, "tfidf", "test_macro_f1"),
            fmt(our_sum, "tfidf", "test_brier"),
            fmt(our_sum, "tfidf", "test_ece"),
        ),
        (
            "TAG2CRED (all tags)",
            fmt(our_sum, "style", "test_accuracy"),
            fmt(our_sum, "style", "test_roc_auc"),
            fmt(our_sum, "style", "test_macro_f1"),
            fmt(our_sum, "style", "test_brier"),
            fmt(our_sum, "style", "test_ece"),
        ),
        (
            "Ensemble",
            fmt(our_sum, "combined", "test_accuracy"),
            fmt(our_sum, "combined", "test_roc_auc"),
            fmt(our_sum, "combined", "test_macro_f1"),
            fmt(our_sum, "combined", "test_brier"),
            fmt(our_sum, "combined", "test_ece"),
        ),
        (
            "GPT-4o-mini (zero-shot)",
            fmt(zero_summary, "zero_shot", "test_accuracy"),
            fmt(zero_summary, "zero_shot", "test_roc_auc"),
            fmt(zero_summary, "zero_shot", "test_macro_f1"),
            fmt(zero_summary, "zero_shot", "test_brier"),
            fmt(zero_summary, "zero_shot", "test_ece"),
        ),
        (
            "PASTEL (signals + Snorkel LabelModel)",
            fmt(pastel_summary, "pastel", "test_accuracy"),
            fmt(pastel_summary, "pastel", "test_roc_auc"),
            fmt(pastel_summary, "pastel", "test_macro_f1"),
            fmt(pastel_summary, "pastel", "test_brier"),
            fmt(pastel_summary, "pastel", "test_ece"),
        ),
    ]

    table = [
        "| Model | Acc | AUC | Macro-F1 | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, acc, auc, f1m, brier, ece in rows:
        table.append(f"| {model} | {acc} | {auc} | {f1m} | {brier} | {ece} |")

    # Signal collapse (distribution)
    sig_lines = []
    for s in PASTEL_SIGNALS:
        k = s["key"]
        vc = signals_df[k].value_counts(normalize=True)
        sig_lines.append(
            f"- {k}: Yes={vc.get('Yes', 0.0):.3f}, No={vc.get('No', 0.0):.3f}, Unsure={vc.get('Unsure', 0.0):.3f}"
        )

    # Batch vs single agreement (if present)
    agree_path = RESULTS_DIR / "pastel_batch_vs_single_agreement.csv"
    agreement_block = ""
    if agree_path.exists():
        agree = pd.read_csv(agree_path)
        mean_agree = float(agree["agreement"].mean()) if len(agree) else float("nan")
        agreement_block = (
            "## PASTEL batching deviation check\n\n"
            f"- Full run mode: `{pastel_mode}`\n"
            "- Deviation: the paper extracts one signal per call; we batch all 19 signals per call for scalability.\n"
            f"- Sanity subset agreement (mean over 19 signals): `{mean_agree:.3f}`\n"
            f"- Per-signal agreement CSV: `{agree_path}`\n"
        )

    def summarize_raw_dir(raw_dir: Path) -> dict[str, Any]:
        ok = 0
        err = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        elapsed_sum = 0.0
        for rec in iter_jsonl_dir(raw_dir):
            if rec.get("error") is not None:
                err += 1
                continue
            ok += 1
            usage = rec.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            if "elapsed_s" in rec:
                try:
                    elapsed_sum += float(rec["elapsed_s"])
                except Exception:
                    pass
        return {
            "calls_ok": ok,
            "calls_error": err,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "elapsed_sum_s": elapsed_sum,
        }

    # Cost/time accounting (tokens and call counts)
    usage_rows: list[dict[str, Any]] = []
    usage_zero = summarize_raw_dir(RESULTS_DIR / "zero_shot_raw")
    usage_rows.append({"component": "zero_shot", **usage_zero})
    if pastel_mode == "batch":
        usage_pastel = summarize_raw_dir(RESULTS_DIR / "pastel_signals_raw_batch")
    else:
        usage_pastel = summarize_raw_dir(RESULTS_DIR / "pastel_signals_raw_single")
    usage_rows.append({"component": f"pastel_{pastel_mode}", **usage_pastel})
    usage_df = pd.DataFrame(usage_rows)
    usage_df.to_csv(RESULTS_DIR / "usage_summary.csv", index=False)

    price_in = os.environ.get("OPENAI_USD_PER_1M_INPUT")
    price_out = os.environ.get("OPENAI_USD_PER_1M_OUTPUT")
    cost_note = (
        "Set `OPENAI_USD_PER_1M_INPUT` and `OPENAI_USD_PER_1M_OUTPUT` env vars to compute a $ estimate.\n"
    )
    est_cost_line = ""
    if price_in and price_out:
        pin = float(price_in)
        pout = float(price_out)
        input_tok = int(usage_df["prompt_tokens"].sum())
        output_tok = int(usage_df["completion_tokens"].sum())
        est = (input_tok / 1_000_000.0) * pin + (output_tok / 1_000_000.0) * pout
        est_cost_line = f"- Estimated cost: `${est:.2f}` using input=${pin}/1M, output=${pout}/1M\n"
        cost_note = ""

    cost_block = (
        "## Cost / throughput\n\n"
        f"- Total calls: `{int(usage_df['calls_ok'].sum())}` (ok), `{int(usage_df['calls_error'].sum())}` (error records)\n"
        f"- Total tokens: `{int(usage_df['prompt_tokens'].sum())}` prompt, `{int(usage_df['completion_tokens'].sum())}` completion\n"
        + est_cost_line
        + cost_note
        + f"- Raw usage CSV: `{RESULTS_DIR / 'usage_summary.csv'}`\n"
    )

    # Failure analysis: seed-0 test errors
    def load_scores(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    zs_scores = load_scores(RESULTS_DIR / "zero_shot_scores.csv")
    ps_scores = load_scores(RESULTS_DIR / "pastel_scores.csv")
    op_scores = load_scores(RESULTS_DIR / "our_pipeline_scores.csv")

    msg = message_lookup[["message_id", "message"]].copy()
    msg["message"] = msg["message"].astype(str)

    def top_errors(scores: pd.DataFrame, name: str) -> str:
        if scores.empty:
            return f"- {name}: scores missing\n"
        test = scores[scores["split"] == "test"].copy()
        if test.empty:
            return f"- {name}: no test split rows\n"
        tp = int(((test["y_true"] == 1) & (test["y_pred"] == 1)).sum())
        tn = int(((test["y_true"] == 0) & (test["y_pred"] == 0)).sum())
        fp = int(((test["y_true"] == 0) & (test["y_pred"] == 1)).sum())
        fn = int(((test["y_true"] == 1) & (test["y_pred"] == 0)).sum())

        fp_ex = (
            test[(test["y_true"] == 0) & (test["y_pred"] == 1)]
            .sort_values("p_hat_raw", ascending=False)
            .head(5)
            .merge(msg, on="message_id", how="left")
        )
        fn_ex = (
            test[(test["y_true"] == 1) & (test["y_pred"] == 0)]
            .sort_values("p_hat_raw", ascending=True)
            .head(5)
            .merge(msg, on="message_id", how="left")
        )

        def fmt_row(r: pd.Series) -> str:
            text = str(r.get("message", "")).replace("\n", " ").strip()
            if len(text) > 220:
                text = text[:220] + "…"
            return f"  - message_id={int(r['message_id'])} p={float(r['p_hat_raw']):.3f} text=\"{text}\""

        lines = [
            f"- {name} (seed0 test): TP={tp} TN={tn} FP={fp} FN={fn}",
            "  - False positives (top-5 by confidence):",
            *[fmt_row(r) for _, r in fp_ex.iterrows()],
            "  - False negatives (top-5 by confidence):",
            *[fmt_row(r) for _, r in fn_ex.iterrows()],
        ]
        return "\n".join(lines) + "\n"

    failure_block = (
        "## Failure analysis (seed=0 test split)\n\n"
        + top_errors(zs_scores, "GPT-4o-mini zero-shot")
        + "\n"
        + top_errors(ps_scores, f"PASTEL ({pastel_mode})")
        + "\n"
        + top_errors(op_scores, "OUR_PIPELINE (seed0 combined)")
    )

    n_messages = int(len(signals_df))
    n_note = "" if n_messages == 87936 else f"(subset run: N={n_messages})"

    pipeline_ref = REPO_ROOT / "notebooks" / "01_mbfc_url_masked_logreg_v6.ipynb"
    report = f"""# GPT-4o-mini Zero-shot + PASTEL replication (MBFC-linkable Telegram subset)

## Primary pipeline reference

- `{pipeline_ref}`

## Dataset + label

- Input: `{DATA_PATH}` filtered to MBFC-linkable rows with `risk_label` and `normalized_domain` present.
- N = {n_messages} messages {n_note}
- Objective label: `y = risk_label` (1 = higher-risk / lower-credibility, 0 = higher-credibility), constructed from MBFC credibility + factuality via noisy-OR with high-confidence thresholds (`tau_low=0.3`, `tau_high=0.8`).

## Preprocessing

- URL masking: replace `(https?://|http://|www\\.[^\\s]*|t\\.me/[^\\s]*)` with a space, then strip; drop rows that become empty.

## Splits + evaluation (copied from our pipeline)

- Domain-disjoint splits using `normalized_domain` groups.
- For each seed in 0..9:
  - `GroupShuffleSplit(test_size=0.2, random_state=seed)` to create train+val vs test by domain.
  - `train_test_split(test_size=0.125, random_state=100+seed, stratify=y)` within train+val to create train vs val.
- Threshold selection: sweep `t ∈ {{0.05, 0.10, …, 0.95}}` on **val** to maximize Macro-F1, then evaluate on test.
- Metrics: Accuracy, ROC-AUC, Macro-F1, Brier, ECE (10 equal-width bins).

## PASTEL reference

- Questions are embedded in `scripts/run_all_baselines.py` (adapted from the official PASTEL repo).
- Weak supervision: Snorkel `LabelModel`, trained on TRAIN signals only for 500 epochs.

## Prompts

Prompts are saved verbatim in `{prompts_path}`.

## Results (mean ± std over 10 domain partitions)

{chr(10).join(table)}

{agreement_block}

{cost_block}

## PASTEL signal distribution (all messages in run)

{chr(10).join(sig_lines)}

{failure_block}
"""

    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument(
        "--pastel-mode",
        choices=["batch", "single"],
        default="batch",
        help="PASTEL extraction mode for the full dataset (batch=1 call/message; single=19 calls/message).",
    )
    parser.add_argument(
        "--pastel-sanity-n",
        type=int,
        default=200,
        help="If >0, run both single and batch on this many messages and save agreement stats.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM inference steps and only compile/evaluate from existing raw files.",
    )
    args = parser.parse_args()

    df_full = load_mbfc_linkable_subset()
    if args.max_messages is not None:
        df = df_full.sort_values("message_id").head(int(args.max_messages)).copy()
    else:
        df = df_full
    ensure_our_pipeline_scores()

    # Optional sanity check: compare PASTEL single vs batch on a small subset.
    if not args.skip_llm and args.pastel_sanity_n and args.pastel_sanity_n > 0:
        df_sanity = df_full.sample(
            n=min(int(args.pastel_sanity_n), len(df_full)),
            random_state=123,
            replace=False,
        ).copy()
        append_progress(f"[sanity] pastel_sanity_n={len(df_sanity)}")
        asyncio.run(
            run_pastel_signals_single(
                df_sanity,
                concurrency=args.concurrency,
                shard_size=args.shard_size,
                log_every=args.log_every,
            )
        )
        asyncio.run(
            run_pastel_signals_batch(
                df_sanity,
                concurrency=args.concurrency,
                shard_size=args.shard_size,
                log_every=args.log_every,
            )
        )
        sig_single = compile_pastel_signals_single(RESULTS_DIR / "pastel_signals_raw_single")
        sig_batch = compile_pastel_signals_batch(RESULTS_DIR / "pastel_signals_raw_batch")
        sig_single = sig_single[sig_single["message_id"].isin(df_sanity["message_id"])]
        sig_batch = sig_batch[sig_batch["message_id"].isin(df_sanity["message_id"])]
        sig_single.to_csv(RESULTS_DIR / "pastel_signals_sanity_single.csv", index=False)
        sig_batch.to_csv(RESULTS_DIR / "pastel_signals_sanity_batch.csv", index=False)

        keys = [s["key"] for s in PASTEL_SIGNALS]
        merged = sig_single[["message_id"] + keys].merge(
            sig_batch[["message_id"] + keys],
            on="message_id",
            how="inner",
            suffixes=("_single", "_batch"),
        )
        agree_rows = []
        for k in keys:
            a = merged[f"{k}_single"].values
            b = merged[f"{k}_batch"].values
            agree = float(np.mean(a == b)) if len(merged) else float("nan")
            agree_rows.append({"signal": k, "agreement": agree, "n": int(len(merged))})
        pd.DataFrame(agree_rows).to_csv(
            RESULTS_DIR / "pastel_batch_vs_single_agreement.csv", index=False
        )

    # Zero-shot
    raw_zero_dir = RESULTS_DIR / "zero_shot_raw"
    if not args.skip_llm:
        asyncio.run(
            run_zero_shot_inference(
                df,
                concurrency=args.concurrency,
                shard_size=args.shard_size,
                log_every=args.log_every,
                max_messages=None,
            )
        )
    zero_df = compile_zero_shot_outputs(raw_zero_dir)
    if len(zero_df) != len(df):
        raise RuntimeError(f"zero_shot predictions incomplete: {len(zero_df)}/{len(df)}. Re-run.")
    zero_df.to_csv(RESULTS_DIR / "zero_shot_outputs.csv", index=False)
    zero_seed, zero_summary = evaluate_zero_shot(df, zero_df)
    zero_seed.to_csv(RESULTS_DIR / "zero_shot_metrics_by_seed.csv", index=False)
    zero_summary.to_csv(RESULTS_DIR / "zero_shot_metrics_summary.csv")
    write_zero_shot_scores_seed0(df, zero_df, zero_seed)

    # PASTEL signals + label model
    if not args.skip_llm:
        if args.pastel_mode == "batch":
            asyncio.run(
                run_pastel_signals_batch(
                    df,
                    concurrency=args.concurrency,
                    shard_size=args.shard_size,
                    log_every=args.log_every,
                )
            )
        else:
            asyncio.run(
                run_pastel_signals_single(
                    df,
                    concurrency=args.concurrency,
                    shard_size=args.shard_size,
                    log_every=args.log_every,
                )
            )

    if args.pastel_mode == "batch":
        raw_sig_dir = RESULTS_DIR / "pastel_signals_raw_batch"
        signals_df = compile_pastel_signals_batch(raw_sig_dir)
    else:
        raw_sig_dir = RESULTS_DIR / "pastel_signals_raw_single"
        signals_df = compile_pastel_signals_single(raw_sig_dir)

    if len(signals_df) != len(df):
        raise RuntimeError(f"pastel signals incomplete: {len(signals_df)}/{len(df)}. Re-run.")
    signals_df.to_csv(RESULTS_DIR / "pastel_signals.csv", index=False)
    pastel_seed, pastel_summary = evaluate_pastel(df, signals_df)
    pastel_seed.to_csv(RESULTS_DIR / "pastel_metrics_by_seed.csv", index=False)
    pastel_summary.to_csv(RESULTS_DIR / "pastel_metrics_summary.csv")
    write_pastel_scores_seed0(df, signals_df, pastel_seed)

    our_metrics_path = Path(
        os.environ.get(
            "OUR_METRICS_PATH",
            str(REPO_ROOT / "results" / "url_masked_val_tuned_metrics_no_url.csv"),
        )
    )
    generate_report(
        our_metrics_path=our_metrics_path,
        zero_summary=zero_summary,
        pastel_summary=pastel_summary,
        signals_df=signals_df,
        pastel_mode=args.pastel_mode,
        message_lookup=df_full[["message_id", "message"]],
    )

    append_progress("[all] done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        append_progress("[all] interrupted")
        raise
