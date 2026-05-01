"""
codebook_normalize_and_fill.py

Reproducible, deterministic pipeline for:
1) Backfilling missing `channel` and `date` in full_risk_v2_core.csv by matching
   against the raw Telegram corpus in mega_messages_all_raw.csv.xz.
2) Normalizing Qwen-generated labels into the closed tag vocabulary defined by
   the codebook (Theme, Claim types, CTAs, Evidence), with hard constraint
   enforcement.

Outputs:
  - full_risk_v2_core_final.csv  (filled + codebook-normalized)
  - fill summaries and audit columns for traceability

This script is designed to be open-sourced and rerun end-to-end.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# -----------------------------
# Paths (edit if needed)
# -----------------------------
CORE_IN = Path("full_risk_v2_core.csv")
MEGA_RAW = Path("mega_messages_all_raw.csv.xz")

OUT_FILLED_EXACT = Path("full_risk_v2_core_filled.csv")
OUT_FILLED_FUZZY = Path("full_risk_v2_core_filled_fuzzy.csv")
OUT_FINAL = Path("full_risk_v2_core_final.csv")

OUT_EXACT_SUMMARY = Path("full_risk_v2_core_fill_summary.json")
OUT_FUZZY_SUMMARY = Path("full_risk_v2_core_fuzzy_fill_summary.json")
OUT_CODEBOOK_SUMMARY = Path("full_risk_v2_core_codebook_summary.json")


# -----------------------------
# Codebook vocabularies
# -----------------------------
THEMES = [
    "Finance/Crypto",
    "Public health & medicine",
    "Politics",
    "Crime & public safety",
    "News/Information",
    "Technology",
    "Lifestyle & well‑being",
    "Gaming/Gambling",
    "Sports",
    "Conversation/Chat/Other",
    "Other (Theme)",
]
THEME_ORDER = {t: i for i, t in enumerate(THEMES)}

CLAIMS = [
    "No substantive claim",
    "Announcement",
    "Speculative forecast / prediction",
    "Promotional hype / exaggerated profit guarantee",
    "Scarcity/FOMO tactic",
    "Misleading context / cherry‑picking",
    "Emotional appeal / fear‑mongering",
    "Rumour / unverified report",
    "Opinion / subjective statement",
    "Verifiable factual statement",
    "Other (Claim type)",
]
CLAIM_ORDER = {c: i for i, c in enumerate(CLAIMS)}

CTAS = [
    "Share / repost / like",
    "Engage/Ask questions",
    "Visit external link / watch video",
    "Buy / invest / donate",
    "Join/Subscribe",
    "Attend event / livestream",
    "No CTA",
]
CTA_ORDER = {c: i for i, c in enumerate(CTAS)}

EVID = [
    "None / assertion only",
    "Link/URL",
    "Quotes/Testimony",
    "Statistics",
    "Chart / price graph / TA diagram",
    "Other (Evidence)",
]
EVID_ORDER = {e: i for i, e in enumerate(EVID)}


# -----------------------------
# Deterministic synonym maps
# -----------------------------
THEME_SYNONYMS: Dict[str, str] = {
    "lifestyle & well-being": "Lifestyle & well‑being",
    "lifestyle & well being": "Lifestyle & well‑being",
    "healthcare": "Public health & medicine",
    "health": "Public health & medicine",
    "medicine": "Public health & medicine",
    "public health": "Public health & medicine",
    "finance": "Finance/Crypto",
    "economics": "Finance/Crypto",
    "cryptocurrency trading / investment": "Finance/Crypto",
    "cryptocurrency trading/investment": "Finance/Crypto",
    "crypto/blockchain": "Finance/Crypto",
    "cryptocurrency/blockchain": "Finance/Crypto",
    "cryptocurrency": "Finance/Crypto",
    "blockchain": "Finance/Crypto",
    "nft creation / minting": "Finance/Crypto",
    "promotion / advertising": "Finance/Crypto",
}

CLAIM_SYNONYMS: Dict[str, str] = {
    "rumor / unverified report": "Rumour / unverified report",
    "rumour/unverified report": "Rumour / unverified report",
    "speculative forecast/prediction": "Speculative forecast / prediction",
    "promotional hype / exaggerated profit guarantee ": "Promotional hype / exaggerated profit guarantee",
    "emotional appeal / fear-mongering": "Emotional appeal / fear‑mongering",
    "misleading context / cherry-picking": "Misleading context / cherry‑picking",
}

CTA_SYNONYMS: Dict[str, str] = {
    "visit external link / watch video ": "Visit external link / watch video",
    "engage/ask questions ": "Engage/Ask questions",
    "join/subscribe ": "Join/Subscribe",
    "share / repost / like ": "Share / repost / like",
    "buy / invest / donate ": "Buy / invest / donate",
    "attend event / livestream ": "Attend event / livestream",
}

EVID_SYNONYMS: Dict[str, str] = {
    "none": "None / assertion only",
    "link/url ": "Link/URL",
    "quotes/testimony": "Quotes/Testimony",
    "chart / price graph / ta diagram": "Chart / price graph / TA diagram",
}


# -----------------------------
# Keyword heuristics for messy labels
# -----------------------------
KW_THEME = [
    (
        re.compile(
            r"(crypto|cryptocurrency|token|coin|btc|eth|defi|airdrop|staking|wallet|exchange|price|liquidity|holders|market|trading|nft|web3|blockchain)",
            re.I,
        ),
        "Finance/Crypto",
    ),
    (
        re.compile(
            r"(vaccine|covid|health|medicine|medical|disease|doctor|clinic|pharma|virus|public health)",
            re.I,
        ),
        "Public health & medicine",
    ),
    (
        re.compile(
            r"(election|politic|government|president|senate|parliament|policy|democrat|republican|ukraine|russia|israel|gaza)",
            re.I,
        ),
        "Politics",
    ),
    (
        re.compile(
            r"(crime|police|shooting|attack|terror|security alert|missing person|public safety)",
            re.I,
        ),
        "Crime & public safety",
    ),
    (
        re.compile(
            r"(ai|artificial intelligence|software|app|platform|tech|technology|robot|innovation|cyber|data|model)",
            re.I,
        ),
        "Technology",
    ),
    (re.compile(r"(casino|gambl|betting|slots|poker|blackjack)", re.I), "Gaming/Gambling"),
    (
        re.compile(
            r"(sport|football|soccer|nba|nfl|match|tournament|goal|league)",
            re.I,
        ),
        "Sports",
    ),
    (
        re.compile(
            r"(fitness|diet|well[-\s]?being|wellness|lifestyle|productivity|mindset)",
            re.I,
        ),
        "Lifestyle & well‑being",
    ),
    (
        re.compile(
            r"(hello|thanks|thank you|gm\b|good morning|good night|welcome|admin|channel meta|housekeeping)",
            re.I,
        ),
        "Conversation/Chat/Other",
    ),
]

KW_CLAIM = [
    (re.compile(r"no substantive claim|no claim|no substantive", re.I), "No substantive claim"),
    (re.compile(r"announcement|housekeeping|schedule|available now", re.I), "Announcement"),
    (re.compile(r"forecast|prediction|target|will|expect|projected", re.I), "Speculative forecast / prediction"),
    (
        re.compile(r"guarantee|profit guarantee|set to explode|no risk|100x|10x|5x|moon", re.I),
        "Promotional hype / exaggerated profit guarantee",
    ),
    (re.compile(r"fomo|scarcity|last chance|ends today|limited", re.I), "Scarcity/FOMO tactic"),
    (re.compile(r"cherry|misleading context", re.I), "Misleading context / cherry‑picking"),
    (re.compile(r"fear|fear-mongering|emotional appeal|panic|outrage", re.I), "Emotional appeal / fear‑mongering"),
    (re.compile(r"rumou?r|unverified|allegedly|leaked|sources say", re.I), "Rumour / unverified report"),
    (re.compile(r"opinion|subjective", re.I), "Opinion / subjective statement"),
    (re.compile(r"verifiable factual|factual statement|confirmed|reports that", re.I), "Verifiable factual statement"),
]

KW_CTA = [
    (re.compile(r"share|repost|retweet|like", re.I), "Share / repost / like"),
    (re.compile(r"comment|reply|vote|dm|tell us|what do you think|ask", re.I), "Engage/Ask questions"),
    (re.compile(r"watch|click|read more|link below|👇|👉|➡️", re.I), "Visit external link / watch video"),
    (re.compile(r"buy|sell|invest|donate|long|short|entry|tp|sl", re.I), "Buy / invest / donate"),
    (re.compile(r"join|subscribe|follow|register|whitelist", re.I), "Join/Subscribe"),
    (re.compile(r"live now|livestream|ama|webinar|event|stream", re.I), "Attend event / livestream"),
]

KW_EVID = [
    (re.compile(r"link|url|http", re.I), "Link/URL"),
    (re.compile(r"\"|\bsaid\b|according to|testimony", re.I), "Quotes/Testimony"),
    (re.compile(r"\b\d+\b|%|million|billion|price|profit", re.I), "Statistics"),
    (re.compile(r"chart|graph|tradingview|ta\b", re.I), "Chart / price graph / TA diagram"),
]


def norm_text(s: str) -> str:
    """Unicode + whitespace canonicalization for labels."""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("–", "-").replace("—", "-").replace("‑", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def map_single(
    raw: str,
    canon_list: List[str],
    syn_map: Dict[str, str],
    kw_list: List[Tuple[re.Pattern, str]],
    other_label: Optional[str],
) -> Optional[str]:
    """Map one raw label into codebook tags, else Other."""
    r = norm_text(raw)
    if not r or r.lower() == "nan":
        return None
    r_low = r.lower()
    for c in canon_list:
        if r_low == c.lower():
            return c
    if r_low in syn_map:
        return syn_map[r_low]
    for pat, lab in kw_list:
        if pat.search(r):
            return lab
    return other_label


def map_field(
    raw_field: str,
    canon_list: List[str],
    syn_map: Dict[str, str],
    kw_list: List[Tuple[re.Pattern, str]],
    order_map: Dict[str, int],
    other_label: str,
    allow_other: bool = True,
) -> str:
    """Split multi-label raw fields on commas, map, dedupe, sort."""
    if pd.isna(raw_field):
        return other_label
    r = norm_text(raw_field)
    if not r or r.lower() == "nan":
        return other_label
    parts = [p.strip() for p in r.split(",") if p.strip()]
    mapped: List[str] = []
    for p in parts:
        lab = map_single(p, canon_list, syn_map, kw_list, other_label if allow_other else None)
        if lab:
            mapped.append(lab)
    mapped = list(dict.fromkeys(mapped))
    if not mapped:
        return other_label
    if other_label in mapped and len(mapped) > 1:
        mapped = [m for m in mapped if m != other_label]
    mapped.sort(key=lambda x: order_map.get(x, 999))
    return ", ".join(mapped)


def enforce_claim_rules(label_str: str) -> str:
    """Apply forbidden-pair and max-3 rules for claim types."""
    if not label_str:
        return "Other (Claim type)"
    labs = [l.strip() for l in label_str.split(",") if l.strip()]
    labs = list(dict.fromkeys(labs))
    if not labs:
        return "Other (Claim type)"
    if "No substantive claim" in labs:
        return "No substantive claim"

    def drop_pair(a: str, b: str) -> None:
        if a in labs and b in labs:
            keep = a if CLAIM_ORDER[a] < CLAIM_ORDER[b] else b
            drop = b if keep == a else a
            labs.remove(drop)

    drop_pair("Rumour / unverified report", "Verifiable factual statement")
    drop_pair("Announcement", "Verifiable factual statement")

    labs.sort(key=lambda x: CLAIM_ORDER.get(x, 999))
    labs = labs[:3]
    return ", ".join(labs)


def enforce_cta_rules(label_str: str) -> str:
    """Remove No CTA when any CTA exists."""
    labs = [l.strip() for l in (label_str or "").split(",") if l.strip()]
    labs = [l for l in dict.fromkeys(labs) if l in CTA_ORDER]
    if not labs:
        return "No CTA"
    if "No CTA" in labs and len(labs) > 1:
        labs = [l for l in labs if l != "No CTA"]
    labs.sort(key=lambda x: CTA_ORDER.get(x, 999))
    return ", ".join(labs)


def enforce_evidence_rules(label_str: str) -> str:
    """None / assertion only cannot co-occur with other evidence."""
    labs = [l.strip() for l in (label_str or "").split(",") if l.strip()]
    labs = list(dict.fromkeys(labs))
    if not labs:
        return "Other (Evidence)"
    if "None / assertion only" in labs and len(labs) > 1:
        labs = [l for l in labs if l != "None / assertion only"]
    labs.sort(key=lambda x: EVID_ORDER.get(x, 999))
    return ", ".join(labs)


# -----------------------------
# Filling helpers
# -----------------------------
URL_RE = re.compile(r"(https?://[^\s<>\"]+|www\.[^\s<>\"]+)", re.I)
USER_RE = re.compile(r"@\w{2,}")


def norm_key_exact(text: str) -> str:
    return str(text).strip()


def norm_key_fuzzy(text: str) -> str:
    """Aggressive but safe key; used only with unique-channel constraint."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = URL_RE.sub("<url>", t)
    t = USER_RE.sub("<user>", t)
    t = re.sub(r"[^a-z0-9<>]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def scan_mega_for_keys(
    keys: set,
    key_fn,
    usecols: List[str],
    chunk_size: int = 200_000,
) -> Dict[str, List[Dict]]:
    """Scan mega raw corpus once; collect matches for provided keys."""
    matches = defaultdict(list)
    reader = pd.read_csv(
        MEGA_RAW,
        compression="xz",
        chunksize=chunk_size,
        low_memory=False,
        usecols=usecols,
    )
    for chunk in reader:
        kser = chunk["text"].fillna("").astype(str).apply(key_fn)
        mask = kser.isin(keys)
        if mask.any():
            sub = chunk.loc[mask, usecols].copy()
            sub["key"] = kser[mask].values
            for row in sub.itertuples(index=False):
                matches[row.key].append(
                    {"channel": row.channel, "date": row.date, "msg_id": row.msg_id, "text": row.text}
                )
    return matches


def choose_unique_channel_fill(matches: Dict[str, List[Dict]]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int], Dict[str, str]]:
    """Only fill when all matches for a key come from exactly one channel."""
    fill_channel, fill_date, fill_count, fill_channels = {}, {}, {}, {}
    for k, lst in matches.items():
        chans = {m["channel"] for m in lst}
        fill_count[k] = len(lst)
        fill_channels[k] = ";".join(sorted(chans))
        if len(chans) == 1:
            fill_channel[k] = next(iter(chans))
            dates = pd.to_datetime([m["date"] for m in lst], errors="coerce", utc=True)
            dates = dates[pd.notna(dates)]
            fill_date[k] = dates.min().isoformat() if len(dates) else None
    return fill_channel, fill_date, fill_count, fill_channels


def main() -> None:
    # -------------------------
    # 1) Exact backfill
    # -------------------------
    core_full = pd.read_csv(CORE_IN, low_memory=False)
    missing_mask = core_full["channel"].isna() & core_full["date"].isna()
    missing_texts = core_full.loc[missing_mask, "message_original"].fillna("").astype(str)
    missing_keys = set(missing_texts.apply(norm_key_exact))

    exact_matches = scan_mega_for_keys(
        missing_keys,
        norm_key_exact,
        usecols=["channel", "msg_id", "date", "text"],
    )

    fill_channel, fill_date, fill_count, fill_channels = choose_unique_channel_fill(exact_matches)

    core_full.loc[missing_mask, "fill_match_count"] = missing_texts.apply(norm_key_exact).map(fill_count)
    core_full.loc[missing_mask, "fill_match_channels"] = missing_texts.apply(norm_key_exact).map(fill_channels)
    core_full.loc[missing_mask, "filled_channel"] = missing_texts.apply(norm_key_exact).map(fill_channel)
    core_full.loc[missing_mask, "filled_date"] = missing_texts.apply(norm_key_exact).map(fill_date)

    core_full.loc[missing_mask, "channel"] = core_full.loc[missing_mask, "channel"].fillna(
        core_full.loc[missing_mask, "filled_channel"]
    )
    core_full.loc[missing_mask, "date"] = core_full.loc[missing_mask, "date"].fillna(
        core_full.loc[missing_mask, "filled_date"]
    )

    # Fill date-only missing using (channel,msg_id)
    date_only_mask = core_full["date"].isna() & core_full["channel"].notna()
    if date_only_mask.any():
        sub = core_full.loc[date_only_mask, ["channel", "msg_id"]].copy()
        sub["channel"] = sub["channel"].astype(str)
        sub["msg_id_norm"] = pd.to_numeric(sub["msg_id"], errors="coerce").astype("Int64").astype(str)
        keys = set(sub["channel"] + "|" + sub["msg_id_norm"])
        key_to_date: Dict[str, str] = {}
        reader = pd.read_csv(
            MEGA_RAW,
            compression="xz",
            chunksize=200_000,
            low_memory=False,
            usecols=["channel", "msg_id", "date"],
        )
        for chunk in reader:
            chunk["channel"] = chunk["channel"].astype(str)
            mid_norm = pd.to_numeric(chunk["msg_id"], errors="coerce").astype("Int64").astype(str)
            kser = chunk["channel"] + "|" + mid_norm
            mask = kser.isin(keys)
            if mask.any():
                for k, d in zip(kser[mask], chunk.loc[mask, "date"]):
                    key_to_date.setdefault(k, d)
            if len(key_to_date) == len(keys):
                break
        core_full["msg_id_norm"] = pd.to_numeric(core_full["msg_id"], errors="coerce").astype("Int64").astype(str)
        core_full["key"] = core_full["channel"].astype(str) + "|" + core_full["msg_id_norm"]
        core_full.loc[date_only_mask, "date"] = core_full.loc[date_only_mask, "key"].map(key_to_date)
        core_full.drop(columns=["msg_id_norm", "key"], inplace=True)

    core_full.to_csv(OUT_FILLED_EXACT, index=False)

    exact_summary = {
        "missing_rows": int(missing_mask.sum()),
        "matched_texts": len(exact_matches),
        "unique_channel_matches": len(fill_channel),
        "multi_channel_matches": sum(1 for v in fill_channels.values() if ";" in v),
        "filled_rows": int(core_full.loc[missing_mask, "channel"].notna().sum()),
        "remaining_missing_rows": int((core_full["channel"].isna() & core_full["date"].isna()).sum()),
    }
    OUT_EXACT_SUMMARY.write_text(json.dumps(exact_summary, indent=2))

    # -------------------------
    # 2) Fuzzy backfill for remaining
    # -------------------------
    remaining_mask = core_full["channel"].isna() & core_full["date"].isna()
    remaining_texts = core_full.loc[remaining_mask, "message_original"].fillna("").astype(str)
    remaining_keys = set(remaining_texts.apply(norm_key_fuzzy))

    fuzzy_matches = scan_mega_for_keys(
        remaining_keys,
        norm_key_fuzzy,
        usecols=["channel", "msg_id", "date", "text"],
    )
    f_ch, f_dt, f_ct, f_chs = choose_unique_channel_fill(fuzzy_matches)

    core_full.loc[remaining_mask, "fuzzy_match_count"] = remaining_texts.apply(norm_key_fuzzy).map(f_ct)
    core_full.loc[remaining_mask, "fuzzy_match_channels"] = remaining_texts.apply(norm_key_fuzzy).map(f_chs)
    core_full.loc[remaining_mask, "fuzzy_filled_channel"] = remaining_texts.apply(norm_key_fuzzy).map(f_ch)
    core_full.loc[remaining_mask, "fuzzy_filled_date"] = remaining_texts.apply(norm_key_fuzzy).map(f_dt)

    core_full.loc[remaining_mask, "channel"] = core_full.loc[remaining_mask, "channel"].fillna(
        core_full.loc[remaining_mask, "fuzzy_filled_channel"]
    )
    core_full.loc[remaining_mask, "date"] = core_full.loc[remaining_mask, "date"].fillna(
        core_full.loc[remaining_mask, "fuzzy_filled_date"]
    )

    core_full.to_csv(OUT_FILLED_FUZZY, index=False)

    fuzzy_summary = {
        "missing_rows_before": int(remaining_mask.sum()),
        "fuzzy_matched_keys": len(fuzzy_matches),
        "fuzzy_unique_channel_keys": len(f_ch),
        "filled_rows_added": int(core_full.loc[remaining_mask, "channel"].notna().sum()),
        "remaining_missing_rows": int((core_full["channel"].isna() & core_full["date"].isna()).sum()),
    }
    OUT_FUZZY_SUMMARY.write_text(json.dumps(fuzzy_summary, indent=2))

    # -------------------------
    # 3) Codebook normalization
    # -------------------------
    norm_counts = defaultdict(Counter)

    reader = pd.read_csv(OUT_FILLED_FUZZY, low_memory=False, chunksize=100_000)
    first = True
    for chunk in reader:
        chunk["theme_cb"] = chunk["theme"].apply(
            lambda x: map_field(x, THEMES, THEME_SYNONYMS, KW_THEME, THEME_ORDER, "Other (Theme)")
        )
        chunk["claim_types_cb"] = chunk["claim_types"].apply(
            lambda x: map_field(x, CLAIMS, CLAIM_SYNONYMS, KW_CLAIM, CLAIM_ORDER, "Other (Claim type)")
        )
        chunk["ctas_cb"] = chunk["ctas"].apply(
            lambda x: map_field(x, CTAS, CTA_SYNONYMS, KW_CTA, CTA_ORDER, "No CTA", allow_other=False)
        )
        chunk["evidence_cb"] = chunk["evidence"].apply(
            lambda x: map_field(x, EVID, EVID_SYNONYMS, KW_EVID, EVID_ORDER, "Other (Evidence)")
        )

        chunk["claim_types_cb"] = chunk["claim_types_cb"].apply(enforce_claim_rules)
        chunk["ctas_cb"] = chunk["ctas_cb"].apply(enforce_cta_rules)
        chunk["evidence_cb"] = chunk["evidence_cb"].apply(enforce_evidence_rules)

        for col in ["theme_cb", "claim_types_cb", "ctas_cb", "evidence_cb"]:
            norm_counts[col].update(chunk[col].fillna("MISSING").astype(str))

        chunk.to_csv(OUT_FINAL, mode="w" if first else "a", header=first, index=False)
        first = False

    OUT_CODEBOOK_SUMMARY.write_text(json.dumps({c: v.most_common(30) for c, v in norm_counts.items()}, indent=2))

    print("Wrote final file:", OUT_FINAL)


if __name__ == "__main__":
    main()

