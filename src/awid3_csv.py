"""AWID3-CSV (one row per packet, Wireshark/tshark field names) -> window features.

Produces the same table as build_dataset.py, so train.py works unchanged:
    python src/awid3_csv.py --inspect  path/to/Deauth_chunk.csv      # look first!
    python src/awid3_csv.py --csv data/awid3/*.csv --out data/features.csv

Why --inspect first: column names vary between releases/re-uploads, so the loader
looks for several likely tshark names and tells you which ones it found.

Pseudo-sessions: a CSV chunk contains both normal and attack packets, so we cut each
file into consecutive blocks of --block-seconds and treat each block as a "session".
train.py then keeps whole blocks together in train OR test (a time-based split)
instead of mixing neighbouring seconds across both.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from features import windowize

# Candidate tshark column names, in order of preference.
CANDS = {
    "t": ["frame.time_epoch", "frame.time_relative", "frame.time"],
    "src": ["wlan.ta", "wlan.sa"],
    "dst": ["wlan.da", "wlan.ra"],
    "type_subtype": ["wlan.fc.type_subtype"],
    "type": ["wlan.fc.type"],
    "subtype": ["wlan.fc.subtype"],
    "rssi": ["radiotap.dbm_antsignal"],
    "len": ["frame.len"],
}
MAC_RE = r"([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})"
NUM_RE = r"(-?\d+(?:\.\d+)?)"


def header_of(path):
    return list(pd.read_csv(path, nrows=0).columns)


def find_columns(header):
    found = {k: [c for c in v if c in header] for k, v in CANDS.items()}
    lab = [c for c in header if c.strip().lower() == "label"]
    found["label"] = lab or [header[-1]]  # readme: one extra label column, normally last
    return found


def clean(s):
    s = s.astype("string").str.strip()
    return s.mask(s.isin(["", "-", "nan", "NaN", "<NA>"]))


def first_num(s):
    return pd.to_numeric(clean(s).str.extract(NUM_RE)[0], errors="coerce")


def coalesce(df, cols, fn):
    out = pd.Series(np.nan, index=df.index, dtype="float64") if fn is first_num else pd.Series(pd.NA, index=df.index, dtype="string")
    for c in cols:
        v = fn(df[c])
        out = out.where(out.notna(), v)
    return out


def hex_or_int(v):
    try:
        return int(str(v), 0)
    except (ValueError, TypeError):
        return np.nan


def load_frames(path, found):
    use = sorted({c for k, cols in found.items() for c in cols})
    raw = pd.read_csv(path, usecols=use, dtype=str, on_bad_lines="skip", low_memory=False)
    f = pd.DataFrame(index=raw.index)

    if "frame.time_epoch" in found["t"] or "frame.time_relative" in found["t"]:
        f["t"] = coalesce(raw, [c for c in found["t"] if c != "frame.time"], first_num)
    else:  # human-readable timestamp string
        f["t"] = pd.to_datetime(clean(raw[found["t"][0]]), errors="coerce").astype("int64") / 1e9
    f["src"] = coalesce(raw, found["src"], lambda s: clean(s).str.extract(MAC_RE)[0].str.lower())
    f["dst"] = coalesce(raw, found["dst"], lambda s: clean(s).str.extract(MAC_RE)[0].str.lower()).fillna("")

    ftype = coalesce(raw, found["type"], first_num) if found["type"] else pd.Series(np.nan, index=raw.index)
    fsub = coalesce(raw, found["subtype"], first_num) if found["subtype"] else pd.Series(np.nan, index=raw.index)
    if found["type_subtype"]:  # e.g. 0x000c = management(0) subtype 12 = deauthentication
        ts = clean(raw[found["type_subtype"][0]]).str.extract(r"(0x[0-9a-fA-F]+|\d+)")[0].map(hex_or_int)
        ftype = ftype.where(ftype.notna(), ts // 16)
        fsub = fsub.where(fsub.notna(), ts % 16)
    f["type"], f["subtype"] = ftype, fsub
    f["rssi"] = coalesce(raw, found["rssi"], first_num) if found["rssi"] else np.nan
    f["len"] = coalesce(raw, found["len"], first_num) if found["len"] else np.nan

    lab = clean(raw[found["label"][0]]).str.lower()
    f["label_text"] = lab
    f["mal"] = (lab.notna() & ~lab.isin(["normal", "0", "benign"])).astype(int)

    before = len(f)
    f = f.dropna(subset=["t", "src", "type", "subtype"]).sort_values("t")
    f[["type", "subtype"]] = f[["type", "subtype"]].astype(int)
    return f.reset_index(drop=True), before


def inspect(path):
    h = header_of(path)
    found = find_columns(h)
    print(f"{path}\n  {len(h)} columns")
    for k, v in found.items():
        print(f"  {k:13s} -> {v if v else 'NOT FOUND'}")
    missing = [k for k in ("t", "src", "type", "subtype") if not found[k] and not (k in ("type", "subtype") and found["type_subtype"])]
    f, before = load_frames(path, found)
    print(f"  rows read: {before}, usable after cleaning: {len(f)}")
    print("  label values:\n" + f["label_text"].value_counts(dropna=False).head(10).to_string())
    print(f"  frame types: management={int((f['type']==0).sum())} deauth(subtype 12)={int(((f['type']==0)&(f['subtype']==12)).sum())}")
    print(f"  time span: {f['t'].max()-f['t'].min():.1f} s, rssi present on {f['rssi'].notna().mean():.0%} of rows")
    if missing:
        print(f"  !! MISSING essential columns: {missing} - send me this output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", help="AWID3 CSV file(s); globs allowed")
    ap.add_argument("--inspect", nargs="+", help="just report columns/labels for these files")
    ap.add_argument("--out", default="data/features.csv")
    ap.add_argument("--block-seconds", type=int, default=30, help="pseudo-session length")
    a = ap.parse_args()

    if a.inspect:
        for p in [x for g in a.inspect for x in glob.glob(g)]:
            inspect(p)
        return

    parts = []
    for path in sorted(x for g in a.csv for x in glob.glob(g)):
        f, before = load_frames(path, find_columns(header_of(path)))
        feats = windowize(f)
        if feats.empty:
            print(f"{os.path.basename(path)}: no usable rows"); continue
        blk = (feats["win"] - feats["win"].min()) // a.block_seconds
        feats["session"] = os.path.basename(path) + "#" + blk.astype(str)
        parts.append(feats)
        print(f"{os.path.basename(path)}: {len(f)}/{before} rows -> {len(feats)} windows, {int(feats['label'].sum())} malicious")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"\nSaved {a.out}: {len(out)} rows, class counts:\n{out['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
