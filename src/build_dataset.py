"""pcap(s) + manifest.json  ->  data/features.csv (one row per 1 s window per MAC).

manifest.json lists one entry per capture session:
  {"sessions": [{"name": "attack_01", "pcap": "data/x.pcap",
                 "attacker_macs": ["de:ad:be:ef:00:66"],
                 "attack_start": 1700000010.0, "attack_end": 1700000025.0}, ...]}

A frame is labelled malicious if it was sent by a listed attacker MAC between
attack_start and attack_end (epoch seconds). Normal captures use an empty
attacker_macs list. For your own lab capture, take the attacker MAC and the
start/end times from your lab notebook.

The same pipeline works for any pcap (a subset of AWID3 included) as long as you
can say which MAC/time range is the attack.
"""
import argparse
import json

import pandas as pd

from features import frames_from_pcap, windowize


def label_frames(df, s):
    macs = [m.lower() for m in s.get("attacker_macs", [])]
    if not macs:
        df["mal"] = 0
    else:
        mal = (df["src"].isin(macs)
               & (df["t"] >= s["attack_start"])
               & (df["t"] <= s["attack_end"]))
        if s.get("deauth_only"):
            # Spoofed attack (e.g. aireplay-ng forges the AP's MAC as source): the AP also
            # sends legitimate beacons/data, so label only the deauth frames in the interval.
            mal &= (df["type"] == 0) & (df["subtype"] == 12)
        df["mal"] = mal.astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/synth/manifest.json")
    ap.add_argument("--out", default="data/features.csv")
    a = ap.parse_args()
    with open(a.manifest) as f:
        sessions = json.load(f)["sessions"]
    parts = []
    for s in sessions:
        frames = label_frames(frames_from_pcap(s["pcap"]), s)
        feats = windowize(frames)
        feats["session"] = s["name"]
        parts.append(feats)
        print(f"{s['name']}: {len(frames)} frames -> {len(feats)} windows, "
              f"{int(feats['label'].sum())} malicious")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"\nSaved {a.out}: {len(out)} rows, class counts:\n{out['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
