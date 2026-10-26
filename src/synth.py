"""Generate SYNTHETIC 802.11 pcaps (with RadioTap RSSI) to test the pipeline.

This is a stand-in so the code can be developed and tested before the real
dataset / lab capture is available. It is NOT a substitute for real data: results
on it are expected to be unrealistically good and must not go on the poster as
the headline numbers.

Writes data/synth/*.pcap and data/synth/manifest.json (ground truth for labels).
"""
import argparse
import json
import os

import numpy as np
from scapy.all import (Dot11, Dot11Beacon, Dot11Deauth, Dot11Elt, Dot11ProbeReq,
                       Raw, RadioTap, wrpcap)

AP = "aa:aa:aa:00:00:01"
BCAST = "ff:ff:ff:ff:ff:ff"
CLIENTS = [f"c0:ff:ee:00:00:{i:02x}" for i in range(1, 7)]
VISITOR = "5a:5a:5a:00:00:77"      # benign device present in every session
ATTACKER = "de:ad:be:ef:00:66"


def mk(t, typ, sub, src, dst, rssi, plen=0):
    d = Dot11(type=typ, subtype=sub, addr1=dst, addr2=src, addr3=AP)
    if typ == 0 and sub == 12:
        body = Dot11Deauth(reason=7)
    elif typ == 0 and sub == 8:
        body = Dot11Beacon() / Dot11Elt(ID=0, info=b"LabAP")
    elif typ == 0 and sub == 4:
        body = Dot11ProbeReq() / Dot11Elt(ID=0, info=b"")
    else:
        body = Raw(b"\x00" * plen)
    # `present=` must name the field, or scapy silently drops the RSSI value
    p = RadioTap(present="dBm_AntSignal", dBm_AntSignal=int(round(rssi))) / d / body
    p.time = float(t)
    return p


def session(rng, t0, dur, attack_rate):
    pk = []
    # AP beacons ~10/s
    t = t0
    while t < t0 + dur:
        pk.append(mk(t, 0, 8, AP, BCAST, -40 + rng.normal(0, 1.5)))
        t += 0.1024
    active = list(rng.choice(CLIENTS, size=rng.integers(3, 6), replace=False)) + [VISITOR]
    heavy = rng.choice(active) if rng.random() < 0.5 else None  # a streaming device
    for c in active:
        rate = rng.uniform(120, 200) if c == heavy else rng.choice([4, 8, 15, 25, 40])
        rc = rng.uniform(-70, -45)
        for s in range(dur):
            for ts in t0 + s + rng.random(rng.poisson(rate)):          # uplink
                pk.append(mk(ts, 2, 0, c, AP, rc + rng.normal(0, 2), int(rng.integers(60, 1200))))
            for ts in t0 + s + rng.random(rng.poisson(rate * 0.8)):    # downlink
                pk.append(mk(ts, 2, 0, AP, c, -40 + rng.normal(0, 1.5), int(rng.integers(60, 1400))))
            for ts in t0 + s + rng.random(rng.poisson(0.3)):           # probe requests
                pk.append(mk(ts, 0, 4, c, BCAST, rc + rng.normal(0, 2)))
    # a few legitimate single deauths from the AP (clients leaving)
    for _ in range(rng.integers(2, 4)):
        pk.append(mk(t0 + rng.uniform(2, dur - 2), 0, 12, AP, rng.choice(active), -40 + rng.normal(0, 1.5)))
    meta = {"attacker_macs": [], "attack_start": 0, "attack_end": 0}
    if attack_rate:
        a0, a1 = t0 + 10, t0 + dur - 5
        victim = rng.choice(CLIENTS)
        for s in range(int(a0 - t0), int(a1 - t0)):
            for ts in t0 + s + rng.random(rng.poisson(attack_rate)):
                dst = victim if rng.random() < 0.7 else BCAST
                pk.append(mk(ts, 0, 12, ATTACKER, dst, -38 + rng.normal(0, 3)))
        meta = {"attacker_macs": [ATTACKER], "attack_start": a0, "attack_end": a1}
    pk.sort(key=lambda p: p.time)
    return pk, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--dur", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rates = [0, 0, 0, 0, 0, 0, 0, 0, 25, 60, 150, 300, 100, 40, 200, 80]  # 8 normal + 8 attack
    manifest = []
    for i, r in enumerate(rates):
        name = f"{'attack' if r else 'normal'}_{i:02d}"
        pk, meta = session(rng, 1_700_000_000 + i * 1000, a.dur, r)
        path = os.path.join(a.out, name + ".pcap")
        wrpcap(path, pk)
        manifest.append({"name": name, "pcap": path, **meta})
        print(f"{name}: {len(pk)} frames, attack_rate={r}")
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump({"sessions": manifest}, f, indent=2)


if __name__ == "__main__":
    main()
