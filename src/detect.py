"""Wireless IDS (and optional WIPS-lite) - live capture or pcap replay.

  Replay (safe fallback for the demo):
      python src/detect.py --pcap data/synth/attack_11.pcap
      python src/detect.py --pcap capture.pcap --speed 1        # replay in real time

  Live (needs root, an adapter in monitor mode, and YOUR OWN authorised lab network):
      sudo python src/detect.py --iface wlan0mon

  Add --wips to also block alerting MACs (writes blocklist.txt), and
  --protect <AP_MAC>,<my_adapter_MAC> so those are never blocked.

How it works: frames -> 1-second windows per transmitter MAC -> the same features
used in training -> Random Forest -> alert if it says "deauth flood".
"""
import argparse
import datetime
import os
import time

import joblib

import prevent
from features import FEATURES, LiveWindower, PcapReader, packet_to_record


class Detector:
    def __init__(self, model_path, cooldown, wips, protect, log):
        self.clf = joblib.load(model_path)
        self.cooldown = cooldown
        self.wips = wips
        self.protect = [m.lower() for m in protect]
        self.log = log
        self.last_alert = {}
        self.stats = {"windows": 0, "alerts": 0, "blocked": 0}
        os.makedirs(os.path.dirname(log) or ".", exist_ok=True)

    def emit(self, line):
        print(line, flush=True)
        with open(self.log, "a") as f:
            f.write(line + "\n")

    def process(self, feats):
        if feats is None or feats.empty:
            return
        proba = self.clf.predict_proba(feats[FEATURES])[:, 1]
        for (_, r), p in zip(feats.iterrows(), proba):
            self.stats["windows"] += 1
            if p < 0.5:
                continue
            if r.t_start - self.last_alert.get(r.src, -1e12) < self.cooldown:
                continue  # one alert per attacker per cooldown period
            self.last_alert[r.src] = r.t_start
            self.stats["alerts"] += 1
            ts = datetime.datetime.fromtimestamp(r.t_start).strftime("%H:%M:%S")
            msg = (f"[ALERT {ts}] deauth flood from {r.src}: {int(r.deauth_count)} deauth frames/s "
                   f"({r.fps:.0f} fps, {r.uniq_dst:.0f} target(s), RSSI {r.rssi_mean:.0f} dBm, "
                   f"confidence {p:.2f})")
            if self.wips:
                action = prevent.block(r.src, f"deauth flood {int(r.deauth_count)}/s", protect=self.protect)
                if action == "blocked":
                    self.stats["blocked"] += 1
                msg += f" -> WIPS: {action}"
            self.emit(msg)

    def summary(self):
        s = self.stats
        print(f"\nDone. windows scored: {s['windows']}, alerts: {s['alerts']}, MACs blocked: {s['blocked']}")


def run_replay(det, path, speed):
    w = LiveWindower()
    last_t = None
    with PcapReader(path) as reader:
        for p in reader:
            rec = packet_to_record(p)
            if rec is None:
                continue
            if speed > 0 and last_t is not None and rec["t"] > last_t:
                time.sleep(min((rec["t"] - last_t) / speed, 1.0))
            last_t = rec["t"]
            det.process(w.add(rec))
    det.process(w.flush())


def run_live(det, iface):
    from scapy.all import sniff
    w = LiveWindower()
    print(f"Sniffing on {iface} (Ctrl+C to stop)...")

    def handle(p):
        rec = packet_to_record(p)
        if rec is not None:
            det.process(w.add(rec))

    try:
        sniff(iface=iface, prn=handle, store=False)
    except KeyboardInterrupt:
        pass
    det.process(w.flush())


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcap", help="replay a capture file")
    src.add_argument("--iface", help="live capture interface (monitor mode)")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--cooldown", type=float, default=5.0, help="seconds between alerts per MAC")
    ap.add_argument("--speed", type=float, default=0, help="replay speed (0 = as fast as possible, 1 = real time)")
    ap.add_argument("--wips", action="store_true", help="block alerting MACs (WIPS-lite)")
    ap.add_argument("--protect", default="", help="comma-separated MACs that must never be blocked")
    ap.add_argument("--log", default="results/alerts.log")
    a = ap.parse_args()
    protect = [m for m in a.protect.split(",") if m]
    det = Detector(a.model, a.cooldown, a.wips, protect, a.log)
    if a.pcap:
        run_replay(det, a.pcap, a.speed)
    else:
        run_live(det, a.iface)
    det.summary()


if __name__ == "__main__":
    main()
