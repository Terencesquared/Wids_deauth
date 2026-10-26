# Wi-Fi Intrusion Detection System (WIDS) with optional WIPS-lite

Detects 802.11 **deauthentication flood** attacks using a scikit-learn Random Forest
on simple per-device, per-second features. With `--wips` it also blocks the offending MAC.

**Ethics / scope:** all capture and attack generation must be on your own or an approved
isolated lab network. Do not run attacks against networks you do not own or have written
permission to test.

## Pipeline

```
pcap / live monitor-mode capture
  -> features.py   1-second windows per transmitter MAC:
                   fps, deauth_count, deauth_ratio, uniq_dst,
                   rssi_mean, rssi_std, mgmt_ratio, len_mean
  -> train.py      Decision Tree + Random Forest, split BY CAPTURE SESSION
  -> detect.py     live or replay; alert + cooldown
  -> prevent.py    optional: append MAC to blocklist.txt (never blocks --protect MACs)
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# 0. (testing only) synthetic data - NOT for reported results
python src/synth.py

# 1. pcaps + manifest.json -> features.csv   (see docstring in build_dataset.py)
python src/build_dataset.py --manifest data/synth/manifest.json

# 2. train + evaluate -> model.joblib, results/*.png, results/metrics.json
python src/train.py

# 3a. replay a capture (demo fallback)
python src/detect.py --pcap data/synth/attack_11.pcap
# 3b. live, own lab only (root, adapter already in monitor mode on the AP's channel)
sudo python src/detect.py --iface wlan0mon
# 3c. WIPS-lite: also write blocklist.txt; never block the AP or your own adapter
sudo python src/detect.py --iface wlan0mon --wips --protect <AP_MAC>,<ADAPTER_MAC>
```

## Known limitations

- A spoofed source MAC (attacker forges the AP's address) defeats MAC-based blocking;
  that is why `--protect` exists. Detection still works on the deauth rate.
- Trained on a small dataset and one attack family; other attacks (disassociation
  floods, evil twin, beacon flood) are not covered.
- Python/scapy parsing handles roughly a couple of thousand frames/s, enough for a
  lab but not a busy environment.
- 802.11w (Protected Management Frames) prevents deauth spoofing at the source.

## AI assistance statement

(Fill in for the poster: which tool, what it was used for, and that you reviewed and
can explain the code.)
