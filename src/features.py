"""Feature extraction shared by training AND live detection.

Using one code path for both means the detector sees exactly the same
features the model was trained on (no train/serve mismatch).

A "window" is 1 second of traffic. For every (window, transmitter MAC) pair we
compute a handful of simple features that describe how that device behaved.
"""
import numpy as np
import pandas as pd
from scapy.all import Dot11, PcapReader, RadioTap

WIN = 1.0  # seconds

# The feature columns fed to the classifier
FEATURES = [
    "fps",           # frames per second sent by this MAC
    "deauth_count",  # number of deauthentication frames sent in the window
    "deauth_ratio",  # deauth_count / total frames from this MAC
    "uniq_dst",      # number of distinct destination MACs it talked to
    "rssi_mean",     # average signal strength (dBm) as seen by the sniffer
    "rssi_std",      # variation in signal strength
    "mgmt_ratio",    # fraction of its frames that are management frames
    "len_mean",      # average frame length (bytes)
]


def packet_to_record(p):
    """Turn one scapy packet into a flat dict, or None if it is not usable."""
    if not p.haslayer(Dot11):
        return None
    d = p[Dot11]
    if d.addr2 is None:  # ACK/CTS frames have no transmitter address
        return None
    rt = p.getlayer(RadioTap)
    rssi = getattr(rt, "dBm_AntSignal", None) if rt is not None else None
    return {
        "t": float(p.time),
        "src": d.addr2.lower(),
        "dst": (d.addr1 or "").lower(),
        "type": int(d.type),        # 0 = management, 1 = control, 2 = data
        "subtype": int(d.subtype),  # for management: 12 = deauthentication
        "rssi": float(rssi) if rssi is not None else np.nan,
        "len": len(p),
    }


def frames_from_pcap(path):
    """Read a pcap into a DataFrame with one row per usable 802.11 frame."""
    rows = []
    with PcapReader(path) as reader:
        for p in reader:
            rec = packet_to_record(p)
            if rec is not None:
                rows.append(rec)
    return pd.DataFrame(rows)


def windowize(df, win=WIN):
    """Frame table -> one feature row per (time window, source MAC).

    If the frame table has a 0/1 column called `mal`, the window label is 1 when
    any frame in the window is malicious (used for training only).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["win", "src", "t_start", *FEATURES, "label"])
    df = df.copy()
    df["win"] = (df["t"] // win).astype(np.int64)
    df["is_deauth"] = ((df["type"] == 0) & (df["subtype"] == 12)).astype(int)
    df["is_mgmt"] = (df["type"] == 0).astype(int)
    if "mal" not in df:
        df["mal"] = 0
    out = (
        df.groupby(["win", "src"])
        .agg(
            n=("t", "size"),
            t_start=("t", "min"),
            deauth_count=("is_deauth", "sum"),
            uniq_dst=("dst", "nunique"),
            rssi_mean=("rssi", "mean"),
            rssi_std=("rssi", "std"),
            mgmt_ratio=("is_mgmt", "mean"),
            len_mean=("len", "mean"),
            label=("mal", "max"),
        )
        .reset_index()
    )
    out["fps"] = out["n"] / win
    out["deauth_ratio"] = out["deauth_count"] / out["n"]
    out["rssi_mean"] = out["rssi_mean"].fillna(-100.0)
    out["rssi_std"] = out["rssi_std"].fillna(0.0)
    return out.drop(columns=["n"])


class LiveWindower:
    """Feed frames one at a time; get a feature table back each time a window closes."""

    def __init__(self, win=WIN):
        self.win = win
        self.buf = []
        self.cur = None

    def add(self, rec):
        w = int(rec["t"] // self.win)
        out = None
        if self.cur is not None and w > self.cur:
            out = self.flush()
        if self.cur is None or w > self.cur:
            self.cur = w
        self.buf.append(rec)
        return out

    def flush(self):
        if not self.buf:
            return None
        feats = windowize(pd.DataFrame(self.buf), self.win)
        self.buf = []
        return feats
