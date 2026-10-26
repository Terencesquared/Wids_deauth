"""WIPS-lite: simple automatic response when the detector raises an alert.

What it does:  append the offending MAC to blocklist.txt (one MAC per line, so the
file can be used as a hostapd `deny_mac_file` on a lab AP you control) and write a
timestamped entry to blocklist.log.

What it deliberately does NOT do: send counter-attack frames. That is risky,
can affect other people's networks, and is unnecessary for a prototype.

Safety: MACs listed in `protect` (your AP, your monitor adapter, known good
clients) are never blocked, so a false positive or a spoofed source address
cannot make the system block your own AP.
"""
import datetime
import os


def _load(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {ln.strip().lower() for ln in f if ln.strip()}


def block(mac, reason, blocklist="blocklist.txt", log="blocklist.log", protect=()):
    """Returns 'blocked', 'already' or 'protected'."""
    mac = mac.lower()
    if mac in {p.lower() for p in protect}:
        return "protected"
    if mac in _load(blocklist):
        return "already"
    with open(blocklist, "a") as f:
        f.write(mac + "\n")
    with open(log, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{mac}\t{reason}\n")
    return "blocked"
