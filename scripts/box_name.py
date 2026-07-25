#!/usr/bin/env python3
"""Print this machine's FLEET name (box1, box2, ...) by matching its IPs
against fleet.yml, falling back to the hostname.

Exists so a box identifies itself with the same vocabulary the inventory and
every command use. Provider hostnames like 'srv1856364' are an accident of
purchase order, and a heartbeat filed under one is a translation step in the
middle of an incident.
"""
import os
import socket
import subprocess
import sys

try:
    import yaml
except ImportError:
    print(socket.gethostname())
    sys.exit(0)

FLEET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "fleet.yml")


def main():
    try:
        boxes = yaml.safe_load(open(FLEET))["boxes"]
    except Exception:
        print(socket.gethostname())
        return
    ips = set()
    try:
        ips |= set(subprocess.run(["hostname", "-I"], capture_output=True,
                                  text=True, timeout=5).stdout.split())
    except Exception:
        pass
    for b in boxes:
        if b.get("host", "").split("@")[-1] in ips:
            print(b["name"])
            return
    print(socket.gethostname())


if __name__ == "__main__":
    main()
