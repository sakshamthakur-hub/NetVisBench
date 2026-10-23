"""Task family 2: IP addressing on a LAN diagram.

The image shows a router connected to several LANs. Each LAN box is
labelled with its subnet (CIDR) and contains hosts labelled with their
IP addresses. Two question variants:

* ``misconfigured`` - exactly one host has an address outside its LAN's
  subnet; the model must name it. On medium, hard and expert items the wrong
  address sits just outside the subnet boundary, so you must actually do the maths.
* ``broadcast``     - give the broadcast address of the subnet that a
  named host belongs to.
"""
from __future__ import annotations

import ipaddress
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ..common import Item, make_rng, save_fig

# (number of LANs, hosts per LAN, allowed prefix lengths)
SIZES = {"easy": (2, 2, [24]), "medium": (3, 3, [25, 26]), "hard": (4, 3, [26, 27, 28]),
         "expert": (6, 4, [27, 28, 29])}


def _build(rng, n_lans, hosts_per, prefixes):
    base = ipaddress.ip_network(f"10.{rng.randint(1, 250)}.0.0/16")
    lans, used = [], set()
    while len(lans) < n_lans:
        p = rng.choice(prefixes)
        subs = list(base.subnets(new_prefix=p))
        net = rng.choice(subs[: min(len(subs), 64)])
        if any(net.overlaps(u) for u in used):
            continue
        used.add(net)
        lans.append(net)
    hosts, h = [], 1
    for li, net in enumerate(lans):
        pool = list(net.hosts())
        for ip in rng.sample(pool[1:], hosts_per):  # .1 reserved for the router
            hosts.append({"name": f"H{h}", "lan": li, "ip": ip})
            h += 1
    return lans, hosts


def _break_one(rng, lans, hosts, hard: bool):
    victim = rng.choice(hosts)
    net = lans[victim["lan"]]
    if hard:  # just past the broadcast address or just before the network address
        cand = [net.broadcast_address + rng.randint(1, 6), net.network_address - rng.randint(1, 6)]
    else:
        cand = [ipaddress.ip_address(int(net.network_address) + (1 << 16) + rng.randint(2, 50))]
    for ip in cand:
        if all(ip not in n for n in lans) and int(ip) & 0xFF not in (0, 255):
            victim["ip"] = ip
            return victim["name"]
    return None


def _draw(lans, hosts):
    n = len(lans)
    fig, ax = plt.subplots(figsize=(1.9 * n + 2.2, 4.2))
    width = 1.7 * n + 0.6 * (n - 1)
    ax.set_xlim(-0.3, width + 0.3)
    ax.set_ylim(-0.2, 4.4)
    rx, ry = width / 2, 3.8
    ax.add_patch(plt.Circle((rx, ry), 0.32, fc="#ffe2b8", ec="#a35a00", lw=1.5, zorder=3))
    ax.text(rx, ry, "R1", ha="center", va="center", fontsize=11, weight="bold", zorder=4)
    for li, net in enumerate(lans):
        x0 = li * 2.3
        ax.plot([rx, x0 + 0.85], [ry - 0.32, 2.95], color="#555555", lw=1.4, zorder=1)
        ax.add_patch(FancyBboxPatch((x0, 0.0), 1.7, 2.95, boxstyle="round,pad=0.04",
                                    fc="#f2f7ff", ec="#1f4e9a", lw=1.4))
        ax.text(x0 + 0.85, 2.7, f"LAN {chr(65 + li)}", ha="center", fontsize=10, weight="bold")
        ax.text(x0 + 0.85, 2.38, str(net), ha="center", fontsize=9.5, family="monospace",
                color="#1f4e9a")
        lan_hosts = [h for h in hosts if h["lan"] == li]
        top, step = (1.75, 0.62) if len(lan_hosts) <= 3 else (1.85, 0.5)
        for hi, h in enumerate(lan_hosts):
            y = top - hi * step
            ax.add_patch(FancyBboxPatch((x0 + 0.12, y - 0.22), 1.46, 0.48,
                                        boxstyle="round,pad=0.02", fc="white", ec="#777777"))
            ax.text(x0 + 0.85, y + 0.07, h["name"], ha="center", fontsize=9, weight="bold")
            ax.text(x0 + 0.85, y - 0.13, str(h["ip"]), ha="center", fontsize=8.5,
                    family="monospace")
    ax.set_axis_off()
    return fig


def _text(lans, hosts):
    lines = ["Router R1 connects the following LANs:"]
    for li, net in enumerate(lans):
        hs = ", ".join(f"{h['name']}={h['ip']}" for h in hosts if h["lan"] == li)
        lines.append(f"  LAN {chr(65 + li)} (subnet {net}): {hs}")
    return "\n".join(lines)


def generate(out_dir: Path, seed: int, per_difficulty: int) -> list[Item]:
    items: list[Item] = []
    for diff, (n, hp, prefixes) in SIZES.items():
        k, attempt = 0, 0
        while k < per_difficulty:
            attempt += 1
            rng = make_rng(seed, "subnet", diff, attempt)
            lans, hosts = _build(rng, n, hp, prefixes)
            subtype = "misconfigured" if k % 2 == 0 else "broadcast"
            iid = f"subnet-{diff}-{k:03d}"
            if subtype == "misconfigured":
                bad = _break_one(rng, lans, hosts, hard=(diff != "easy"))
                if bad is None:
                    continue
                q = ("The diagram shows LANs connected to router R1. Each LAN is labelled with its "
                     "subnet, and each host with its IP address. Exactly one host has an IP address "
                     "that does NOT belong to the subnet of its LAN. Which host is it? "
                     "Answer with the host name only (e.g. H3).")
                ans, atype = bad, "label"
            else:
                h = rng.choice(hosts)
                q = (f"The diagram shows LANs connected to router R1. Each LAN is labelled with its "
                     f"subnet. What is the broadcast address of the subnet that host {h['name']} "
                     f"belongs to? Answer with a dotted-decimal IPv4 address only.")
                ans, atype = str(lans[h["lan"]].broadcast_address), "label"
            img = save_fig(_draw(lans, hosts), out_dir, iid)
            items.append(Item(id=iid, family="subnet", subtype=subtype, difficulty=diff,
                              question=q, answer=ans, answer_type=atype, image=img,
                              text_rendering=_text(lans, hosts),
                              meta={"lans": [str(x) for x in lans]}))
            k += 1
    return items
