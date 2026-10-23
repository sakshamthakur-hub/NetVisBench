"""Independent symbolic solver ("oracle") for every task family.

The oracle reads ONLY the item's text rendering and the question - never
the stored answer or the generator's internal state - and solves the
question with ordinary networking logic. If it agrees with the stored
answer on 100% of items, that shows two things:

1. the ground truth is correct (two independent computations agree), and
2. the text rendering contains all the information needed, so the
   image-vs-text comparison in the paper is a fair one.
"""
from __future__ import annotations

import ipaddress
import re

import networkx as nx

from .common import Item


def _int(tok: str) -> int:
    return int(tok, 16) if tok.lower().startswith("0x") else int(tok)


def solve_topology(it: Item) -> str:
    g = nx.Graph()
    for u, v, w in re.findall(r"(\w)-(\w): (\d+)", it.text_rendering):
        g.add_edge(u, v, weight=int(w))
    s, d = re.findall(r"router (\w) (?:to|towards) router (\w)", it.question)[0]
    path = nx.shortest_path(g, s, d, weight="weight")
    if it.subtype == "path_cost":
        return str(nx.path_weight(g, path, weight="weight"))
    return path[1]


def solve_subnet(it: Item) -> str:
    lans = []
    for net, hosts in re.findall(r"subnet ([\d./]+)\): (.*)", it.text_rendering):
        hs = dict(h.split("=") for h in hosts.split(", "))
        lans.append((ipaddress.ip_network(net), hs))
    if it.subtype == "misconfigured":
        bad = [h for net, hs in lans for h, ip in hs.items()
               if ipaddress.ip_address(ip) not in net]
        assert len(bad) == 1
        return bad[0]
    host = re.findall(r"host (H\d+)", it.question)[0]
    return str(next(net.broadcast_address for net, hs in lans if host in hs))


def solve_tcp_seq(it: Item) -> str:
    segs = [(int(s), int(L), "LOST" in rest) for s, L, rest in
            re.findall(r"Client sends Seq=(\d+), Len=(\d+): (.*)", it.text_rendering)]
    if it.subtype == "cum_ack":
        return str(max(s + L for s, L, _ in segs))
    acks = [int(a) for a in re.findall(r"replies ACK=(\d+)", it.text_rendering)]
    # duplicates = ACKs equal to the previous ACK value
    return str(sum(1 for a, b in zip(acks, acks[1:]) if a == b))


def solve_cwnd(it: Item) -> str:
    trace = [int(c) for _, c in re.findall(r"r(\d+):(\d+)", it.text_rendering)]
    drops = [(prev, nxt) for prev, nxt in zip(trace, trace[1:]) if nxt < prev]
    if it.subtype == "n_losses":
        return str(len(drops))
    prev, nxt = drops[0]
    if it.subtype == "loss_type":
        return "A" if nxt == 1 else "B"
    return str(max(prev // 2, 2))


def solve_ipv4(it: Item) -> str:
    f = dict(re.findall(r"^  ([\w /+]+): (\S+)", it.text_rendering, flags=re.M))
    if "Version+IHL" in f:  # expert layout: unpack the combined fields
        f["IHL"] = str(_int(f["Version+IHL"]) & 0xF)
        word = _int(f["Flags+Fragment Offset"])
        f["Fragment Offset"] = str(word & 0x1FFF)
        f["Flags"] = format(word >> 13, "03b")
    if it.subtype == "payload_len":
        return str(_int(f["Total Length"]) - _int(f["IHL"]) * 4)
    if it.subtype == "frag_offset":
        return str(_int(f["Fragment Offset"]) * 8)
    flags = f["Flags"]
    mf = flags.startswith("MF=1") if "MF=" in flags else flags.endswith("1")
    return "A" if mf else "B"


def solve_capture(it: Item) -> str:
    rows = [line.split("\t") for line in it.text_rendering.splitlines()[2:]]
    infos = [(int(r[0]), r[6]) for r in rows]
    if it.subtype == "n_conns":
        return str(sum(1 for _, i in infos if "[SYN]" in i))
    if it.subtype == "refused":
        return next(i.split(" ")[0] for _, i in infos if "[RST" in i)
    seen = set()
    for no, info in infos:
        m = re.match(r"(\d+) → (\d+) \[(.*?)\] Seq=(\d+).*Len=(\d+)", info)
        key = (m.group(1), m.group(2), m.group(4), m.group(5))
        if int(m.group(5)) > 0 and key in seen:
            return str(no)
        seen.add(key)
    raise ValueError("no retransmission found")


SOLVERS = {"topology": solve_topology, "subnet": solve_subnet, "tcp_seq": solve_tcp_seq,
           "cwnd": solve_cwnd, "ipv4_header": solve_ipv4, "capture": solve_capture}


def solve(it: Item) -> str:
    return SOLVERS[it.family](it)
