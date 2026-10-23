"""Task family 6: packet-capture listings (Wireshark-style packet list).

The image is a packet list table (No., Time, Source, Destination,
Protocol, Length, Info) of several interleaved TCP connections. Unlike a
real analyser, we do NOT print helper labels such as "[TCP
Retransmission]" - the model has to work it out from the fields.
Every capture contains exactly one retransmitted segment and exactly one
refused connection (SYN answered by RST). Variants:

* ``n_conns``  - how many TCP connection attempts (SYNs from clients)?
* ``retrans``  - which packet number is a retransmission?
* ``refused``  - which server port refused a connection?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..common import Item, make_rng, save_fig

SIZES = {"easy": (2, 3), "medium": (3, 4), "hard": (4, 6),
         "expert": (8, 10)}  # range of connection attempts
PORTS = [22, 80, 443, 3306, 5432, 8080, 8443]
EXTRA_PORTS = [21, 25, 53, 110, 143, 993, 1883, 6379]  # used only by expert captures
COLS = ["No.", "Time", "Source", "Destination", "Proto", "Len", "Info"]


def _conn_events(rng, cip, sip, cport, sport, refused, retrans):
    ev = []

    def add(src, dst, sp, dp, flags, seq, ack, ln, tag=""):
        info = f"{sp} → {dp} [{flags}] Seq={seq}" + (f" Ack={ack}" if ack is not None else "")
        info += f" Len={ln}"
        ev.append({"src": src, "dst": dst, "info": info, "len": 54 + ln, "tag": tag})

    add(cip, sip, cport, sport, "SYN", 0, None, 0)
    if refused:
        add(sip, cip, sport, cport, "RST, ACK", 1, 1, 0)
        return ev
    add(sip, cip, sport, cport, "SYN, ACK", 0, 1, 0)
    add(cip, sip, cport, sport, "ACK", 1, 1, 0)
    cseq, sseq = 1, 1
    n_data = rng.randint(2, 3)
    rt_at = rng.randrange(n_data) if retrans else -1
    for i in range(n_data):
        ln = rng.choice([120, 256, 517, 1024, 1448])
        add(cip, sip, cport, sport, "PSH, ACK", cseq, sseq, ln)
        if i == rt_at:  # the ACK is lost/delayed; client resends the same bytes
            add(cip, sip, cport, sport, "PSH, ACK", cseq, sseq, ln, tag="retrans")
        cseq += ln
        rln = rng.choice([0, 0, 300, 1448])
        add(sip, cip, sport, cport, "PSH, ACK" if rln else "ACK", sseq, cseq, rln)
        sseq += rln
    return ev


def _capture(rng, n_conns):
    conns, used_ports = [], set()
    refused_i = rng.randrange(n_conns)
    retrans_i = rng.choice([i for i in range(n_conns) if i != refused_i])
    pool = PORTS if n_conns <= len(PORTS) else PORTS + EXTRA_PORTS
    for i in range(n_conns):
        sport = rng.choice([p for p in pool if p not in used_ports])
        used_ports.add(sport)
        cip = f"192.168.1.{rng.randint(10, 60)}"
        sip = f"10.0.{rng.randint(0, 9)}.{rng.randint(2, 99)}"
        conns.append(_conn_events(rng, cip, sip, rng.randint(49152, 65535), sport,
                                  i == refused_i, i == retrans_i))
    # random interleaving that keeps each connection's own order
    rows, idx = [], [0] * n_conns
    while any(idx[i] < len(c) for i, c in enumerate(conns)):
        live = [i for i, c in enumerate(conns) if idx[i] < len(c)]
        i = rng.choice(live)
        rows.append(conns[i][idx[i]])
        idx[i] += 1
    t = 0.0
    for n, r in enumerate(rows, 1):
        t += rng.uniform(0.0004, 0.09)
        r["no"], r["time"] = n, f"{t:.6f}"
    refused_port = [r for r in rows if "RST" in r["info"]][0]["info"].split(" ")[0]
    retrans_no = [r["no"] for r in rows if r["tag"] == "retrans"][0]
    return rows, int(refused_port), retrans_no


def _cells(rows):
    return [[str(r["no"]), r["time"], r["src"], r["dst"], "TCP", str(r["len"]), r["info"]]
            for r in rows]


def _draw(rows):
    cells = _cells(rows)
    fig, ax = plt.subplots(figsize=(10.5, 0.27 * (len(rows) + 1) + 0.3))
    tab = ax.table(cellText=cells, colLabels=COLS, loc="center", cellLoc="left",
                   colLoc="left", colWidths=[0.05, 0.09, 0.12, 0.11, 0.06, 0.05, 0.52])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.25)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.get_text().set_fontfamily("monospace")
        if r == 0:
            cell.set_facecolor("#dfe6ee")
            cell.get_text().set_weight("bold")
        else:
            # neutral zebra stripes only: colour must not leak information
            # that the text rendering does not contain
            cell.set_facecolor("#f4f6f8" if r % 2 == 0 else "white")
    ax.set_axis_off()
    return fig


def _text(rows):
    lines = ["\t".join(COLS)]
    lines += ["\t".join(c) for c in _cells(rows)]
    return "Packet capture:\n" + "\n".join(lines)


def generate(out_dir: Path, seed: int, per_difficulty: int) -> list[Item]:
    items: list[Item] = []
    for diff, (lo, hi) in SIZES.items():
        for k in range(per_difficulty):
            rng = make_rng(seed, "capture", diff, k)
            n_conns = rng.randint(lo, hi)
            rows, refused_port, retrans_no = _capture(rng, n_conns)
            subtype = ("n_conns", "retrans", "refused")[k % 3]
            iid = f"capture-{diff}-{k:03d}"
            if subtype == "n_conns":
                q = ("The image shows a packet capture. How many TCP connection attempts were made "
                     "(count the initial SYN packets sent by clients, whether or not the connection "
                     "succeeded)? Answer with a single integer.")
                ans = str(n_conns)
            elif subtype == "retrans":
                q = ("The image shows a packet capture. Exactly one packet is a TCP retransmission "
                     "(it repeats data already sent: same connection, same Seq and Len). What is its "
                     "packet number (No. column)? Answer with a single integer.")
                ans = str(retrans_no)
            else:
                q = ("The image shows a packet capture. One connection attempt was refused by the "
                     "server. Which server TCP port refused it? Answer with a single integer.")
                ans = str(refused_port)
            img = save_fig(_draw(rows), out_dir, iid)
            items.append(Item(id=iid, family="capture", subtype=subtype, difficulty=diff,
                              question=q, answer=ans, answer_type="int", image=img,
                              text_rendering=_text(rows), meta={"rows": len(rows)}))
    return items
