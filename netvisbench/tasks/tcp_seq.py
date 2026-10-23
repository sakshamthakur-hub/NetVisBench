"""Task family 3: TCP sequence (time-line) diagrams.

The image shows a client and a server as two vertical time lines. The
client sends a window of data segments (labelled Seq/Len); one segment
is lost (marked with a red X). The server answers every segment that
arrives with a cumulative ACK, so segments after the loss trigger
duplicate ACKs. Finally the client retransmits the lost segment.

* ``cum_ack``   - what ACK number does the server send after the
  retransmission arrives? (requires understanding cumulative ACKs)
* ``dup_count`` - how many duplicate ACKs did the server send?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..common import Item, make_rng, save_fig

SIZES = {"easy": (4, False), "medium": (6, False), "hard": (8, True),
         "expert": (14, True)}  # (segments, varied length)
GAP, TRAVEL = 0.55, 1.3


def _scenario(rng, n, varied):
    isn = rng.choice([1, 101, 1001, 5001])
    lens = [rng.choice([100, 200, 300, 500]) if varied else 500 for _ in range(n)]
    segs, seq = [], isn
    for L in lens:
        segs.append((seq, L))
        seq += L
    # lost >= 1 so an "original" ACK exists before the duplicates;
    # lost <= n-2 so at least one segment (one duplicate ACK) follows the loss
    lost = rng.randint(1, n - 2)
    acks, expected = [], isn
    for i, (s, L) in enumerate(segs):
        if i == lost:
            acks.append(None)
            continue
        if s == expected:
            expected = s + L
        acks.append(expected)
    final_ack = segs[-1][0] + segs[-1][1]
    dup = n - 1 - lost
    return segs, lost, acks, final_ack, dup


def _draw(segs, lost, acks):
    n = len(segs)
    total = (n + 1) * GAP + 2 * TRAVEL + 1.2
    fig, ax = plt.subplots(figsize=(5.6, 0.62 * total + 0.8))
    cx, sx = 0.0, 4.0
    top = total
    for x, name in ((cx, "Client"), (sx, "Server")):
        ax.plot([x, x], [0, top], color="black", lw=1.5)
        ax.text(x, top + 0.25, name, ha="center", fontsize=11, weight="bold")
    arrow = dict(arrowstyle="-|>", lw=1.1, mutation_scale=11)
    for i, (s, L) in enumerate(segs):
        y0 = top - 0.4 - i * GAP
        y1 = y0 - TRAVEL
        if i == lost:
            xm, ym = (cx + sx) * 0.55, y0 - TRAVEL * 0.55
            ax.annotate("", xy=(xm, ym), xytext=(cx, y0), arrowprops=dict(arrow, color="#1f4e9a",
                                                                           arrowstyle="-"))
            ax.text(xm, ym, "X", color="#d00000", fontsize=15, weight="bold", ha="center",
                    va="center")
        else:
            ax.annotate("", xy=(sx, y1), xytext=(cx, y0), arrowprops=dict(arrow, color="#1f4e9a"))
            a = acks[i]
            ax.annotate("", xy=(cx, y1 - TRAVEL), xytext=(sx, y1),
                        arrowprops=dict(arrow, color="#2e7d32", ls="--"))
            ax.text(sx + 0.1, y1, f"ACK={a}", fontsize=8, color="#2e7d32", va="center")
        ax.text(cx - 0.1, y0, f"Seq={s}, Len={L}", fontsize=8, color="#1f4e9a", ha="right",
                va="center")
    # retransmission
    s, L = segs[lost]
    y0 = top - 0.4 - n * GAP - TRAVEL - 0.3
    y1 = y0 - TRAVEL
    ax.annotate("", xy=(sx, y1), xytext=(cx, y0), arrowprops=dict(arrow, color="#c45c00", lw=1.5))
    ax.text(cx - 0.1, y0, f"Seq={s}, Len={L}\n(retransmission)", fontsize=8, color="#c45c00",
            ha="right", va="center")
    ax.annotate("", xy=(cx, max(y1 - TRAVEL, 0.05)), xytext=(sx, y1),
                arrowprops=dict(arrow, color="#2e7d32", ls="--"))
    ax.text(sx + 0.1, y1, "ACK=?", fontsize=9, color="#2e7d32", weight="bold", va="center")
    ax.set_xlim(-2.4, 5.6)
    ax.set_ylim(-0.1, top + 0.6)
    ax.set_axis_off()
    return fig


def _text(segs, lost, acks):
    lines = ["Events in time order (client -> server data, server -> client ACKs):"]
    for i, (s, L) in enumerate(segs):
        if i == lost:
            lines.append(f"  Client sends Seq={s}, Len={L}: LOST (never arrives)")
        else:
            lines.append(f"  Client sends Seq={s}, Len={L}: arrives; server replies ACK={acks[i]}")
    s, L = segs[lost]
    lines.append(f"  Client retransmits Seq={s}, Len={L}: arrives; server replies ACK=?")
    return "\n".join(lines)


def generate(out_dir: Path, seed: int, per_difficulty: int) -> list[Item]:
    items: list[Item] = []
    for diff, (n, varied) in SIZES.items():
        for k in range(per_difficulty):
            rng = make_rng(seed, "tcp_seq", diff, k)
            segs, lost, acks, final_ack, dup = _scenario(rng, n, varied)
            iid = f"tcp_seq-{diff}-{k:03d}"
            subtype = "cum_ack" if k % 2 == 0 else "dup_count"
            if subtype == "cum_ack":
                q = ("The diagram is a TCP time-line between a client and a server. Data segments "
                     "are labelled with sequence number and length; a red X marks a lost segment. "
                     "The server uses cumulative ACKs and buffers out-of-order data. What ACK number "
                     "does the server send after the retransmitted segment arrives (the arrow "
                     "labelled ACK=?)? Answer with a single integer.")
                ans = str(final_ack)
            else:
                q = ("The diagram is a TCP time-line between a client and a server. A red X marks a "
                     "lost segment. How many DUPLICATE ACKs did the server send before the "
                     "retransmission (do not count the first ACK carrying that number)? "
                     "Answer with a single integer.")
                ans = str(dup)
            img = save_fig(_draw(segs, lost, acks), out_dir, iid)
            items.append(Item(id=iid, family="tcp_seq", subtype=subtype, difficulty=diff,
                              question=q, answer=ans, answer_type="int", image=img,
                              text_rendering=_text(segs, lost, acks),
                              meta={"segments": n, "lost_index": lost}))
    return items
