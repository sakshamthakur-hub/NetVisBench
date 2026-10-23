"""Task family 5: IPv4 header diagrams.

The image is the familiar 32-bit-wide IPv4 header layout, filled with the
values of one concrete packet. Reading the right box is not enough: the
model must also apply header semantics. Variants:

* ``payload_len``  - payload bytes = Total Length - IHL * 4.
* ``frag_offset``  - byte offset of this fragment = Fragment Offset * 8.
* ``more_frags``   - is the MF (more fragments) flag set? (A yes / B no)

Difficulty controls the number format: decimal (easy), mixed decimal and
hex (medium), hex everywhere with binary flags (hard). Expert items draw the
header as a packet analyser's hex view would split it: Version+IHL is one
byte and Flags+Fragment Offset is one 16-bit word, so the model must pull
the bit fields out itself.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..common import Item, make_rng, save_fig

EXPERT_LAYOUT = [
    [("Version+IHL", 8), ("DSCP/ECN", 8), ("Total Length", 16)],
    [("Identification", 16), ("Flags+Fragment Offset", 16)],
    [("TTL", 8), ("Protocol", 8), ("Header Checksum", 16)],
    [("Source Address", 32)],
    [("Destination Address", 32)],
]
LAYOUT = [  # (field, bits) per 32-bit row
    [("Version", 4), ("IHL", 4), ("DSCP/ECN", 8), ("Total Length", 16)],
    [("Identification", 16), ("Flags", 3), ("Fragment Offset", 13)],
    [("TTL", 8), ("Protocol", 8), ("Header Checksum", 16)],
    [("Source Address", 32)],
    [("Destination Address", 32)],
]
EXPERT_NOTE = ("Version+IHL: high 4 bits Version, low 4 bits IHL.  Flags+Fragment Offset: "
               "top 3 bits [reserved, DF, MF], low 13 bits offset.")
CHOICES = ["A) yes, MF is set", "B) no, MF is not set"]


def _packet(rng):
    ihl = rng.choice([5, 5, 6, 7, 8])
    payload = rng.randint(20, 1460)
    mf = rng.random() < 0.5
    off = rng.randint(0, 180) if (mf or rng.random() < 0.6) else 0
    return {
        "Version": 4, "IHL": ihl, "DSCP/ECN": rng.choice([0, 0, 0x28, 0xB8]),
        "Total Length": ihl * 4 + payload, "Identification": rng.randint(1, 65535),
        "Flags": (0b001 if mf else 0b000), "Fragment Offset": off,
        "TTL": rng.choice([32, 63, 64, 127, 128, 255]), "Protocol": rng.choice([6, 17, 1]),
        "Header Checksum": rng.randint(0, 65535),
        "Source Address": f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
        "Destination Address": f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
    }


def _expert_values(pkt):
    v = dict(pkt)
    v["Version+IHL"] = (pkt["Version"] << 4) | pkt["IHL"]
    v["Flags+Fragment Offset"] = (pkt["Flags"] << 13) | pkt["Fragment Offset"]
    return v


def _layout(diff):
    return EXPERT_LAYOUT if diff == "expert" else LAYOUT


def _fmt(field, val, diff):
    if isinstance(val, str):
        return val
    if diff == "expert":
        width = 2 if field in ("Version+IHL", "DSCP/ECN", "TTL", "Protocol") else 4
        return f"0x{val:0{width}X}"
    if field == "Flags":
        return f"{val:03b}" if diff == "hard" else ("MF=1" if val & 1 else "MF=0") + ", DF=0"
    hexed = {"medium": {"Total Length", "Identification", "Header Checksum", "Fragment Offset"},
             "hard": {f for row in LAYOUT for f, _ in row}}.get(diff, {"Identification",
                                                                       "Header Checksum"})
    return f"0x{val:X}" if field in hexed else str(val)


def _draw(pkt, diff):
    layout = _layout(diff)
    if diff == "expert":
        pkt = _expert_values(pkt)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    unit = 1 / 32
    for r, row in enumerate(layout):
        x = 0.0
        y = len(LAYOUT) - 1 - r
        for field, bits in row:
            w = bits * unit
            ax.add_patch(Rectangle((x, y), w, 1, fc="#f7f9fc", ec="#333333", lw=1.2))
            ax.text(x + w / 2, y + 0.72, field, ha="center", fontsize=7.5, color="#444444")
            ax.text(x + w / 2, y + 0.32, _fmt(field, pkt[field], diff), ha="center",
                    fontsize=10, family="monospace", weight="bold")
            x += w
    for b in range(0, 33, 8):
        ax.text(b * unit, len(LAYOUT) + 0.12, str(b), ha="center", fontsize=7)
    if diff == "hard":
        ax.text(0.5, -0.35, "Flags bits: [reserved, DF, MF]", ha="center", fontsize=8)
    if diff == "expert":
        ax.text(0.5, -0.35, EXPERT_NOTE, ha="center", fontsize=7.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(LAYOUT) + 0.4)
    ax.set_title("IPv4 header", fontsize=10)
    ax.set_axis_off()
    return fig


def _text(pkt, diff):
    if diff == "expert":
        pkt = _expert_values(pkt)
    lines = ["IPv4 header fields:"]
    lines += [f"  {f}: {_fmt(f, pkt[f], diff)}" for row in _layout(diff) for f, _ in row]
    if diff == "hard":
        lines.append("  (Flags bits are [reserved, DF, MF])")
    if diff == "expert":
        lines.append(f"  ({EXPERT_NOTE})")
    return "\n".join(lines)


def generate(out_dir: Path, seed: int, per_difficulty: int) -> list[Item]:
    items: list[Item] = []
    for diff in ("easy", "medium", "hard", "expert"):
        for k in range(per_difficulty):
            rng = make_rng(seed, "ipv4", diff, k)
            pkt = _packet(rng)
            subtype = ("payload_len", "frag_offset", "more_frags")[k % 3]
            iid = f"ipv4_header-{diff}-{k:03d}"
            choices: list[str] = []
            if subtype == "payload_len":
                q = ("The diagram shows the IPv4 header of a packet. How many bytes of payload "
                     "(data after the IP header; the header and any options are not counted) does "
                     "this packet carry? Remember IHL counts 32-bit words. Answer with a single integer.")
                ans, atype = str(pkt["Total Length"] - pkt["IHL"] * 4), "int"
            elif subtype == "frag_offset":
                q = ("The diagram shows the IPv4 header of a packet. At what BYTE offset in the "
                     "original datagram does this fragment's data start? Remember the Fragment "
                     "Offset field counts 8-byte units. Answer with a single integer.")
                ans, atype = str(pkt["Fragment Offset"] * 8), "int"
            else:
                q = ("The diagram shows the IPv4 header of a packet. Is the More Fragments (MF) "
                     "flag set? Options: A) yes, MF is set, B) no, MF is not set. Answer A or B.")
                ans, atype, choices = ("A" if pkt["Flags"] & 1 else "B"), "choice", CHOICES
            img = save_fig(_draw(pkt, diff), out_dir, iid)
            items.append(Item(id=iid, family="ipv4_header", subtype=subtype, difficulty=diff,
                              question=q, answer=ans, answer_type=atype, image=img,
                              text_rendering=_text(pkt, diff), choices=choices,
                              meta={"packet": pkt}))
    return items
