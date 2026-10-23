"""Task family 4: TCP Reno congestion-window plots.

We simulate TCP Reno per round-trip time (RTT): slow start doubles cwnd
until ssthresh, congestion avoidance adds 1 MSS per RTT. A loss detected
by triple duplicate ACK halves cwnd (fast recovery); a timeout resets cwnd
to 1 MSS. The image is the classic textbook "cwnd vs. transmission round"
plot. Question variants:

* ``loss_type`` - was the FIRST loss detected by timeout or by triple
  duplicate ACK? (multiple choice)
* ``ssthresh``  - value of ssthresh just after the first loss event.
* ``n_losses``  - how many loss events occur in the plot?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..common import Item, make_rng, save_fig

# (rounds, min and max number of loss events) - the count varies within a
# difficulty level so that it cannot be guessed from the plot's length
SIZES = {"easy": (16, (1, 2)), "medium": (26, (1, 3)), "hard": (36, (2, 4)),
         "expert": (60, (3, 6))}
CHOICES = ["A) timeout", "B) triple duplicate ACK"]


def simulate(rounds: int, ssthresh0: int, losses: dict[int, str]):
    """Return list of (round, cwnd) and the list of loss events applied."""
    cwnd, ssthresh = 1, ssthresh0
    trace, events = [], []
    for r in range(1, rounds + 1):
        trace.append((r, cwnd))
        if r in losses:
            kind = losses[r]
            ssthresh = max(cwnd // 2, 2)
            events.append({"round": r, "kind": kind, "cwnd": cwnd, "ssthresh": ssthresh})
            cwnd = 1 if kind == "TO" else ssthresh
        elif cwnd < ssthresh:
            cwnd = min(cwnd * 2, ssthresh)
        else:
            cwnd += 1
    return trace, events


def _draw(trace):
    xs, ys = zip(*trace)
    fig, ax = plt.subplots(figsize=(6.2 if len(trace) <= 40 else 8.4, 3.6))
    ax.plot(xs, ys, marker="o", ms=3.8, lw=1.6, color="#1f4e9a")
    ax.set_xlabel("Transmission round (RTT)")
    ax.set_ylabel("Congestion window (MSS)")
    ax.set_xticks(range(0, xs[-1] + 1, 2 if xs[-1] <= 40 else 4))
    step = 2 if max(ys) <= 24 else 4
    ax.set_yticks(range(0, max(ys) + step + 1, step))
    ax.grid(True, ls=":", alpha=0.7)
    ax.set_xlim(0, xs[-1] + 1)
    ax.set_ylim(0, max(ys) + 2)
    ax.set_title("TCP Reno congestion window")
    return fig


def _text(trace):
    pairs = ", ".join(f"r{r}:{c}" for r, c in trace)
    return ("TCP Reno congestion window (in MSS) at each transmission round:\n" + pairs)


def generate(out_dir: Path, seed: int, per_difficulty: int) -> list[Item]:
    items: list[Item] = []
    for diff, (rounds, (lo, hi)) in SIZES.items():
        for k in range(per_difficulty):
            rng = make_rng(seed, "cwnd", diff, k)
            n_loss = rng.randint(lo, hi)
            ssthresh0 = rng.choice([8, 16, 32]) if diff != "easy" else rng.choice([8, 16])
            # spread the loss events over the trace, at least 5 rounds apart
            losses: dict[int, str] = {}
            seg = (rounds - 6) // n_loss
            for j in range(n_loss):
                r = 6 + j * seg + rng.randint(0, max(seg - 6, 0))
                losses[min(r, rounds - 2)] = rng.choice(["TO", "TD"])
            trace, events = simulate(rounds, ssthresh0, losses)
            first = events[0]
            subtype = ("loss_type", "ssthresh", "n_losses")[k % 3]
            iid = f"cwnd-{diff}-{k:03d}"
            choices: list[str] = []
            if subtype == "loss_type":
                q = ("The plot shows the congestion window of a TCP Reno connection. Was the FIRST "
                     "loss event detected by a timeout or by a triple duplicate ACK? "
                     "Options: A) timeout, B) triple duplicate ACK. Answer with A or B.")
                ans, atype, choices = ("A" if first["kind"] == "TO" else "B"), "choice", CHOICES
            elif subtype == "ssthresh":
                q = ("The plot shows the congestion window of a TCP Reno connection. On a loss, "
                     "Reno sets ssthresh to half of the current window (rounded down). What is the "
                     "value of ssthresh (in MSS) right after the FIRST loss event? "
                     "Answer with a single integer.")
                ans, atype = str(first["ssthresh"]), "int"
            else:
                q = ("The plot shows the congestion window of a TCP Reno connection. How many loss "
                     "events (timeouts or triple duplicate ACKs) occur in this plot? "
                     "Answer with a single integer.")
                ans, atype = str(len(events)), "int"
            img = save_fig(_draw(trace), out_dir, iid)
            items.append(Item(id=iid, family="cwnd", subtype=subtype, difficulty=diff,
                              question=q, answer=ans, answer_type=atype, image=img,
                              text_rendering=_text(trace), choices=choices,
                              meta={"events": events, "ssthresh0": ssthresh0}))
    return items
