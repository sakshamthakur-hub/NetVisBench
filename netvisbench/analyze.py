"""Turn raw results into the paper's tables, figures and numbers.

  python -m netvisbench.analyze --results results --dataset data/netvisbench \\
      --paper paper

Writes paper/tables/*.tex, paper/figures/*.pdf and results/summary.json.
Every number in the paper comes from summary.json - nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import DIFFICULTIES, load_items  # noqa: E402

FAMILIES = ["topology", "subnet", "tcp_seq", "cwnd", "ipv4_header", "capture"]
FAM_LABEL = {"topology": "Topology", "subnet": "Subnet", "tcp_seq": "TCP seq.",
             "cwnd": "cwnd plot", "ipv4_header": "IPv4 hdr.", "capture": "Capture"}
COND_LABEL = {"blind": "Blind", "vision": "Vision", "sgd": "SG-DR", "text": "Text"}
COND_ORDER = ["blind", "vision", "sgd", "text"]
# validated categorical slots (light mode): blue, orange, aqua; grey for the baseline
COLORS = {"vision": "#2a78d6", "text": "#eb6834", "sgd": "#1baf7a", "blind": "#9a9893"}
INK, MUTED = "#0b0b0b", "#52514e"


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def chance(it) -> float:
    """Expected accuracy of uniform guessing among the visible options."""
    if it.answer_type == "choice":
        return 0.5
    if it.family == "topology" and it.subtype == "next_hop":
        return 1 / (it.meta["nodes"] - 1)
    if it.family == "subnet" and it.subtype == "misconfigured":
        return 1 / sum(1 for _ in it.text_rendering.split("=")[1:])
    return 0.0


# ---------------------------------------------------------------- loading
def load_results(res_dir: Path, include_mock: bool = False):
    data = defaultdict(dict)  # data[model][cond] -> {id: rec}
    for f in sorted(res_dir.glob("*/*.jsonl")):
        recs = [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
        if not recs or (recs[0]["model"].startswith("mock:") and not include_mock):
            continue
        model = recs[0]["model"]
        data[model][f.stem] = {r["id"]: r for r in recs}
    return data


def acc(recs, pred=lambda r: True):
    sel = [r for r in recs if pred(r)]
    k = sum(r["correct"] for r in sel)
    return k, len(sel)


PRETTY = {"gemma-4-31b-it": "Gemma 4 31B", "gemma-4-26b-a4b-it": "Gemma 4 26B-A4B",
          "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
          "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite"}
FAM_MACRO = {"topology": "Topology", "subnet": "Subnet", "tcp_seq": "Tcpseq", "cwnd": "Cwnd",
             "ipv4_header": "Ipvfour", "capture": "Capture"}  # TeX macro names: letters only


def pretty_model(m: str) -> str:
    mid = m.split(":", 1)[1]
    return PRETTY.get(mid, mid)


# ---------------------------------------------------------------- tables
def pct(k, n):
    return "--" if n == 0 else f"{100 * k / n:.1f}"


def table_dataset(items, path: Path):
    rows = []
    for fam in FAMILIES:
        its = [i for i in items if i.family == fam]
        subs = sorted({i.subtype.replace("_", r"\_") for i in its})
        rows.append(f"{FAM_LABEL[fam]} & {', '.join(subs)} & {len(its)} \\\\")
    body = "\n".join(rows)
    path.write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{NetVisBench task families. Every answer is "
        "computed by code.}\n\\label{tab:dataset}\n\\small\n"
        "\\begin{tabular}{@{}llr@{}}\n\\toprule\nFamily & Question variants & Items \\\\\n"
        f"\\midrule\n{body}\n\\midrule\nTotal & & {len(items)} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table_main(data, items, path: Path, summary):
    """Vision accuracy per family, plus overall accuracy per condition."""
    ch_overall = 100 * sum(chance(i) for i in items) / len(items)
    head = " & ".join(FAM_LABEL[f] for f in FAMILIES)
    lines = [f"Chance & " + " & ".join(
        f"{100 * sum(chance(i) for i in items if i.family == f) / max(1, sum(1 for i in items if i.family == f)):.1f}"
        for f in FAMILIES) + f" & {ch_overall:.1f} \\\\", "\\midrule"]
    for m in sorted(data):
        for cond in COND_ORDER:
            if cond not in data[m]:
                continue
            recs = list(data[m][cond].values())
            cells = [pct(*acc(recs, lambda r, f=f: r["family"] == f)) for f in FAMILIES]
            k, n = acc(recs)
            label = f"\\quad {COND_LABEL[cond]}"
            lines.append(f"{label} & " + " & ".join(cells) + f" & \\textbf{{{pct(k, n)}}} \\\\")
        lines.insert(len(lines) - sum(c in data[m] for c in COND_ORDER),
                     f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{pretty_model(m)}}}}} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    path.write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Accuracy (\\%) by task family and input "
        "condition. \\emph{Blind}: question only. \\emph{Vision}: question + figure. "
        "\\emph{SG-DR}: schema-guided describe-then-reason pipeline (figure only). "
        "\\emph{Text}: question + the same facts written as text (upper reference).}\n"
        "\\label{tab:main}\n\\small\n\\begin{tabular}{@{}l" + "r" * (len(FAMILIES) + 1) + "@{}}\n"
        f"\\toprule\nModel / condition & {head} & Overall \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\end{tabular}\n\\end{table*}\n")
    summary["chance_overall"] = round(ch_overall, 1)


def table_errors(data, path: Path, summary):
    """Split vision errors into perception vs reasoning using the text condition."""
    rows = []
    for m in sorted(data):
        if not {"vision", "text"} <= set(data[m]):
            continue
        v, t = data[m]["vision"], data[m]["text"]
        ids = set(v) & set(t)
        wrong = [i for i in ids if not v[i]["correct"]]
        perc = sum(1 for i in wrong if t[i]["correct"])
        reas = len(wrong) - perc
        b = sum(1 for i in ids if v[i]["correct"] and not t[i]["correct"])
        c = sum(1 for i in ids if not v[i]["correct"] and t[i]["correct"])
        p = mcnemar_exact(b, c)
        s = summary["models"][m]
        s["vision_errors"] = len(wrong)
        s["perception_errors"] = perc
        s["reasoning_errors"] = reas
        s["perception_share"] = round(100 * perc / max(1, len(wrong)), 1)
        s["p_vision_vs_text"] = p
        rows.append(f"{pretty_model(m)} & {len(wrong)} & {perc} ({s['perception_share']:.0f}\\%)"
                    f" & {reas} & {_p(p)} \\\\")
    path.write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Where do vision errors come from? An error is "
        "a \\emph{perception} error if the model answers the same item correctly from text, "
        "otherwise a \\emph{reasoning} error. $p$: exact McNemar test, vision vs.\\ text.}\n"
        "\\label{tab:errors}\n\\small\n\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
        "Model & Errors & Perception & Reasoning & $p$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table_expert(data, items, path: Path, summary):
    """Vision vs. text on the expert level, for every model that ran it."""
    exp_ids = {i.id for i in items if i.difficulty == "expert"}
    rows, out = [], {}
    for m in sorted(data):
        v = {k: r for k, r in data[m].get("vision", {}).items() if k in exp_ids}
        t = {k: r for k, r in data[m].get("text", {}).items() if k in exp_ids}
        ids = set(v) & set(t)
        if len(ids) < len(exp_ids):
            continue
        kv = sum(v[i]["correct"] for i in ids)
        kt = sum(t[i]["correct"] for i in ids)
        b = sum(1 for i in ids if v[i]["correct"] and not t[i]["correct"])
        c = sum(1 for i in ids if not v[i]["correct"] and t[i]["correct"])
        p = mcnemar_exact(b, c)
        n = len(ids)
        fam_v = {f: _ratio(acc([v[i] for i in ids], lambda r, f=f: r["family"] == f))
                 for f in FAMILIES}
        fam_t = {f: _ratio(acc([t[i] for i in ids], lambda r, f=f: r["family"] == f))
                 for f in FAMILIES}
        out[m] = {"n": n, "vision": round(100 * kv / n, 1), "text": round(100 * kt / n, 1),
                  "gap": round(100 * (kt - kv) / n, 1), "p": p, "vision_only": b, "text_only": c,
                  "by_family_vision": {f: round(100 * x, 1) for f, x in fam_v.items()},
                  "by_family_text": {f: round(100 * x, 1) for f, x in fam_t.items()}}
        rows.append(f"{pretty_model(m)} & {100 * kv / n:.1f} & {100 * kt / n:.1f} & "
                    f"{100 * (kt - kv) / n:+.1f} & {_p(p)} \\\\")
    summary["expert"] = out
    path.write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Expert level (" + str(len(exp_ids)) +
        " items): accuracy (\\%) from the figure vs.\\ from the same facts as text. "
        "Gap = text $-$ vision. $p$: exact McNemar test.}\n\\label{tab:expert}\n\\small\n"
        "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nModel & Vision & Text & Gap & $p$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def _p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def table_cost(data, path: Path, summary):
    rows = []
    for m in sorted(data):
        cells = []
        for cond in ("vision", "sgd"):
            recs = list(data[m].get(cond, {}).values())
            if recs:
                tok = sum(total_gen_tokens(r) for r in recs) / len(recs)
                lat = sum(r["latency_s"] for r in recs) / len(recs)
                cells += [f"{tok:,.0f}", f"{lat:.1f}"]
                summary["models"][m][f"{cond}_tokens_per_item"] = round(tok)
                summary["models"][m][f"{cond}_latency_s"] = round(lat, 2)
            else:
                cells += ["--", "--"]
        rows.append(f"{pretty_model(m)} & " + " & ".join(cells) + " \\\\")
    path.write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Average cost per question: generated tokens "
        "(hidden reasoning + answer, summed over both calls for SG-DR) and latency in seconds.}\n\\label{tab:cost}\n\\small\n"
        "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n & \\multicolumn{2}{c}{Vision} & "
        "\\multicolumn{2}{c}{SG-DR} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
        "Model & Tokens & Sec. & Tokens & Sec. \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ---------------------------------------------------------------- token budget
BUDGETS = [512, 1024, 2048, 4096, 8192, 16384]


def gen_tokens(r) -> int:
    """Generated tokens (hidden thinking + visible output) of the LARGEST single
    call behind a record. The provider's max-output limit applies per call, so a
    record would survive a budget B exactly when this number is <= B."""
    total = r.get("thinking_tokens", 0) + r.get("output_tokens", 0)
    if r["condition"] != "sgd":
        return total
    t_out = r.get("transcript_tokens", [0, 0])[1] + r.get("transcript_thinking_tokens", 0)
    return max(t_out, total - t_out)


def total_gen_tokens(r) -> int:
    return r.get("thinking_tokens", 0) + r.get("output_tokens", 0)


def acc_at_budget(recs, budget: int) -> float:
    recs = list(recs)
    ok = sum(1 for r in recs if r["correct"] and gen_tokens(r) <= budget)
    return ok / len(recs) if recs else 0.0


def budget_summary(full, summary):
    for m, d in full.items():
        s = summary["models"][m]
        wrong = [r for r in d["vision"].values() if not r["correct"]]
        dup = [r for r in wrong if r["subtype"] == "dup_count"]
        s["dup_bias"] = {"errors": len(dup),
                         "said_three": sum(1 for r in dup if r["pred"] == "3"),
                         "tcp_errors": sum(1 for r in wrong if r["family"] == "tcp_seq")}
        s["budget"] = {c: {str(b): round(100 * acc_at_budget(d[c].values(), b), 1)
                           for b in BUDGETS} for c in ("vision", "text", "sgd")}
        for c in ("vision", "text", "sgd", "blind"):  # exact overall mean (not a mean of rounded means)
            s[f"gen_mean_{c}"] = round(_mean([total_gen_tokens(r) for r in d[c].values()]))
        s["gen_tokens"] = {
            c: {lvl: round(_mean([total_gen_tokens(r) for r in d[c].values()
                                  if r["difficulty"] == lvl]))
                for lvl in DIFFICULTIES} for c in ("vision", "text", "sgd", "blind")}
        s["no_answer"] = {c: sum(1 for r in d[c].values() if r.get("no_answer"))
                          for c in d}
        sg = list(d["sgd"].values())
        cut = [r for r in sg if r.get("finish") == "MAX_TOKENS"]
        s["sgd_errors"] = {"total": sum(1 for r in sg if not r["correct"]),
                           "cut": len(cut),
                           "cut_tcp": sum(1 for r in cut if r["family"] == "tcp_seq"),
                           "wrong_answers": sum(1 for r in sg if not r["correct"]
                                                and r.get("finish") != "MAX_TOKENS")}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def fig_budget(full, path: Path):
    models = sorted(full)
    fig, axes = plt.subplots(1, len(models), figsize=(3.5, 2.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for c in ("text", "sgd", "vision"):
            ys = [100 * acc_at_budget(full[m][c].values(), b) for b in BUDGETS]
            ax.plot(range(len(BUDGETS)), ys, marker="o", ms=3.5, lw=1.8, color=COLORS[c],
                    label=COND_LABEL[c])
        ax.set_xticks(range(len(BUDGETS)))
        ax.set_xticklabels([f"{b // 1024}k" if b >= 1024 else "0.5k" for b in BUDGETS],
                           fontsize=6.5)
        ax.set_title(pretty_model(m), fontsize=7.5, color=INK, loc="left", pad=2)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Output-token budget", fontsize=7, color=MUTED)
        _style(ax)
    axes[0, 0].set_ylabel("Accuracy (%)", fontsize=7, color=MUTED)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def fig_tokens(full, path: Path):
    """Mean generated tokens per question by difficulty: figure vs. text input."""
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    styles = {m: ls for m, ls in zip(sorted(full), ("-", "--", ":"))}
    for m in sorted(full):
        for c in ("vision", "text"):
            ys = [_mean([total_gen_tokens(r) for r in full[m][c].values()
                         if r["difficulty"] == d]) for d in DIFFICULTIES]
            ax.plot(range(len(DIFFICULTIES)), ys, marker="o", ms=3.5, lw=1.8,
                    color=COLORS[c], ls=styles[m],
                    label=f"{COND_LABEL[c]} - {pretty_model(m)}")
    ax.set_xticks(range(len(DIFFICULTIES)))
    ax.set_xticklabels([d.capitalize() for d in DIFFICULTIES], fontsize=7)
    ax.set_ylabel("Generated tokens / question", fontsize=7, color=MUTED)
    ax.legend(fontsize=5.8, frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def fig_family(data, path: Path):
    """Small multiples: one panel per model, grouped bars per family by condition."""
    models = sorted(data)
    conds = [c for c in ("vision", "sgd", "text") if any(c in data[m] for m in models)]
    fig, axes = plt.subplots(len(models), 1, figsize=(3.5, 1.55 * len(models) + 0.4),
                             sharex=True, squeeze=False)
    w = 0.8 / len(conds)
    for ax, m in zip(axes[:, 0], models):
        for j, c in enumerate(conds):
            recs = list(data[m].get(c, {}).values())
            ys = [100 * _ratio(acc(recs, lambda r, f=f: r["family"] == f)) for f in FAMILIES]
            xs = [i + (j - (len(conds) - 1) / 2) * w for i in range(len(FAMILIES))]
            ax.bar(xs, ys, width=w * 0.92, color=COLORS[c], label=COND_LABEL[c], lw=0)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel("Acc. (%)", fontsize=7, color=MUTED)
        ax.set_title(pretty_model(m), fontsize=8, color=INK, loc="left", pad=2)
        _style(ax)
    axes[-1, 0].set_xticks(range(len(FAMILIES)))
    axes[-1, 0].set_xticklabels([FAM_LABEL[f] for f in FAMILIES], fontsize=7, rotation=20)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(conds), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def _ratio(kn):
    k, n = kn
    return k / n if n else 0.0


def fig_difficulty(data, path: Path):
    """Accuracy vs difficulty per condition, averaged over models."""
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    lows = []
    for c in ("text", "sgd", "vision"):
        ys = []
        for d in DIFFICULTIES:
            vals = [_ratio(acc(data[m][c].values(), lambda r: r["difficulty"] == d))
                    for m in data if c in data[m]]
            ys.append(100 * sum(vals) / len(vals) if vals else float("nan"))
        if all(math.isnan(y) for y in ys):
            continue
        lows.append(min(ys))
        ax.plot(range(len(DIFFICULTIES)), ys, marker="o", ms=4, lw=2, color=COLORS[c],
                label=COND_LABEL[c])
    ax.set_xticks(range(len(DIFFICULTIES)))
    ax.set_xticklabels([d.capitalize() for d in DIFFICULTIES], fontsize=7)
    lo = max(0, 5 * math.floor((min(lows) - 2) / 5))
    ax.set_ylim(lo, 100.5)
    ax.set_xlim(-0.2, len(DIFFICULTIES) - 0.8)
    ax.set_ylabel("Mean accuracy (%)", fontsize=7, color=MUTED)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_examples(items, root: Path, path: Path):
    """One example image per family (for the paper's overview figure)."""
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.9))
    for ax, fam in zip(axes.flat, FAMILIES):
        it = next(i for i in items if i.family == fam and i.difficulty == "medium")
        ax.imshow(plt.imread(root / it.image))
        ax.set_title(FAM_LABEL[fam], fontsize=8, color=INK)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- decoding ablation
def table_greedy(full, greedy, path: Path, summary):
    """Same model, same items, same 16k budget: greedy (T=0) vs default sampling."""
    rows, out = [], {}
    for m in sorted(full):
        for c in ("vision", "text"):
            g = greedy.get(m, {}).get(c, {})
            d = full[m].get(c, {})
            ids = set(g) & set(d)
            if len(ids) < 100:
                continue
            n = len(ids)
            ag = 100 * sum(g[i]["correct"] for i in ids) / n
            ad = 100 * sum(d[i]["correct"] for i in ids) / n
            ng = 100 * sum(1 for i in ids if g[i].get("finish") == "MAX_TOKENS") / n
            nd = 100 * sum(1 for i in ids if d[i].get("finish") == "MAX_TOKENS") / n
            tg = _mean([total_gen_tokens(g[i]) for i in ids])
            td = _mean([total_gen_tokens(d[i]) for i in ids])
            out[f"{m}|{c}"] = {"n": n, "acc_greedy": round(ag, 1), "acc_default": round(ad, 1),
                               "cut_greedy": round(ng, 1), "cut_default": round(nd, 1),
                               "tok_greedy": round(tg), "tok_default": round(td)}
            rows.append(f"{pretty_model(m)} & {COND_LABEL[c]} & {n} & {ag:.1f} & {ad:.1f} & "
                        f"{ng:.1f} & {nd:.1f} \\\\")
    summary["greedy_ablation"] = out
    path.write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Decoding matters. The same items under "
        "greedy decoding (temperature 0) and the provider's default sampling, both with a "
        "16k-token budget. \\emph{Cut}: replies stopped by the token limit before giving an "
        "answer (endless reasoning loops). Greedy text runs were stopped early, so $n$ < 720 "
        "for those rows.}\n\\label{tab:greedy}\n\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n & & & \\multicolumn{2}{c}{Accuracy (\\%)} & "
        "\\multicolumn{2}{c}{Cut (\\%)} \\\\\n\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\n"
        "Model & Input & $n$ & Greedy & Default & Greedy & Default \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ---------------------------------------------------------------- macros
def write_macros(summary, items, path: Path):
    """LaTeX macros so the paper text quotes numbers straight from the results."""
    L = ["% generated by netvisbench.analyze - do not edit by hand"]

    def mac(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    def f1(x):
        return f"{x:.1f}"

    mac("NItems", len(items))
    mac("NPerLevel", sum(1 for i in items if i.difficulty == "expert"))
    if "chance_overall" in summary:
        mac("Chance", f1(summary["chance_overall"]))
    full = summary.get("full_models", [])
    tags = "ABCDEFG"
    for tag, m in zip(tags, full):
        s = summary["models"][m]
        mac(f"Name{tag}", pretty_model(m))
        for c, cname in (("vision", "Vis"), ("text", "Text"), ("blind", "Blind"), ("sgd", "Sgd")):
            mac(f"{cname}{tag}", f1(s[c]["acc"]))
            for d in DIFFICULTIES:
                mac(f"{cname}{d.capitalize()}{tag}", f1(s[c]["by_difficulty"][d]))
            for fam in FAMILIES:
                mac(f"{cname}{FAM_MACRO[fam]}{tag}", f1(s[c]["by_family"][fam]))
        mac(f"VisCILo{tag}", f1(s["vision"]["ci"][0]))
        mac(f"VisCIHi{tag}", f1(s["vision"]["ci"][1]))
        mac(f"PercShare{tag}", f1(s.get("perception_share", 0)))
        mac(f"PercErr{tag}", s.get("perception_errors", 0))
        mac(f"ReasErr{tag}", s.get("reasoning_errors", 0))
        mac(f"VisErr{tag}", s.get("vision_errors", 0))
        mac(f"NoAns{tag}", f1(s["vision"]["no_answer_rate"]))
        mac(f"SgdWins{tag}", s.get("sgd_wins", 0))
        mac(f"SgdLosses{tag}", s.get("sgd_losses", 0))
        mac(f"PVisSgd{tag}", _p(s.get("p_vision_vs_sgd", 1.0)))
        mac(f"PVisText{tag}", _p(s.get("p_vision_vs_text", 1.0)))
        mac(f"VisTok{tag}", f"{s.get('vision_tokens_per_item', 0):,}")
        if "budget" in s:
            for c, cname in (("vision", "Vis"), ("text", "Text"), ("sgd", "Sgd")):
                for b, bname in ((1024, "One"), (2048, "Two"), (4096, "Four"), (8192, "Eight")):
                    mac(f"Bud{cname}{bname}{tag}", f1(s["budget"][c][str(b)]))
                gt = s["gen_tokens"][c]
                mac(f"Gen{cname}{tag}", f"{s[f'gen_mean_{c}']:,}")
                for d in DIFFICULTIES:
                    mac(f"Gen{cname}{d.capitalize()}{tag}", f"{gt[d]:,}")
        if "sgd_errors" in s:
            e = s["sgd_errors"]
            mac(f"SgdErr{tag}", e["total"])
            mac(f"SgdCut{tag}", e["cut"])
            mac(f"SgdCutTcp{tag}", e["cut_tcp"])
            mac(f"SgdWrong{tag}", e["wrong_answers"])
        if "dup_bias" in s:
            mac(f"DupErr{tag}", s["dup_bias"]["errors"])
            mac(f"DupThree{tag}", s["dup_bias"]["said_three"])
            mac(f"TcpErr{tag}", s["dup_bias"]["tcp_errors"])
        mac(f"SgdTok{tag}", f"{s.get('sgd_tokens_per_item', 0):,}")
    if full:
        for c, cname in (("vision", "Vis"), ("text", "Text"), ("sgd", "Sgd"), ("blind", "Blind")):
            for d in DIFFICULTIES:
                vals = [summary["models"][m][c]["by_difficulty"][d] for m in full]
                mac(f"Mean{cname}{d.capitalize()}", f1(sum(vals) / len(vals)))
    for j, (m, e) in enumerate(sorted(summary.get("expert", {}).items())):
        t = "XYZW"[j]
        mac(f"ExpName{t}", pretty_model(m))
        mac(f"ExpVis{t}", f1(e["vision"]))
        mac(f"ExpText{t}", f1(e["text"]))
        mac(f"ExpGap{t}", f1(e["gap"]))
        mac(f"ExpP{t}", _p(e["p"]))
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--dataset", default="data/netvisbench", type=Path)
    ap.add_argument("--paper", default="paper", type=Path)
    ap.add_argument("--include-mock", action="store_true", help="for pipeline tests")
    ap.add_argument("--greedy", type=Path, default=Path("results_greedy"),
                    help="results of the greedy-decoding ablation (optional)")
    a = ap.parse_args()
    items = load_items(a.dataset)
    data = load_results(a.results, a.include_mock)
    (a.paper / "tables").mkdir(parents=True, exist_ok=True)
    (a.paper / "figures").mkdir(parents=True, exist_ok=True)
    summary = {"n_items": len(items), "models": {}}
    for m in data:
        s = summary["models"][m] = {}
        for c, recs in data[m].items():
            k, n = acc(recs.values())
            lo, hi = wilson(k, n)
            s[c] = {"acc": round(100 * k / n, 1), "n": n, "ci": [round(100 * lo, 1),
                                                                 round(100 * hi, 1)],
                    "by_family": {f: round(100 * _ratio(acc(recs.values(),
                                                            lambda r, f=f: r["family"] == f)), 1)
                                  for f in FAMILIES},
                    "by_difficulty": {d: round(100 * _ratio(acc(recs.values(),
                                                                lambda r, d=d: r["difficulty"] == d)), 1)
                                      for d in DIFFICULTIES},
                    "answer_tag_rate": round(100 * sum(r["tagged"] for r in recs.values()) / n, 1),
                    "no_answer_rate": round(100 * sum(r.get("no_answer", False)
                                                      for r in recs.values()) / n, 1)}
        if {"vision", "sgd"} <= set(data[m]):
            v, g = data[m]["vision"], data[m]["sgd"]
            ids = set(v) & set(g)
            b = sum(1 for i in ids if v[i]["correct"] and not g[i]["correct"])
            c = sum(1 for i in ids if not v[i]["correct"] and g[i]["correct"])
            s["p_vision_vs_sgd"] = mcnemar_exact(b, c)
            s["sgd_wins"], s["sgd_losses"] = c, b
    table_dataset(items, a.paper / "tables" / "dataset.tex")
    fig_examples(items, a.dataset, a.paper / "figures" / "examples.pdf")
    # "full" models were run on every item in every condition; partial ones
    # (free-tier daily caps) only on a subset and are reported separately.
    full = {m: d for m, d in data.items()
            if set(COND_ORDER) <= set(d) and all(len(d[c]) == len(items) for c in COND_ORDER)}
    summary["full_models"] = sorted(full)
    for m in data:
        if m not in full:
            print(f"note: {m} is partial: " +
                  ", ".join(f"{c}={len(v)}" for c, v in data[m].items()))
    if full:
        table_main(full, items, a.paper / "tables" / "main.tex", summary)
        table_errors(full, a.paper / "tables" / "errors.tex", summary)
        table_cost(full, a.paper / "tables" / "cost.tex", summary)
        fig_family(full, a.paper / "figures" / "family.pdf")
        fig_difficulty(full, a.paper / "figures" / "difficulty.pdf")
        budget_summary(full, summary)
        fig_budget(full, a.paper / "figures" / "budget.pdf")
        fig_tokens(full, a.paper / "figures" / "tokens.pdf")
        if a.greedy.exists():
            table_greedy(full, load_results(a.greedy), a.paper / "tables" / "greedy.tex", summary)
    if data:
        table_expert(data, items, a.paper / "tables" / "expert.tex", summary)
    write_macros(summary, items, a.paper / "numbers.tex")
    out = a.results / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:3000])


if __name__ == "__main__":
    main()
