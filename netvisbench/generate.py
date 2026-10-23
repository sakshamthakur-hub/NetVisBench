"""Build the full benchmark: python -m netvisbench.generate --out data/netvisbench"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from .tasks import FAMILIES


def build(out: Path, seed: int, per_difficulty: int, families=None) -> list:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    items = []
    for name, mod in FAMILIES.items():
        if families and name not in families:
            continue
        items += mod.generate(out, seed, per_difficulty)
    ids = [it.id for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.to_json() + "\n")
    stats = {
        "seed": seed,
        "total": len(items),
        "by_family": Counter(i.family for i in items),
        "by_difficulty": Counter(i.difficulty for i in items),
        "by_subtype": Counter(f"{i.family}/{i.subtype}" for i in items),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/netvisbench")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--per-difficulty", type=int, default=30,
                    help="items per (family, difficulty); default 30 -> 540 items")
    ap.add_argument("--families", nargs="*", help="subset of families")
    a = ap.parse_args()
    items = build(Path(a.out), a.seed, a.per_difficulty, a.families)
    print(f"wrote {len(items)} items to {a.out}")


if __name__ == "__main__":
    main()
