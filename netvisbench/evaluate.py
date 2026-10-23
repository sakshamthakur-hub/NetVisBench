"""Run models on the benchmark.

Example:
  python -m netvisbench.evaluate --dataset data/netvisbench \\
      --models openai:gpt-4o-mini gemini:gemini-2.5-flash \\
      --conditions vision text blind sgd --workers 6

Results are appended to results/<model>/<condition>.jsonl, one line per
item. Re-running skips items that already have a result, so an
interrupted run can simply be restarted.
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import prompts
from .common import Item, load_items
from .models import Model, Reply, load_model
from .oracle import solve
from .scoring import extract_answer, is_correct, used_tag

CONDITIONS = ("vision", "text", "blind", "sgd")
_lock = threading.Lock()


def run_item(model: Model, it: Item, cond: str, root: Path) -> dict:
    img = str(root / it.image)
    rec = {"id": it.id, "family": it.family, "subtype": it.subtype,
           "difficulty": it.difficulty, "condition": cond, "model": model.name,
           "gold": it.answer, "answer_type": it.answer_type}
    if model.name == "mock:oracle":
        reply, extra = Reply(f"ANSWER: {solve(it)}"), {}
    elif cond == "vision":
        reply, extra = model(prompts.SYSTEM, prompts.vision_prompt(it.question), img), {}
    elif cond == "text":
        reply = model(prompts.SYSTEM, prompts.text_prompt(it.question, it.text_rendering))
        extra = {}
    elif cond == "blind":
        reply, extra = model(prompts.SYSTEM, prompts.blind_prompt(it.question)), {}
    elif cond == "sgd":
        tr = model(prompts.TRANSCRIBE_SYSTEM, prompts.transcribe_prompt(it.family), img)
        reply = model(prompts.SYSTEM,
                      prompts.answer_from_transcript_prompt(it.question, tr.text))
        extra = {"transcript": tr.text,
                 "transcript_tokens": [tr.input_tokens, tr.output_tokens]}
        extra["transcript_finish"] = tr.finish
        extra["transcript_thinking_tokens"] = tr.thinking_tokens
        reply = Reply(reply.text, reply.input_tokens + tr.input_tokens,
                      reply.output_tokens + tr.output_tokens, reply.latency_s + tr.latency_s,
                      reply.finish, reply.thinking_tokens + tr.thinking_tokens)
    else:
        raise ValueError(cond)
    pred = extract_answer(reply.text, it.answer_type)
    rec.update(pred=pred, correct=is_correct(pred, it.answer, it.answer_type),
               tagged=used_tag(reply.text), raw=reply.text,
               no_answer=not reply.text.strip(), finish=reply.finish,
               thinking_tokens=reply.thinking_tokens,
               input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
               latency_s=round(reply.latency_s, 2), **extra)
    return rec


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def evaluate(model_spec: str, dataset: Path, out: Path, conditions, workers: int,
             limit: int | None = None, families=None, difficulties=None) -> None:
    items = load_items(dataset)
    if families:
        items = [i for i in items if i.family in families]
    if difficulties:
        items = [i for i in items if i.difficulty in difficulties]
    if limit:
        items = items[:limit]
    model = load_model(model_spec)
    mdir = out / model.name.replace(":", "__").replace("/", "_")
    mdir.mkdir(parents=True, exist_ok=True)
    for cond in conditions:
        path = mdir / f"{cond}.jsonl"
        todo = [i for i in items if i.id not in _done_ids(path)]
        print(f"[{model.name}] {cond}: {len(todo)} items to run "
              f"({len(items) - len(todo)} cached)", flush=True)
        n_ok = n_done = n_err = 0
        with ThreadPoolExecutor(max_workers=workers) as ex, open(path, "a",
                                                                 encoding="utf-8") as f:
            futs = {ex.submit(run_item, model, it, cond, dataset): it for it in todo}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:  # keep going; failed items are retried next run
                    n_err += 1
                    print(f"  ! {futs[fut].id}: {str(e)[:200]}", flush=True)
                    continue
                with _lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                n_done += 1
                n_ok += rec["correct"]
                if n_done % 50 == 0:
                    print(f"  {n_done}/{len(todo)} done, running acc "
                          f"{n_ok / n_done:.3f}", flush=True)
        if n_done:
            print(f"  finished {cond}: acc {n_ok / n_done:.3f} on {n_done} new items, "
                  f"{n_err} errors", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/netvisbench", type=Path)
    ap.add_argument("--out", default="results", type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="only the first N items (for a smoke test)")
    ap.add_argument("--families", nargs="*")
    ap.add_argument("--difficulties", nargs="*", help="e.g. --difficulties expert")
    a = ap.parse_args()
    for m in a.models:
        evaluate(m, a.dataset, a.out, a.conditions, a.workers, a.limit, a.families,
                 a.difficulties)


if __name__ == "__main__":
    main()
