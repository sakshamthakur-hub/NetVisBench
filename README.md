# NetVisBench

**Can multimodal AI models read computer-networking figures?**

NetVisBench is a benchmark of 720 questions about the six kinds of figures that
networking students and engineers read every day:

| Family | What the figure shows | Example question |
|---|---|---|
| `topology` | routers and link costs | least-cost path cost / next hop |
| `subnet` | LANs with CIDR subnets and host IPs | which host is misconfigured? broadcast address? |
| `tcp_seq` | TCP time-line with a lost segment | cumulative ACK after retransmission, number of duplicate ACKs |
| `cwnd` | TCP Reno congestion-window plot | timeout or triple dup-ACK? ssthresh? number of losses? |
| `ipv4_header` | filled-in IPv4 header layout | payload length, fragment byte offset, MF flag |
| `capture` | packet list of interleaved TCP connections | number of connection attempts, which packet is a retransmission, refused port |

There are four difficulty levels (easy, medium, hard, expert) with 30 items per family
and level. Every figure is drawn by code and every answer is **computed** (never hand-labelled).
Each item also has a *text rendering* that holds exactly the same facts as the figure,
so we can test whether a model fails because it cannot **see** or because it cannot
**reason**.

## Quick start

```bash
pip install -r requirements.txt
python -m netvisbench.generate --out data/netvisbench   # 720 items, ~5 min
python -m pytest -q tests                               # includes the oracle check
export GEMINI_API_KEY=...                               # and/or OPENAI_API_KEY, ANTHROPIC_API_KEY
python -m netvisbench.evaluate --models gemini:gemini-3.5-flash-lite \
       --conditions blind vision text sgd
python -m netvisbench.analyze                           # tables + figures for the paper
```

Or run everything: `scripts/run_all.sh gemini:gemini-3.5-flash-lite openai:gpt-4o-mini`.

## Evaluation conditions

* **blind** - question only (how much can be guessed?)
* **vision** - question + figure (the normal setting)
* **text** - question + the same facts as text (upper reference: pure reasoning)
* **sgd** - *Schema-Guided Describe-then-Reason* (tested pipeline): the model first
  transcribes the figure with a short, figure-type-specific schema, then answers
  from its own transcription only.

## Code map

```
netvisbench/
  tasks/          one generator per figure family
  common.py       Item dataclass, rendering helpers
  oracle.py       independent solver that re-derives every answer from the text
  prompts.py      prompts for the four conditions
  models.py       minimal OpenAI / Anthropic / Gemini API clients (requests only)
  scoring.py      answer extraction and exact-match scoring
  evaluate.py     resumable, parallel evaluation runner
  analyze.py      statistics (Wilson CIs, McNemar), LaTeX tables, figures
tests/            pytest suite
paper/            IEEE-format LaTeX paper (Overleaf-ready)
```

## Reproducibility

* Fixed seed (2026); generation is deterministic (tested).
* `oracle.py` agrees with the stored answers on 100% of items (tested).
* Evaluation is resumable: re-running skips finished items.
* All paper numbers are produced by `analyze.py` into `results/summary.json`.

## Settings that matter (read before running thinking models)

* Use the provider's **default temperature** (the code does unless you set
  `NVB_TEMPERATURE`). Greedy decoding (0) traps thinking models in reasoning loops.
* Keep a large output budget (`NVB_MAX_TOKENS`, default 16384); hidden reasoning
  counts against it.
* On free API tiers set `NVB_RPM` (e.g. 27) so requests stay under the
  per-minute limit.
* The Gemini client streams responses so that long "thinking" requests are not
  dropped by idle-connection timeouts.

## Results in this release

`results/` holds every model reply (Gemma 4 31B and Gemma 4 26B-A4B, 720 items x 4
conditions). `results_greedy/` holds the greedy-decoding ablation. Run
`python -m netvisbench.analyze` to rebuild all tables, figures and
`results/summary.json`.
