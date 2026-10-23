#!/usr/bin/env bash
# Full reproduction: build the benchmark, evaluate models, make tables/figures, compile paper.
# Usage: OPENAI_API_KEY=... GEMINI_API_KEY=... ANTHROPIC_API_KEY=... scripts/run_all.sh model1 model2 ...
set -euo pipefail
cd "$(dirname "$0")/.."
python -m netvisbench.generate --out data/netvisbench --per-difficulty 30 --seed 2026
python -m pytest -q tests
python -m netvisbench.evaluate --dataset data/netvisbench --out results \
    --models "$@" --conditions blind vision text sgd --workers 6
python -m netvisbench.analyze --results results --dataset data/netvisbench --paper paper
(cd paper && latexmk -pdf -interaction=nonstopmode main.tex)
