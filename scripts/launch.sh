#!/usr/bin/env bash
# launch.sh <logname> <evaluate args...>  - start a resumable background evaluation
# (expects API keys in the environment, e.g. via `source keys.env`)
cd "$(dirname "$0")/.."
log="logs/$1.log"; shift
nohup python3 -m netvisbench.evaluate --out results "$@" >> "$log" 2>&1 &
