# CSE-307 Term Paper — Track 1: Learned Page Replacement

FIFO, LRU and Belady's Optimal page replacement, plus a decision-tree eviction policy
(static and adaptive), tested on a trace whose access pattern shifts at the midpoint.

## Setup
```
pip install -r requirements.txt
cd src && python run_experiments.py
```
Writes `results/results.csv`, `results/page_faults.png`, `results/hit_ratio.png`.

## Layout
- `src/fifo.py`, `lru.py`, `optimal.py` — classical policies (`simulate(trace, frames)`)
- `src/learned.py` — decision-tree eviction policy (features: recency, frequency, age, access rate)
- `src/workload.py` — trace generator: locality-heavy sequential loops + hot set (first half),
  uniform random with bursts (second half)
- `src/run_experiments.py` — runs 5 policies x 3 frame sizes (32/64/128) x 5 seeds

## Method summary
- Training labels come from Belady hindsight: a resident page is a "good eviction" if its
  next use is at/after the median next-use distance of the resident set.
- Training trace is pre-shift only (separate seed), so the static model never sees the new pattern.
- Adaptive variant refits the tree every 2000 accesses on the window just seen.
- Metrics: page faults and hit ratio for the first half ("before"), second half ("after") and overall.

## Results
See `results/` (fill in key findings from your own analysis here).

## AI assistance disclosure
ChatGPT was used for implementation help (code scaffolding for the simulators,
workload generator and plotting). Experimental design choices, results and analysis are my own.
