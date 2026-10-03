"""Runs all policies over several seeds / frame sizes; writes results/results.csv and charts."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fifo, lru, optimal, learned
from workload import make_trace, make_training_trace

FRAMES = [32, 64, 128]
SEEDS = range(5)
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def split_stats(hits, shift):
    for phase, h in (("before", hits[:shift]), ("after", hits[shift:]), ("overall", hits)):
        n, hit = len(h), sum(h)
        yield phase, n, n - hit, hit / n


rows = []
for f in FRAMES:
    model = learned.train(make_training_trace(), f)
    for seed in SEEDS:
        trace, shift = make_trace(seed=seed)
        runs = {
            "FIFO": fifo.simulate(trace, f),
            "LRU": lru.simulate(trace, f),
            "Optimal": optimal.simulate(trace, f),
            "Learned (static)": learned.simulate(trace, f, model),
            "Learned (adaptive)": learned.simulate(trace, f, model, adaptive=True),
        }
        for name, hits in runs.items():
            for phase, n, faults, ratio in split_stats(hits, shift):
                rows.append(dict(policy=name, frames=f, seed=seed, phase=phase,
                                accesses=n, page_faults=faults, hit_ratio=round(ratio, 4)))
    print("done frames =", f)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "results.csv"), index=False)

mean = df[df.phase != "overall"].groupby(["frames", "policy", "phase"])[["page_faults", "hit_ratio"]].mean()
order = ["FIFO", "LRU", "Optimal", "Learned (static)", "Learned (adaptive)"]
for metric, fname, label in (("page_faults", "page_faults.png", "Page faults (mean of 5 seeds)"),
                            ("hit_ratio", "hit_ratio.png", "Hit ratio (mean of 5 seeds)")):
    fig, axes = plt.subplots(1, len(FRAMES), figsize=(14, 4), sharey=True)
    for ax, f in zip(axes, FRAMES):
        m = mean.loc[f][metric].unstack("phase").loc[order]
        m[["before", "after"]].plot.bar(ax=ax, rot=30, width=0.75)
        ax.set_title(f"{f} frames"); ax.set_xlabel("")
    axes[0].set_ylabel(label)
    fig.suptitle(f"{label}: before vs after pattern shift")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)

print(mean.round(3).to_string())
