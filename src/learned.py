"""
Learned eviction: a decision tree scores each resident page; highest score is evicted.
Features per resident page (all computable online):
    recency (t - last access), frequency (accesses since load), age (t - load time),
    rate (frequency / age).
Label (from Belady hindsight, training only): 1 if the page's next use is at or beyond
the median next-use distance of the resident set, i.e. it is a 'good eviction candidate'.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from optimal import next_use_table
def features(r, t):
    age = t - r["load"]
    return [t - r["last"], r["freq"], age, r["freq"] / (age + 1)]


def collect_samples(trace, frames):
    """Replay Belady on 'trace'; at every eviction record (features, label) per resident."""
    nxt, resident, X, y = next_use_table(trace), {}, [], []
    for t, p in enumerate(trace):
        if p in resident:
            r = resident[p]
            r["last"], r["freq"], r["next"] = t, r["freq"] + 1, nxt[t]
            continue
        if len(resident) >= frames:
            pages = list(resident)
            nu = np.array([resident[q]["next"] for q in pages])
            thr = np.median(nu)
            for q, n in zip(pages, nu):
                X.append(features(resident[q], t))
                y.append(int(n >= thr))
            del resident[pages[int(np.argmax(nu))]]
        resident[p] = {"last": t, "freq": 1, "load": t, "next": nxt[t]}
    return np.array(X), np.array(y)


def train(trace, frames, max_depth=5):
    X, y = collect_samples(trace, frames)
    if len(set(y)) < 2:
        return None
    return DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)


def simulate(trace, frames, model, adaptive=False, window=2000):
    """adaptive=True: every 'window' accesses, refit the tree on the window just seen
    (labels via hindsight inside that window). Static model otherwise."""
    resident, hits = {}, []
    for t, p in enumerate(trace):
        if adaptive and t > 0 and t % window == 0:
            m = train(trace[t - window:t], frames)
            if m is not None:
                model = m
        if p in resident:
            r = resident[p]
            r["last"], r["freq"] = t, r["freq"] + 1
            hits.append(True)
            continue
        hits.append(False)
        if len(resident) >= frames:
            pages = list(resident)
            if model is None:                              # fallback: LRU
                victim = min(pages, key=lambda q: resident[q]["last"])
            else:
                X = np.array([features(resident[q], t) for q in pages])
                proba = model.predict_proba(X)
                score = proba[:, 1] if proba.shape[1] == 2 else np.zeros(len(pages))
                score = score + 1e-6 * X[:, 0]
                victim = pages[int(np.argmax(score))]
            del resident[victim]
        resident[p] = {"last": t, "freq": 1, "load": t}
    return hits
