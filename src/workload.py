"""Synthetic page-access trace with a deliberate mid-trace pattern shift."""
import random

NUM_PAGES = 200


def phase1_locality(n, rng):
    """Locality-heavy / sequential: loops over a sequential region + a small hot set.
    The region slowly drifts, so the pattern is stable but not frozen."""
    hot = list(range(8))                      
    region_size, base, trace = 48, 20, []
    while len(trace) < n:
        for p in range(base, base + region_size):   
            trace.append(p)
            if rng.random() < 0.4:                  
                trace.append(rng.choice(hot))
        if rng.random() < 0.3:                      
            base = min(base + 4, NUM_PAGES - region_size)
    return trace[:n]


def phase2_random_bursty(n, rng):
    """Random / bursty: uniform random pages, with short bursts on a single page."""
    trace = []
    while len(trace) < n:
        p = rng.randrange(NUM_PAGES)
        trace.extend([p] * (rng.randint(2, 6) if rng.random() < 0.3 else 1))
    return trace[:n]


def make_trace(length=20000, seed=0):
    """First half = phase 1, second half = phase 2. Returns (trace, shift_index)."""
    rng = random.Random(seed)
    half = length // 2
    return phase1_locality(half, rng) + phase2_random_bursty(length - half, rng), half


def make_training_trace(length=10000, seed=1000):
    """Training data comes from the PRE-shift pattern only (different seed), so the
    learned model has never seen the post-shift behaviour."""
    return phase1_locality(length, random.Random(seed))
