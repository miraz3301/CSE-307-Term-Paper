INF = 10 ** 9
def next_use_table(trace):
    """nxt[i] = index of the next access to trace[i] after position i (INF if none)."""
    nxt, last_seen = [INF] * len(trace), {}
    for i in range(len(trace) - 1, -1, -1):
        nxt[i] = last_seen.get(trace[i], INF)
        last_seen[trace[i]] = i
    return nxt

def simulate(trace, frames):
    """Belady's optimal: evict the resident page whose next use is farthest away."""
    nxt, resident, hits = next_use_table(trace), {}, []
    for i, p in enumerate(trace):
        if p in resident:
            resident[p] = nxt[i]
            hits.append(True)
            continue
        hits.append(False)
        if len(resident) >= frames:
            del resident[max(resident, key=resident.get)]
        resident[p] = nxt[i]
    return hits
