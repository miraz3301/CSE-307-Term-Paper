from collections import OrderedDict


def simulate(trace, frames):
    """Returns a list of booleans: True = hit, False = page fault."""
    cache, hits = OrderedDict(), []
    for p in trace:
        if p in cache:
            cache.move_to_end(p)               
            hits.append(True)
            continue
        hits.append(False)
        if len(cache) >= frames:
            cache.popitem(last=False)          
        cache[p] = True
    return hits
