from collections import deque
def simulate(trace, frames):
    #Returns a list of booleans: True = hit, False = page fault
    queue, resident, hits = deque(), set(), []
    for p in trace:
        if p in resident:
            hits.append(True)
            continue
        hits.append(False)
        if len(queue) >= frames:
            resident.discard(queue.popleft())
        queue.append(p)
        resident.add(p)
    return hits
