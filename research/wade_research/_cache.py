"""Exact, bounded CDF cache owned by one immutable kernel instance."""
from collections import OrderedDict

import numpy as np


class KernelNodeCache:
    """Cache exact CDF/primitive columns for one gene, with a pinned base mesh.

    Base nodes recur in every permutation bank. Pin them to avoid an LRU scan
    evicting the next bank's useful entries. Remaining capacity holds exact
    refinement nodes. Large cohorts that cannot fit the base mesh bypass the
    cache. Returned arrays are copies; group centering cannot mutate storage.
    """

    def __init__(self, nodes, states, max_bytes):
        self.base_node_count = len(nodes)
        self.states = int(states)
        self.max_bytes = int(max_bytes)
        self.base_bytes = 2 * self.states * len(nodes) * 8 + 9 * len(nodes)
        self.enabled = self.base_bytes + 1024 <= self.max_bytes
        self.nodes = np.asarray(nodes, float).copy() if self.enabled else np.empty(0)
        self.hits = self.misses = self.evictions = 0
        self.extra = OrderedDict()
        self.base = np.empty((2, states, len(nodes))) if self.enabled else None
        self.valid = np.zeros(len(nodes), bool) if self.enabled else None
        # Conservative allowance for per-column Python/NumPy/dictionary overhead.
        self.entry_bytes = 16 * self.states + 512
        self.extra_limit = max(0, (max_bytes-self.base_bytes-1024)//self.entry_bytes) if self.enabled else 0

    def get(self, x, evaluate):
        x = np.asarray(x, float)
        if not self.enabled:
            self.misses += len(x)
            f, put, _ = evaluate(x)
            return f, put
        index = np.searchsorted(self.nodes, x)
        base_node = index < len(self.nodes)
        base_node[base_node] &= self.nodes[index[base_node]] == x[base_node]
        result = np.empty((2, self.states, len(x)))
        missing = []
        for i, value in enumerate(x):
            if base_node[i] and self.valid[index[i]]:
                result[:, :, i] = self.base[:, :, index[i]]
                self.hits += 1
            elif not base_node[i] and float(value) in self.extra:
                result[:, :, i] = self.extra[float(value)]
                self.extra.move_to_end(float(value))
                self.hits += 1
            else:
                missing.append(i)
                self.misses += 1
        if missing:
            f, put, _ = evaluate(x[missing])
            for j, i in enumerate(missing):
                result[0, :, i] = f[:, j]
                result[1, :, i] = put[:, j]
                if base_node[i]:
                    self.base[:, :, index[i]] = result[:, :, i]
                    self.valid[index[i]] = True
                elif self.extra_limit:
                    self.extra[float(x[i])] = result[:, :, i].copy()
                    if len(self.extra) > self.extra_limit:
                        self.extra.popitem(last=False)
                        self.evictions += 1
        return result[0], result[1]

    def stats(self):
        used = self.base_bytes + 1024 + len(self.extra)*self.entry_bytes if self.enabled else 0
        return dict(enabled=self.enabled, hits=self.hits, misses=self.misses,
                    evictions=self.evictions, base_nodes=self.base_node_count,
                    extra_nodes=len(self.extra), extra_limit=self.extra_limit,
                    estimated_retained_bytes=used, limit_bytes=self.max_bytes)
