"""Quick diagnostic: with allow_cross_traj=False, what's the visited
2-D grid look like, and how disconnected are A and B?"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanism_audit_ras_proper import (
    build_counts_2d, NPZ, _connected_via, _nn_2d_pairs,
)

d = np.load(NPZ, allow_pickle=True)
trajs = d["data"]

for label, allow in (("STRICT (no cross-traj)", False),
                     ("LOOSE  (allow cross-traj)", True)):
    print()
    print("=" * 64)
    print(f"  {label}")
    print("=" * 64)
    build = build_counts_2d(trajs, n_cv=12, n_f=8, lag=1,
                              allow_cross_traj=allow)
    n_cv, n_f, N = build["n_cv"], build["n_f"], build["N"]
    C = build["C_all"]
    visited = set(int(v) for v in np.where(C.sum(axis=1) > 0)[0])
    print(f"  visited: {len(visited)}/{N}")
    state_cv = build["state_cv"]
    A_set = [s for s in range(N) if state_cv[s] < 1.0 and s in visited]
    B_set = [s for s in range(N) if state_cv[s] > 8.0 and s in visited]
    print(f"  |A|={len(A_set)}, |B|={len(B_set)}")

    # render visited grid
    grid = np.zeros((n_cv, n_f), dtype=int)
    for s in visited:
        ic, ifv = divmod(s, n_f)
        grid[ic, ifv] = 1
    # mark A, B
    for s in A_set:
        ic, ifv = divmod(s, n_f); grid[ic, ifv] = 2
    for s in B_set:
        ic, ifv = divmod(s, n_f); grid[ic, ifv] = 3
    sym = {0: ".", 1: "#", 2: "A", 3: "B"}
    print("  visited grid (rows=cv low->high, cols=feat low->high):")
    print("       " + "".join(f"{j:>2}" for j in range(n_f)))
    for i in range(n_cv):
        print(f"   {i:>2}  " + "".join(f" {sym[grid[i,j]]}" for j in range(n_f)))

    # is A reachable from B via NON-zero C cells?
    print(f"  connected (any nonzero path): {_connected_via(C, A_set, B_set)}")

    # NN edges between visited cells
    pairs = _nn_2d_pairs(N, n_cv, n_f, np.array(sorted(visited)))
    print(f"  visited-NN edges (4-conn): {len(pairs)} pairs")

    # BFS from A on visited NN graph (treat as undirected)
    vis_arr = np.array(sorted(visited))
    nn = {v: [] for v in visited}
    for (i, j) in pairs:
        nn[i].append(j)
    seen = set(A_set); stk = list(A_set)
    while stk:
        u = stk.pop()
        for v in nn.get(u, []):
            if v not in seen:
                seen.add(v); stk.append(v)
    Bhit = set(B_set) & seen
    print(f"  reachable via NN: {len(seen & visited)}/{len(visited)}; "
          f"B states hit: {len(Bhit)}/{len(B_set)}")
