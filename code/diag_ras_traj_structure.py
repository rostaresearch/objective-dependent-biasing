"""Check whether the 24 trajectories in the npz are a CHAINED sequence
(end of traj k == start of traj k+1), in which case allow_cross_traj=True
is correctly reconstructing one continuous trajectory and is NOT
introducing artificial transitions."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanism_audit_ras_proper import NPZ

d = np.load(NPZ, allow_pickle=True)
trajs = d["data"]
print(f"  N trajs = {len(trajs)}")
print()
print("  k  name                        phase           iter   n_frames"
      "    cv[0]  cv[-1]")
print("  -- --------------------------- --------------- ----- ----------"
      "  ------- -------")
for k, t in enumerate(trajs):
    name  = t.get("name", "?")
    phase = t.get("phase", "?")
    it    = t.get("iter")
    n     = len(t["cv"])
    print(f"  {k:>2} {name[:27]:<27} {phase:<15} {str(it):>5}  {n:>9}"
          f"   {t['cv'][0]:>6.3f}  {t['cv'][-1]:>6.3f}")

print()
print("  Checking continuity:  cv[-1] of traj k  vs  cv[0] of traj k+1")
gaps = []
for k in range(len(trajs) - 1):
    g = abs(float(trajs[k]["cv"][-1]) - float(trajs[k+1]["cv"][0]))
    gaps.append(g)
print(f"  max gap in CV: {max(gaps):.3f}")
print(f"  median gap:    {np.median(gaps):.3f}")
print(f"  >0.5 jumps:    {sum(1 for g in gaps if g > 0.5)} / {len(gaps)}")
print(f"  >2.0 jumps:    {sum(1 for g in gaps if g > 2.0)} / {len(gaps)}")
