"""Check of the three Gaussian generators used by the kernel (moments, tails, KS distance)."""
import json, sys
import numpy as np
from numba import njit
from scipy import stats
import langevin
from common import HERE


@njit(cache=True)
def draw(n, noise, seed):
    s = np.zeros(4, np.uint64)
    langevin._seed(s, seed, 0)
    out = np.empty(2 * n)
    for k in range(n):
        a, b = langevin._normal_pair(s, noise)
        out[2 * k] = a
        out[2 * k + 1] = b
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25_000_000
    res = {}
    for name, code in [('boxmuller', 0), ('polar', 1), ('ziggurat', 2)]:
        z = draw(n, code, 7)
        ks = stats.kstest(z, 'norm')
        lag1 = float(np.corrcoef(z[:-1], z[1:])[0, 1])
        res[name] = dict(n=int(z.size), mean=float(z.mean()), var=float(z.var()), skew=float(stats.skew(z)),
                         excess_kurtosis=float(stats.kurtosis(z)), ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                         p_abs_gt3=float(np.mean(abs(z) > 3)), p_abs_gt3_exact=float(2 * stats.norm.sf(3)),
                         p_abs_gt4=float(np.mean(abs(z) > 4)), p_abs_gt4_exact=float(2 * stats.norm.sf(4)),
                         lag1_corr=lag1)
        print(name, res[name], flush=True)
    (HERE / 'test_normals.json').write_text(json.dumps(res, indent=1) + '\n')
