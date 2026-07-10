import numpy as np


def empirical_pvalue(observed, null_values):
    """
    One-tailed empirical p-value testing whether `observed` is greater than expected
    under the null (e.g. a diffusion value computed on the real connectome vs. the same
    value recomputed on many randomized/null connectomes).

    observed: scalar statistic computed on the real data.
    null_values: array-like of the same statistic computed on each null replicate.

    Uses the North et al. (2002) bias-corrected estimator, p = (1 + #{null >= observed})
    / (1 + n), so p is never exactly 0 and the smallest representable p-value is
    1 / (n + 1) — i.e. resolution is limited by how many null replicates you have.
    """
    null_values = np.asarray(null_values, dtype=np.float64)
    count_ge = np.count_nonzero(null_values >= observed)
    return (1.0 + count_ge) / (1.0 + len(null_values))


def benjamini_hochberg(pvalues):
    """
    Benjamini-Hochberg FDR correction.

    pvalues: 1D array-like of p-values.

    Returns: array of BH-adjusted q-values, same order/length as 'pvalues'. A test is
    significant at FDR level alpha if its q-value < alpha.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = len(pvalues)

    order = np.argsort(pvalues)
    ranked = pvalues[order]

    q = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity (q-values must be non-decreasing as p increases).
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = q
    return adjusted
