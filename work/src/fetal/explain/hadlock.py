import math
from scipy.stats import norm


def calculate_efw(hc_mm, bpd_mm, ac_mm, fl_mm):
    """Hadlock 1985 four-parameter EFW formula. Inputs in mm, converted
    to cm internally since Hadlock's coefficients are cm-calibrated."""
    hc_cm, bpd_cm, ac_cm, fl_cm = hc_mm / 10, bpd_mm / 10, ac_mm / 10, fl_mm / 10
    log10_efw = (
        1.3596
        - 0.00386 * (ac_cm * fl_cm)
        + 0.0064 * hc_cm
        + 0.00061 * (bpd_cm * ac_cm)
        + 0.0424 * ac_cm
        + 0.174 * fl_cm
    )
    return 10 ** log10_efw


def hadlock_reference_mean_sd(ga_weeks):
    """Hadlock et al. 1991 fetal weight reference standard.
    Mean EFW from the paper's published regression equation;
    SD as a uniform 12.7% coefficient of variation, per the paper."""
    mean = math.exp(0.578 + 0.332 * ga_weeks - 0.00354 * (ga_weeks ** 2))
    sd = 0.127 * mean
    return mean, sd


def growth_percentile(efw_grams, ga_weeks):
    """Converts an EFW at a given gestational age into a percentile
    against the Hadlock 1991 reference standard."""
    mean, sd = hadlock_reference_mean_sd(ga_weeks)
    z = (efw_grams - mean) / sd
    percentile = norm.cdf(z) * 100
    return percentile, z


def flag_growth_restriction(percentile, threshold=10):
    """Standard clinical convention: <10th percentile = growth-restricted."""
    return percentile < threshold

