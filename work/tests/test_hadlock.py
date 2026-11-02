import pytest
from fetal.explain.hadlock import (
    calculate_efw,
    hadlock_reference_mean_sd,
    growth_percentile,
    flag_growth_restriction,
)


def test_hadlock_efw_notebook_cell_13_vector():
    """
    Test vector quoted verbatim from Capstone_fetalDataset.ipynb Cell 13:

    Cell 13:
    ```python
    #Layer - 3 , Hadlock Formula applying

    def calculate_efw(hc_mm, bpd_mm, ac_mm, fl_mm):
        ...
    efw = calculate_efw(hc_mm=280, bpd_mm=80, ac_mm=250, fl_mm=55)
    print(f"Test EFW: {efw:.1f} grams")  # should print ~1403.0
    ```

    Cell 14:
    ```
    ## Test EFW: 1403.0 grams
    ```
    """
    efw = calculate_efw(hc_mm=280, bpd_mm=80, ac_mm=250, fl_mm=55)
    assert abs(efw - 1403.0) < 0.1, f"Expected 1403.0 grams, got {efw:.4f}"
    assert f"{efw:.1f}" == "1403.0"


def test_hadlock_reference_mean_sd():
    """Verify Hadlock 1991 gestational age reference calculation."""
    # At ga_weeks = 30
    mean, sd = hadlock_reference_mean_sd(30.0)
    assert mean > 1000.0 and mean < 2000.0
    assert abs(sd - 0.127 * mean) < 1e-7


def test_growth_percentile_and_restriction_flag():
    """Verify percentile and clinical SGA (<10th percentile) flagging."""
    ga = 30.0
    mean, sd = hadlock_reference_mean_sd(ga)

    # An EFW exactly at mean should give 50th percentile
    pct_mean, z_mean = growth_percentile(mean, ga)
    assert abs(pct_mean - 50.0) < 1e-4
    assert abs(z_mean - 0.0) < 1e-4
    assert not flag_growth_restriction(pct_mean)

    # Severely restricted fetus (Z < -1.5)
    pct_low, z_low = growth_percentile(mean - 2.0 * sd, ga)
    assert pct_low < 10.0
    assert flag_growth_restriction(pct_low)

