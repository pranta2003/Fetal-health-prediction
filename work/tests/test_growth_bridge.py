# FIELD: Fusion
# MODEL: Hadlock 4-Parameter EFW + CDC Live-Birth Percentile Bridge
# STEP: unit_tests
# USED BY: pytest CI suite

import pytest

from fetal.fusion.growth_bridge import assess_fetal_growth, load_cdc_percentiles


def test_hadlock_4param_1403g_vector_and_cdc_band():
    """Verify 1403.0 g Hadlock test vector and CDC percentile band classification."""
    # With explicit bpd_mm=80 and with default bpd_mm=None (280/3.5 = 80.0)
    res_explicit = assess_fetal_growth(hc_mm=280.0, ac_mm=250.0, fl_mm=55.0, ga_weeks=30, bpd_mm=80.0)
    res_inferred = assess_fetal_growth(hc_mm=280.0, ac_mm=250.0, fl_mm=55.0, ga_weeks=30)

    assert f"{res_explicit['efw_g']:.1f}" == "1403.0"
    assert f"{res_inferred['efw_g']:.1f}" == "1403.0"
    # Week 30 CDC live-birth p10=1091.0, p50=1616.0, p90=3240.0 -> 1403.0 is p10_to_p90
    assert res_explicit["p10"] == 1091.0
    assert res_explicit["p50"] == 1616.0
    assert res_explicit["p90"] == 3240.0
    assert res_explicit["band"] == "p10_to_p90"
    assert res_explicit["sga"] is False


def test_cdc_band_boundaries():
    """Verify below_p10, p10_to_p90, and above_p90 bands at week 37."""
    table = load_cdc_percentiles()
    assert 37 in table
    # Small biometrics -> below p10 (2466.0 g at week 37)
    small = assess_fetal_growth(hc_mm=280.0, ac_mm=250.0, fl_mm=55.0, ga_weeks=37, bpd_mm=80.0)
    assert small["efw_g"] < small["p10"]
    assert small["band"] == "below_p10"
    assert small["sga"] is True

    # Large biometrics -> above p90 (3665.6 g at week 37)
    large = assess_fetal_growth(hc_mm=350.0, ac_mm=360.0, fl_mm=76.0, ga_weeks=37, bpd_mm=98.0)
    assert large["efw_g"] > large["p90"]
    assert large["band"] == "above_p90"
    assert large["sga"] is False


def test_edge_weeks_20_and_42():
    """Verify boundary gestational weeks 20 and 42 and out-of-range rejection."""
    w20 = assess_fetal_growth(hc_mm=175.0, ac_mm=150.0, fl_mm=32.0, ga_weeks=20, bpd_mm=47.0)
    assert w20["ga_weeks"] == 20
    assert w20["p10"] == 254.7
    assert w20["p50"] == 336.0
    assert w20["p90"] == 687.4
    assert w20["band"] in {"below_p10", "p10_to_p90", "above_p90"}

    w42 = assess_fetal_growth(hc_mm=350.0, ac_mm=355.0, fl_mm=75.0, ga_weeks=42, bpd_mm=97.0)
    assert w42["ga_weeks"] == 42
    assert w42["p10"] == 2870.0
    assert w42["p50"] == 3435.0
    assert w42["p90"] == 4053.0
    assert w42["band"] in {"below_p10", "p10_to_p90", "above_p90"}

    with pytest.raises(ValueError):
        assess_fetal_growth(hc_mm=280.0, ac_mm=250.0, fl_mm=55.0, ga_weeks=19)
    with pytest.raises(ValueError):
        assess_fetal_growth(hc_mm=280.0, ac_mm=250.0, fl_mm=55.0, ga_weeks=43)
