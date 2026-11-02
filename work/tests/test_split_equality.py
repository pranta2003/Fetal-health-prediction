import pytest
import pandas as pd

import ac_v3_original as orig_ac
from fetal import config
from fetal.data import ac_dataset


def test_split_data_equality():
    """Verify split_data produces identical tr, dv, te frames with exact ordering and patients."""
    orig_ac.BASE = config.DATA_DIR
    data = ac_dataset.build_data(config.DATA_DIR)

    tr_orig, dv_orig, te_orig = orig_ac.split_data(data)
    tr_new, dv_new, te_new = ac_dataset.split_data(data)

    # Verify identical dataframes
    pd.testing.assert_frame_equal(tr_orig, tr_new)
    pd.testing.assert_frame_equal(dv_orig, dv_new)
    pd.testing.assert_frame_equal(te_orig, te_new)

    # Verify exact image_name and patient sequence
    assert list(tr_orig["image_name"]) == list(tr_new["image_name"])
    assert list(dv_orig["image_name"]) == list(dv_new["image_name"])
    assert list(te_orig["image_name"]) == list(te_new["image_name"])

    assert list(tr_orig["patient"]) == list(tr_new["patient"])
    assert list(dv_orig["patient"]) == list(dv_new["patient"])
    assert list(te_orig["patient"]) == list(te_new["patient"])

