import pandas as pd
import numpy as np
import antspymm


def test_version_returns_valid_dict():
    v = antspymm.version()
    assert isinstance(v, dict)
    assert len(v) > 0
    assert "antspyt1w" in v or "tensorflow" in v or "antspyx" in v


def test_shorten_pymm_names():
    test_input = "dti_mean_fa_sagittal_stratum_include_inferior_longitidinal_fasciculus_and_inferior_fronto_occipital_fasciculus"
    shortened1 = antspymm.shorten_pymm_names(test_input)
    assert isinstance(shortened1, str)
    assert len(shortened1) <= 40

    shortened2 = antspymm.shorten_pymm_names2(test_input)
    assert isinstance(shortened2, str)
    assert len(shortened2) <= 40


def test_extend_list_to_length():
    initial = ["a", "b"]
    extended = antspymm.extend_list_to_length(initial, 5, fill_value=None)
    assert extended == ["a", "b", None, None, None]

    # Length already >= target_length should not shrink
    no_change = antspymm.extend_list_to_length(initial, 2, fill_value=None)
    assert no_change == ["a", "b"]


def test_nrg_filename_to_subjectvisit():
    fn = "PPMI-10023-20230515-T1w-001.nii.gz"
    res = antspymm.nrg_filename_to_subjectvisit(fn, separator="-")
    assert res == "PPMI-10023-20230515"

    path_fn = "/data/repo/PPMI-999-20210203-DTI-002.nii.gz"
    res_path = antspymm.nrg_filename_to_subjectvisit(path_fn, separator="-")
    assert res_path == "PPMI-999-20210203"


def test_get_valid_modalities():
    mods = antspymm.get_valid_modalities()
    assert isinstance(mods, list)
    for expected in ["T1w", "T2Flair", "DTI", "rsfMRI", "NM2DMT"]:
        assert expected in mods

    as_str = antspymm.get_valid_modalities(asString=True)
    assert isinstance(as_str, str)
    assert "T1w" in as_str

    qc_mods = antspymm.get_valid_modalities(qc=True)
    assert isinstance(qc_mods, list)
    assert len(qc_mods) > 0


def test_dict_to_dataframe():
    d = {
        "scalar_num": 42.5,
        "scalar_str": "test",
        "scalar_bool": True,
        "num_list": [10.0, 20.0, 30.0],
    }
    df = antspymm.dict_to_dataframe(d)
    assert isinstance(df, pd.DataFrame)
    assert df["scalar_num"].iloc[0] == 42.5
    assert df["scalar_str"].iloc[0] == "test"
    assert df["scalar_bool"].iloc[0] is True or df["scalar_bool"].iloc[0] == True
    assert df["num_list_mean"].iloc[0] == 20.0


def test_remove_unwanted_columns():
    df = pd.DataFrame({
        "keep_me": [1, 2, 3],
        "X": [0, 0, 0],
        "Unnamed: 0": [4, 5, 6],
        "Unnamed: 1": [7, 8, 9],
    })
    cleaned = antspymm.remove_unwanted_columns(df)
    assert list(cleaned.columns) == ["keep_me"]


def test_filter_columns_by_nan_percentage():
    df = pd.DataFrame({
        "mostly_good": [1.0, 2.0, np.nan, 4.0],  # 25% NaN
        "mostly_nan": [np.nan, np.nan, np.nan, 1.0],  # 75% NaN
    })
    filtered = antspymm.filter_columns_by_nan_percentage(df, max_nan_percentage=50.0)
    assert "mostly_good" in filtered.columns
    assert "mostly_nan" not in filtered.columns


def test_clean_tmp_directory_guard():
    # If age_hours is not float, clean_tmp_directory returns immediately without error
    ret = antspymm.clean_tmp_directory(age_hours="not_a_float")
    assert ret is None
