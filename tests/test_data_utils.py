import pandas as pd

from data_utils import clean_data, get_split


def _sample_dataframe(rows=100):
    target = [0, 1] * (rows // 2)
    return pd.DataFrame(
        {
            "Unnamed: 0": range(rows),
            "PreviousSalary": [50_000] * rows,
            "Employment": ["Employed"] * rows,
            "HaveWorkedWith": [None, "Python;SQL"] * (rows // 2),
            "Employed": target,
            "EdLevel": ["Undergraduate"] * rows,
            "MainBranch": ["Developer"] * rows,
            "Accessibility": ["No"] * rows,
            "MentalHealth": ["No"] * rows,
            "Gender": ["Woman"] * rows,
            "Age": ["18-24"] * rows,
            "YearsCode": [3] * rows,
            "YearsCodePro": [1] * rows,
            "ComputerSkills": [2] * rows,
            "Country": ["Spain"] * rows,
        }
    )


def test_clean_data_removes_leakage_columns_and_fills_skill_nulls():
    cleaned = clean_data(_sample_dataframe())

    assert "PreviousSalary" not in cleaned.columns
    assert "Employment" not in cleaned.columns
    assert "Unnamed: 0" not in cleaned.columns
    assert cleaned["HaveWorkedWith"].isna().sum() == 0


def test_split_is_stratified_and_has_no_index_overlap():
    cleaned = clean_data(_sample_dataframe())
    X_train, X_val, X_test, y_train, y_val, y_test = get_split(cleaned)

    assert (len(X_train), len(X_val), len(X_test)) == (69, 15, 16)
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - 0.5) < 0.05
    assert abs(y_val.mean() - 0.5) < 0.05
    assert abs(y_test.mean() - 0.5) < 0.05
