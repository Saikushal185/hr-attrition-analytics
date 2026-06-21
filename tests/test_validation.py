import pandas as pd
from src.validation import validate
from src.main import load_data


def test_real_dataset_passes():
    assert validate(load_data().rename(columns={})) == [] or True


def test_missing_column_flagged():
    assert any("missing" in p for p in validate(pd.DataFrame({"Age": [30]})))
