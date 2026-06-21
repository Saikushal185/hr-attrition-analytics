from src import main


def test_load_data_creates_target():
    df = main.load_data()
    assert "left" in df.columns
    assert set(df["left"].unique()) <= {0, 1}


def test_drop_cols_removed():
    df = main.load_data()
    for col in main.DROP_COLS:
        assert col not in df.columns
