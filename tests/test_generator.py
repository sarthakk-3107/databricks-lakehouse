import pandas as pd
import pytest

from lakehouse.generator import TX_COLUMNS, GeneratorConfig, generate, write_landing


@pytest.fixture(scope="module")
def data():
    return generate(GeneratorConfig(n_accounts=300, days=60, seed=7, compromise_rate=0.2))


def test_schema(data):
    tx = data["transactions"]
    assert list(tx.columns) == TX_COLUMNS + ["_file_date"]
    assert {"account_id", "home_lat", "credit_limit"} <= set(data["accounts"].columns)
    assert "_high_risk" not in data["merchants"].columns  # hidden ground truth not exported


def test_fraud_present_and_rare(data):
    tx = data["transactions"]
    rate = tx["is_fraud"].mean()
    assert 0.001 < rate < 0.05
    scenarios = set(tx["fraud_scenario"].dropna())
    assert {"card_testing", "account_takeover", "cloned_card"} & scenarios


def test_dirty_records_injected(data):
    tx = data["transactions"]
    assert tx["amount"].isna().any()
    assert (tx["amount"] < 0).any()
    assert tx["merchant_id"].astype(str).str.startswith("M_UNKNOWN").any()
    assert tx["transaction_id"].duplicated().any()


def test_card_testing_amounts_are_small(data):
    tx = data["transactions"]
    ct = tx[(tx["fraud_scenario"] == "card_testing") & (tx["amount"] > 0)]
    assert (ct["amount"] < 5).mean() > 0.5


def test_reproducible():
    cfg = GeneratorConfig(n_accounts=50, days=30, seed=3)
    a = generate(cfg)["transactions"]
    b = generate(cfg)["transactions"]
    pd.testing.assert_frame_equal(a, b)


def test_write_landing(tmp_path, data):
    summary = write_landing(data, str(tmp_path / "landing"), hold_back_days=5, pending_dir=str(tmp_path / "pending"))
    assert summary["held_back_files"] == 5
    assert len(list((tmp_path / "pending" / "transactions").glob("*.json"))) == 5
    assert (tmp_path / "landing" / "accounts" / "accounts.csv").exists()
