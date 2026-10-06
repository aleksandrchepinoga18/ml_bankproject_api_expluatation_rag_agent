from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def synthetic_dataset_path(tmp_path: Path) -> Path:
    rows = []
    for idx in range(8):
        target = idx % 2
        rows.append(
            {
                "borrow_block_number": 1000 + idx,
                "borrow_timestamp": 1_700_000_000.0 + idx * 60.0,
                "wallet_address": f"0xsynthetic{idx:02d}",
                "first_tx_timestamp": 1_699_900_000.0 + idx * 30.0,
                "last_tx_timestamp": 1_699_950_000.0 + idx * 30.0,
                "incoming_tx_count": 3 + idx,
                "outgoing_tx_count": 2 + idx,
                "risky_tx_count": target + idx,
                "risky_unique_contract_count": target + 1,
                "borrow_count": 1 + target,
                "repay_count": 1,
                "deposit_count": 2,
                "withdraw_amount_sum_eth": float(idx) / 10.0,
                "total_balance_eth": 10.0 + idx,
                "market_rocp": float(idx),
                "market_apo": float(idx) + 0.1,
                "market_macdsignal_macdfix": float(idx) + 0.2,
                "market_macd_macdfix": float(idx) + 0.3,
                "risky_first_tx_timestamp": 1_699_910_000 + idx,
                "market_macd_macdext": float(idx) + 0.4,
                "market_macd": float(idx) + 0.5,
                "market_macdsignal": float(idx) + 0.6,
                "liquidation_count": 0,
                "target": target,
            }
        )

    path = tmp_path / "synthetic_dataset.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
