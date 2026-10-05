from app import api


def _feature_row(value=0):
    return {feature: value for feature in api.feature_names}


def test_scoring_adapter_scores_full_feature_row():
    result = api.scoring_adapter.score_rows(
        [_feature_row(0)],
        wallet_address="0xadapterwallet0001",
        row_ids=["row-full"],
    )

    assert result.status == "scored"
    assert result.wallet_address == "0xadapterwallet0001"
    assert len(result.rows) == 1
    assert result.rows[0].row_id == "row-full"
    assert result.rows[0].model_version == api.MODEL_VERSION
    assert result.aggregation.records_count == 1
    assert result.aggregation.rule == "max_row_probability"
    assert result.aggregation.representative_row_id == "row-full"


def test_scoring_adapter_returns_insufficient_data_for_address_without_features():
    result = api.scoring_adapter.score_rows(
        [{"wallet_address": "0xonlyaddress"}],
        wallet_address="0xonlyaddress",
        row_ids=["address-only"],
    )

    assert result.status == "insufficient_data"
    assert result.reason == "no_model_features_provided"
    assert result.wallet_address == "0xonlyaddress"
    assert result.rows == ()
    assert result.aggregation is None


def test_scoring_adapter_aggregates_multiple_rows_by_max_probability():
    first = _feature_row(0)
    second = _feature_row(0)
    second["risky_tx_count"] = 100

    result = api.scoring_adapter.score_rows(
        [first, second],
        wallet_address="0xmultirow",
        row_ids=["row-low", "row-high"],
    )

    row_scores = {row.row_id: row.score for row in result.rows}
    representative_id = max(row_scores, key=row_scores.get)

    assert result.status == "scored"
    assert result.aggregation.rule == "max_row_probability"
    assert result.aggregation.records_count == 2
    assert result.aggregation.representative_row_id == representative_id
    assert result.aggregation.score == row_scores[representative_id]
