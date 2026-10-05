from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.inference import build_model_input


@dataclass(frozen=True)
class RowScore:
    row_id: str
    wallet_address: str | None
    score: float
    prediction: int
    threshold: float
    model_version: str
    schema_version: str


@dataclass(frozen=True)
class WalletAggregation:
    rule: str
    score: float
    prediction: int
    records_count: int
    representative_row_id: str


@dataclass(frozen=True)
class ScoringResult:
    status: str
    model_version: str
    schema_version: str
    wallet_address: str | None = None
    reason: str | None = None
    rows: tuple[RowScore, ...] = ()
    aggregation: WalletAggregation | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.aggregation is None:
            result["aggregation"] = None
        return result


class ScoringAdapter:
    """Typed scoring boundary around the row-level LightGBM model."""

    def __init__(
        self,
        model,
        feature_names,
        preprocessing,
        threshold,
        model_version,
        schema_version,
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.preprocessing = preprocessing
        self.threshold = float(threshold)
        self.model_version = model_version
        self.schema_version = schema_version

    def has_model_features(self, feature_row: dict[str, Any]) -> bool:
        return any(feature in feature_row for feature in self.feature_names)

    def score_rows(
        self,
        feature_rows: list[dict[str, Any]] | None = None,
        *,
        wallet_address: str | None = None,
        row_ids: list[str] | None = None,
    ) -> ScoringResult:
        if not feature_rows:
            return self._insufficient(wallet_address, "features_required")

        if not any(self.has_model_features(row) for row in feature_rows):
            return self._insufficient(wallet_address, "no_model_features_provided")

        row_ids = row_ids or [f"input_row_{idx}" for idx in range(len(feature_rows))]
        if len(row_ids) != len(feature_rows):
            raise ValueError("row_ids length must match feature_rows length")

        X = build_model_input(feature_rows, self.feature_names, self.preprocessing)
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        rows = tuple(
            RowScore(
                row_id=str(row_ids[idx]),
                wallet_address=wallet_address or _row_wallet_address(feature_rows[idx]),
                score=float(probability),
                prediction=int(predictions[idx]),
                threshold=self.threshold,
                model_version=self.model_version,
                schema_version=self.schema_version,
            )
            for idx, probability in enumerate(probabilities)
        )

        aggregation = self._aggregate_max_row_probability(rows)
        return ScoringResult(
            status="scored",
            model_version=self.model_version,
            schema_version=self.schema_version,
            wallet_address=wallet_address or _common_wallet_address(rows),
            rows=rows,
            aggregation=aggregation,
        )

    def _aggregate_max_row_probability(self, rows: tuple[RowScore, ...]) -> WalletAggregation:
        scored = pd.DataFrame([asdict(row) for row in rows])
        representative = scored.sort_values(
            ["score", "row_id"],
            ascending=[False, True],
        ).iloc[0]
        return WalletAggregation(
            rule="max_row_probability",
            score=float(representative["score"]),
            prediction=int(representative["prediction"]),
            records_count=int(len(rows)),
            representative_row_id=str(representative["row_id"]),
        )

    def _insufficient(self, wallet_address: str | None, reason: str) -> ScoringResult:
        return ScoringResult(
            status="insufficient_data",
            model_version=self.model_version,
            schema_version=self.schema_version,
            wallet_address=wallet_address,
            reason=reason,
        )


def _row_wallet_address(row: dict[str, Any]) -> str | None:
    value = row.get("wallet_address")
    return None if value is None else str(value)


def _common_wallet_address(rows: tuple[RowScore, ...]) -> str | None:
    addresses = {row.wallet_address for row in rows if row.wallet_address is not None}
    return addresses.pop() if len(addresses) == 1 else None
