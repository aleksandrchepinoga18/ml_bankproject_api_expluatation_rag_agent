import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "models/lightgbm_model.pkl",
    "models/lightgbm_best_threshold.pkl",
    "models/lightgbm_feature_names.pkl",
    "models/lightgbm_preprocessing.pkl",
    "data/corpus/manifest.json",
    "data/corpus/chunks.jsonl",
    "data/vector_index/local_dense_index.json",
]


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (PROJECT_ROOT / path).exists()]
    if missing:
        print("Missing runtime artifacts required for full CI tests:", file=sys.stderr)
        for path in missing:
            print(f"- {path}", file=sys.stderr)
        print(
            "Restore these artifacts from the approved test artifact store before running full CI.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Runtime artifact check passed: {len(REQUIRED_FILES)} required files found.")


if __name__ == "__main__":
    main()
