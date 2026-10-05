import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_corpus import DEFAULT_OUTPUT_DIR, DEFAULT_SEED, PROJECT_ROOT, write_corpus


def parse_args():
    parser = argparse.ArgumentParser(description="Build the demo document corpus.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-wallets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = write_corpus(
        output_dir=Path(args.output_dir),
        n_wallets=args.n_wallets,
        seed=args.seed,
        rebuild=args.rebuild,
    )
    output = Path(args.output_dir)
    try:
        display_path = output.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output
    print(f"Wrote {display_path}")
    print(f"Documents: {manifest['document_count']}")
    print(f"Chunks: {manifest['chunk_count']}")


if __name__ == "__main__":
    main()
