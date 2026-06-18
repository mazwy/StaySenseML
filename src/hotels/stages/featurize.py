"""Stage 2: read cleaned parquet, engineer features, drop redundant columns,
write featured parquet.

Usage: python -m hotels.stages.featurize
"""
from __future__ import annotations

import pandas as pd

from hotels import features
from hotels.stages._io import ensure_parent, load_params, resolve


def main() -> None:
    p = load_params()
    in_path = resolve(p["paths"]["cleaned"])
    out_path = ensure_parent(resolve(p["paths"]["featured"]))

    df = pd.read_parquet(in_path)
    df = features.engineer(df)

    df.to_parquet(out_path, index=False)
    print(f"featurize: {df.shape} rows, {df.shape[1]} cols, written to {out_path.name}")


if __name__ == "__main__":
    main()
