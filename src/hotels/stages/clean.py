"""Stage 1: load the raw csv, run the cleaning pipeline, write parquet.

Usage: python -m hotels.stages.clean
"""
from __future__ import annotations

from hotels import data
from hotels.stages._io import ensure_parent, load_params, resolve


def main() -> None:
    p = load_params()
    raw_path = resolve(p["paths"]["raw"])
    out_path = ensure_parent(resolve(p["paths"]["cleaned"]))

    df = data.load_raw(raw_path)
    df = data.clean(df)

    df.to_parquet(out_path, index=False)
    print(f"clean: {df.shape}, written to {out_path.name}")


if __name__ == "__main__":
    main()
