#!/usr/bin/env python
"""Rebuild the reconstructible ``decoded.json`` expansions of an experimental adaptive chain.

Usage: python scripts/experimental_reconstruct_decoded.py results/experimental_adaptive_grid_v1/chains/CP_3__aer_sim [--write]

Without --write it only verifies that the reconstruction hash equals the recorded
``decoded_result_hash`` of every round. Never contacts any service.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from acrpq.experimental import adaptive_qpu as aq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chain_dir")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    store = aq.ChainStore(Path(args.chain_dir))
    for r in range(store.n_rounds()):
        if not (store.round_dir(r) / "raw_counts.json").exists():
            print(f"round {r}: no raw counts (nothing to reconstruct)")
            continue
        aq.reconstruct_decoded(store, r, write=args.write)
        print(f"round {r}: decoded reconstruction verified" + (" and written" if args.write else ""))
    problems = store.verify(deep=True)
    print("deep verify:", problems or "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
