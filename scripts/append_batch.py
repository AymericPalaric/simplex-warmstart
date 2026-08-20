"""Ajoute un batch au manifest"""

from __future__ import annotations

import argparse
from pathlib import Path

from ruamel.yaml import YAML

from simplex_warmstart.manifest import next_batch_entry, summarise

PARAMS_PATH = Path("params.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-studies", type=int, default=40)
    parser.add_argument("--families", nargs="+", default=["esters", "silicones"])
    parser.add_argument("--protocol", default="v1")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    yaml = YAML()
    yaml.preserve_quotes = True
    document = yaml.load(PARAMS_PATH)

    batches = document["data"]["batches"]
    entry = next_batch_entry(batches, args.n_studies, args.families, args.seed, args.protocol)
    batches.append(entry)
    yaml.dump(document, PARAMS_PATH)

    stats = summarise(batches)
    print(f"Batch {entry['id']} added ({entry['n_studies']} studies, {entry['families']})")
    print(f"Total : {stats['n_batches']} batches, {stats['n_studies']} studies")


if __name__ == "__main__":
    main()
