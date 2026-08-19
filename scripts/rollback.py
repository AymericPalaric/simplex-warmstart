"""Rollback un 'fallen_champion' en 'champion'"""

from __future__ import annotations

import argparse

from mlflow.tracking import MlflowClient

from simplex_warmstart.params import load_params
from simplex_warmstart.tracking import setup_mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--to-version", type=str)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    name = load_params()["registry"]["model_name"]
    setup_mlflow()
    client = MlflowClient()

    if args.list or args.to_version is None:
        for version in client.search_model_versions(f"name='{name}'"):
            aliases = ", ".join(version.aliases) or "-"
            golden = version.tags.get("golden_rmse", "?")
            print(f"v{version.version:>3}   aliases={aliases:<22} golden_rmse={golden}")
        return

    client.set_registered_model_alias(name, "champion", args.to_version)
    print(f"@champion now pointing to version {args.to_version}")


if __name__ == "__main__":
    main()
