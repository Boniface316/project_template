"""Scripts for the CLI application."""

# ruff: noqa: E402

# %% WARNINGS

import warnings

# disable annoying mlflow warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

# %% IMPORTS

import argparse
import json
import sys

from .settings import MainSettings
from .io import Config
# %% PARSERS

parser = argparse.ArgumentParser(description="Run an AI/ML job from YAML/JSON configs.")
parser.add_argument(
    "files", nargs="*", help="Config files for the job (local path only)."
)
parser.add_argument(
    "-e", "--extras", nargs="*", default=[], help="Config strings for the job."
)
parser.add_argument(
    "-s", "--schema", action="store_true", help="Print settings schema and exit."
)

# %% SCRIPTS


def main(argv: list[str] | None = None) -> int:
    """Main script for the application."""
    args = parser.parse_args(argv)
    if args.schema:
        schema = MainSettings.model_json_schema()
        json.dump(schema, sys.stdout, indent=4)
        return 0
    files = [Config.parse_file(file) for file in args.files]
    strings = [Config.parse_string(string) for string in args.extras]
    if len(files) == 0 and len(strings) == 0:
        raise RuntimeError("No configs provided.")
    config = Config.merge_configs([*files, *strings])
    object_ = Config.to_object(config)  # python object
    setting = MainSettings.model_validate(object_)
    with setting.job as runner:
        runner.run()
        return 0
