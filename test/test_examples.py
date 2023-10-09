"""Execute the examples."""

from glob import glob
from os import path
from subprocess import run

from pytest import mark

HERE_DIR = path.dirname(path.abspath(__file__))
EXAMPLES_DIR = path.join(HERE_DIR, "..", "docs", "examples")
EXAMPLE_PATHS = glob(path.join(EXAMPLES_DIR, "example_*.py"))
EXAMPLE_IDS = [path.basename(example) for example in EXAMPLE_PATHS]


@mark.parametrize("script", EXAMPLE_PATHS, ids=EXAMPLE_IDS)
def test_example(script: str):
    """Execute an example script.

    Args:
        script: The path to the script to execute.
    """
    run(["python", script], check=True)
