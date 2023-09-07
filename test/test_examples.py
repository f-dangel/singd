"""Execute the examples."""

from os import path
from subprocess import run

from pytest import mark

HERE_DIR = path.dirname(path.abspath(__file__))
EXAMPLES_DIR = path.join(HERE_DIR, "..", "examples")

SCRIPTS = [
    "sngd_and_adamw_simple.py",
    "sngd_and_adamw_full.py",
]
SCRIPT_PATHS = [path.join(EXAMPLES_DIR, script) for script in SCRIPTS]


@mark.parametrize("script", SCRIPT_PATHS, ids=SCRIPTS)
def test_example(script: str):
    """Execute an example script.

    Args:
        script: The path to the script to execute.
    """
    run(["python", script], check=True)
