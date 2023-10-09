# Developer guide

This guide describes principles and workflows for developers.

## Setup

We recommend programming in a fresh virtual environment. You can set up the
`conda` environment and activate it

```bash
make conda-env
conda activate singd
```

If you don't use `conda`, set up your preferred environment and run

```bash
pip install -e ."[lint,test,doc]"
```
to install the package in editable mode, along with all required development dependencies
(the quotes are for OS compatibility, see
[here](https://github.com/mu-editor/mu/issues/852#issuecomment-498759372)).

## Continuous integration

To standardize code style and enforce high quality, checks are carried out with
Github actions when you push. You can also run them locally, as they are managed
via `make`:

- Run tests with `make test`

- Run all linters with `make lint`, or separately with:

    - Run auto-formatting and import sorting with `make black` and `make isort`

    - Run linting with `make flake8`

    - Run docstring checks with `make pydocstyle-check` and `make darglint-check`

## Documentation

To build the documentation, run the following command from the repository root,

```bash
mkdocs serve
```

and navigate to the displayed address.
