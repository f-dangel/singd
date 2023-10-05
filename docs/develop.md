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
pip install -e .[lint,test,doc]
```

## Continuous integration

In order to standardize the code style and enforce high quality, certain quality
checks are carried out with Github actions when you push. You can also run them
locally, as they are managed via `make`:

- Run tests with `make test`

- Run auto-formatting and import sorting with `make black` and `make isort`

- Run linting with `make flake8`

- Run docstring checks with `make pydocstyle-check` and `make darglint-check`

## Building the documentation

To build the documentation, run the following command from the repository root:

```bash
mkdocs serve
```
