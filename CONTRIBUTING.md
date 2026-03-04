# SparseAutoschedulingBenchmark: Contributing Guide

Thank you for your interest in contributing! Please read the following guidelines to help us maintain a high-quality, collaborative codebase.

## Code of Conduct

We adhere to the [Python Code of Conduct](https://policies.python.org/python.org/code-of-conduct/).

## Collaboration Practices

For those who are new to the process of contributing code, welcome! We value your contribution, and are excited to work with you. GitHub's [pull request guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) will walk you through how to file a PR.

Most importantly: Before implementing a benchmark, claim it! File a github issue describing which application you want to benchmark, include links to relevant source code, and assign yourself to the issue if possible so that others know you're working on that benchmark.

Please follow the [SciML Collaborative Practices](https://docs.sciml.ai/ColPrac/stable/) and [Github Collaborative Practices](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes) guides to help make your PR easier to review.

In this repo, please use the convention <initials>/<branch-name> for pull request branch names, e.g. ms/scheduler-pass.
This way in bash when you type your initials git checkout ms/ and <tab> you can see all your branches. We will use other names for special purposes.

### Pre-commit hooks

Pull requests must pass some formatting, linting, and typing checks before we can merge them. These checks can be run automatically before you make commits, which is why they are sometimes called "pre-commit hooks". We use [pre-commit](https://pre-commit.com/) to run these checks.

To run pre-commit hooks manually:
```bash
poetry run pre-commit run -a
```

### Testing
SparseAutoschedulingBenchmark uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run the
tests:

```bash
poetry install --extras test
poetry run pytest
```

- Tests are located in the `tests/` directory at the project root.
- Write thorough tests for your new features and bug fixes.

#### Optional Static Type Checking

The pytest will run mypy to check for type errors, so you shouldn't need to run it manually.
In case you do need to run mypy manually, you can do so with:

```bash
poetry run mypy ./src/
```

#### Regression Tests
pytest-regression is used to ensure that compiler outputs remain consistent across changes, and to better understand the impacts of compiler changes on the test outputs. To regenerate regression test outputs, run pytest with the `--regen-all` flag. Those who are curious can consult the [`pytest-regression` docs](https://pytest-regressions.readthedocs.io/en/latest/overview.html#using-data-regression).

**If you find an error or unclear section, please fix it or open an issue.**
