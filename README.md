# SparseApplicationBenchmark

Sparse array programming frameworks, such as [SciPy](https://scipy.org) or [pydata/sparse](https://sparse.pydata.org/en/stable/), are getting more advanced. Because sparse performance depends heavily on input sparsity patterns and full application structure, we need realistic applications to make informed design decisions. This benchmark suite consists of applications written with collective operations, such as `+`, `*`, `sum`, and `reduce`, over sparse arrays.

The programs are adapted from real-world applications using a straightforward translation to an [array-programming](https://en.wikipedia.org/wiki/Array_programming#Array_languages) style. The standard benchmark function is plain Python using [Array API](https://data-apis.org/array-api/latest/API_specification/) functions, minimal control flow, and no framework-specific shortcuts.

We take inspiration from benchmark suites in the database community, such as [pandasbench](https://arxiv.org/abs/2506.02345), [Join Order Benchmark](https://dl.acm.org/doi/10.14778/2850583.2850594), and [TPC-H](https://www.tpc.org/tpch/).

## Installation

SparseApplicationBenchmark uses [Poetry](https://python-poetry.org/) for packaging. To install the project and its test/development dependencies:

```bash
poetry install --with test
```

Most commands below assume they are run from the repository root.

## Running Benchmarks

The main entry point is `bin/run_benchmark.py`. By default it builds an ASV benchmark matrix using the built-in framework wrappers in `frameworks/`:

```bash
poetry run ./bin/run_benchmark.py
```

Useful runner options:

- `--tag test`: run datasets tagged for CI-sized correctness/performance checks.
- `--tag standard`: run canonical suite datasets.
- `--re REGEX`: include benchmark, generator, or dataset names matching a regex.
- `--no-re REGEX`: exclude matching benchmark, generator, or dataset names.
- `--metric time peakmem`: collect one or both metrics.
- `--quick`: run each selected benchmark once.
- `--timeout 30`: set a per-benchmark timeout in seconds.
- `--chunk-count N --chunk-index I`: split the selected parameter cases across multiple processes.

For example:

```bash
poetry run ./bin/run_benchmark.py --tag test --quick --timeout 30
poetry run ./bin/run_benchmark.py --tag standard --metric time peakmem
poetry run ./bin/run_benchmark.py --re bfs --no-re toy
```

Runner outputs are written under `.saps/outputs/`, including ASV result files and cached datasets.

## Configuration

The runner auto-detects `saps.conf.json` in the current directory, or you can pass one explicitly:

```bash
poetry run ./bin/run_benchmark.py --config path/to/saps.conf.json
```

The config file can override the ASV environment type, dependency matrix, framework list, and install command. A small config for running only one framework in the current environment looks like:

```json
{
  "environment_type": "existing:same",
  "matrix": {
    "env_nobuild": {
      "SAPS_FRAMEWORK": ["frameworks/saps_numpy.py"],
      "SAPS_REPO_ROOT": ["."]
    }
  }
}
```

If no config is supplied, the runner uses a default matrix with the built-in NumPy, SciPy, and pydata/sparse wrappers.

### `saps.conf.json` Fields

The runner consumes a small, explicit subset of ASV configuration. These top-level fields are supported:

- `environment_type`: ASV environment type for normal benchmark runs. The default is `"virtualenv"`. Use `"existing:same"` to run in the current Poetry environment, which is often easiest while developing a framework wrapper. Metadata generation, tracing, and dataset caching always use the current environment.
- `install_command`: ASV install command list. The default reinstalls the project into each ASV environment with `python -mpip install {build_dir} --force-reinstall`.
- `env_dir`: Directory where ASV creates benchmark environments. The SAPS default is `.saps/results`.
- `results_dir`: Directory where ASV writes benchmark results. The SAPS default is `.saps/outputs/results`.
- `matrix`: ASV environment matrix. This is the main field most users customize.

The `matrix` field has two common sections:

- `req`: Python package version requirements. ASV builds environments for the Cartesian product of these versions. For example, `"numpy": ["2.3"]` pins NumPy to that version in generated environments.
- `env_nobuild`: Environment variables that do not require rebuilding the package. Values are lists because ASV treats them as matrix entries.

Important `env_nobuild` entries:

- `SAPS_FRAMEWORK`: One or more framework wrapper files to benchmark. Relative paths are converted to absolute paths from the current working directory before child processes run.
- `SAPS_REPO_ROOT`: Repository root used by freshness discovery in child processes. Use the repository root path, or `"."` when running from the repository root.
- `REMOTE_STORAGE_BACKEND`: Dataset storage backend. Usually `s3` or `local`. The runner inserts the default unless this is already present or `--remote-storage-backend` is passed.
- `REMOTE_STORAGE_BUCKET`: S3 bucket name, or local directory path when using the `local` backend. The runner inserts the default unless this is already present or `--remote-storage-bucket` is passed.

The runner owns these values and normally you should not set them in `saps.conf.json`:

- `SAPS_CACHE_DIR`: set to `.saps/outputs/cache`.
- `SAPS_MANIFEST_PATH`: set to the repository `manifest.json`.
- `SAPS_TAGGER_STATS_DIR` and `SAPS_STATISTICS_PATH`: set during `--trace-statistics`.
- ASV `project`, `repo`, `branches`, `benchmark_dir`, and `html_dir`: derived from the repository and `.saps/outputs`.

CLI flags take precedence for storage backend and bucket. `--timeout` is also a CLI-level setting; prefer passing it directly rather than putting timeout-like fields in `saps.conf.json`.

## Custom Frameworks

To benchmark your own sparse framework, create a Python file that defines an `xp` variable. `xp` must be an instance of a `saps_framework.Framework` subclass. The runner loads framework wrappers from the `SAPS_FRAMEWORK` entries in the ASV matrix, so custom frameworks should usually be supplied through `saps.conf.json`:

```json
{
  "environment_type": "existing:same",
  "matrix": {
    "env_nobuild": {
      "SAPS_FRAMEWORK": ["/path/to/my_framework.py"],
      "SAPS_REPO_ROOT": ["."]
    }
  }
}
```

Then run:

```bash
poetry run ./bin/run_benchmark.py --config saps.conf.json --tag test --quick
```

A framework wrapper is responsible for:

- `from_binsparse(array)`: convert SAPS `BinsparseFormat` inputs into framework arrays.
- `to_binsparse(array)`: convert framework outputs back into `BinsparseFormat`.
- `compute(array)` and `lazy(array)`: force or preserve evaluation as appropriate for the framework.
- `einsum(...)` and Array API operations used by benchmarks.
- `__getattr__`: commonly used to forward Array API calls to the wrapped module.

See `frameworks/saps_numpy.py`, `frameworks/saps_scipy.py`, and `frameworks/saps_sparse.py` for reference wrappers. Benchmark functions receive this wrapper as their first argument, conventionally named `xp`.

## Testing

Run the full test suite with:

```bash
poetry run pytest
```

For a quick runner smoke test over CI-sized datasets:

```bash
poetry run ./bin/run_benchmark.py \
  --tag test \
  --check-suite \
  --metric time \
  --quick \
  --timeout 30
```

Freshness tests check that generated artifacts still match the source code and metadata in the repository:

- `metadata.json` matches benchmark, generator, and dataset metadata.
- `statistics.json` contains current trace-derived tags for datasets selected by the `trace` suite.
- `manifest.json` records current dataset freshness, dependency names, and dependency versions.
- Every concrete generator is reachable through a benchmark, including shell benchmarks for intentionally standalone generators.
- Every manifest record points to a dataset object that exists in the configured remote storage backend.

Freshness tests are marked with `freshness` and are skipped by default. Run them with `poetry run pytest -m freshness tests/test_freshness.py`.

When freshness tests fail after a benchmark, generator, dependency, or storage change, regenerate the affected artifacts rather than editing hashes by hand.

## Datasets And Storage

Datasets are generated by `Generator` classes and cached through the configured storage backend. The default remote backend is:

```text
REMOTE_STORAGE_BACKEND=s3
REMOTE_STORAGE_BUCKET=sparse-array-programming-suite
```

Public reads should not require AWS credentials. Uploads to S3 do require credentials with write access:

```bash
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
AWS_DEFAULT_REGION=us-east-1 \
  poetry run ./bin/run_benchmark.py --cache-datasets
```

If you use temporary credentials, also set `AWS_SESSION_TOKEN`. If you use long-lived IAM keys, leave `AWS_SESSION_TOKEN` unset.

For local-only testing of dataset caching:

```bash
poetry run ./bin/run_benchmark.py \
  --cache-datasets \
  --remote-storage-backend local \
  --remote-storage-bucket /tmp/saps-remote-storage
```

## Metadata And Tracing

Regenerate benchmark metadata after changing benchmark, generator, or dataset metadata:

```bash
poetry run ./bin/run_benchmark.py --generate-metadata
```

Trace statistics are generated by running selected benchmarks with the tagger framework:

```bash
poetry run ./bin/run_benchmark.py \
  --trace-statistics \
  --tag trace \
  --timeout 30 \
  --show-stderr
```

Datasets get traced when the benchmark, generator, or dataset has the `trace` suite tag. Tracing executes the benchmark with `frameworks/saps_tagger.py`, records which array operations and sparsity-relevant behaviors were observed, and writes those derived tags to `statistics.json`. Tracing is intentionally separate from `metadata.json`: source metadata stays in `metadata.json`, observed behavior stays in `statistics.json`.

## Contributing Benchmarks

If you want to add a benchmark or generator, start with [CONTRIBUTING.md](CONTRIBUTING.md). It describes how to claim a benchmark, what counts as a benchmark function, what metadata and correctness evidence are required, how freshness works, how to choose dataset tags, and how the generative AI disclosure policy applies.
