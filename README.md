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
- `--metrics time peakmem`: collect one or both metrics.
- `--quick`: run each selected benchmark once.
- `--rounds 1`: use one timing round, retaining repeated measurements within it.
- `--timeout 30`: set a per-benchmark timeout in seconds.
- `--chunk-count N --chunk-index I`: split the selected parameter cases across
  multiple processes and isolate transient output under `chunk-I` directories.

For example:

```bash
poetry run ./bin/run_benchmark.py --tag test --quick --timeout 30
poetry run ./bin/run_benchmark.py --tag standard --metrics time peakmem
poetry run ./bin/run_benchmark.py --re bfs --no-re toy
```

Runner outputs are written under `.saps/outputs/`, including ASV result files and cached datasets.

## Competition Runs

Use `competition.config.json` to define the frameworks included in a competition
run. It uses ASV-native `include` entries so every framework can pin a different
set of dependencies:

```bash
poetry run ./bin/run_benchmark.py \
  --config competition.config.json \
  --metrics time peakmem
```

On Slurm, use the wrapper script:

```bash
sbatch scripts/run-competition.slurm
SAPS_COMPETITION_ARGS="--metrics time peakmem" \
  sbatch scripts/run-competition.slurm
```

Slurm stdout and stderr logs go to the directory where you submit the job:
`competition-%A_%a.log`, `upload-%j.log`, `trace-%A_%a.log`, or
`finalize-metadata-%j.log`. The refresh launcher (`scripts/submit-refresh-jobs.sh`)
also preserves the directory where you invoked it for all three jobs' logs.
You can submit the Slurm scripts from the repository root or any subdirectory.

The competition script emails `ahrens@gatech.edu` when the array finishes or
fails. Notifications cover the whole array. Override the recipient at submission
with `sbatch --mail-user=you@example.com scripts/run-competition.slurm`.

The competition config selects the standard datasets and uses one timing round
per benchmark, with ASV's normal repeated measurements. The wrapper submits a
256-task array by default, with a 90-minute time limit per task. Each task runs a
deterministic set of the selected datasets. All competition outputs live in the
run directory:

```text
competition/run_<slurm-array-job-id>/
  task_0/
    results/       # ASV measurements and saved diagnostics
    machine_files/
    outputs/       # task-specific reports
  task_1/
  ...
  results.json     # combined measurements from all tasks and frameworks
  machines.json   # machine descriptions, indexed by ID
```

Slurm benchmark environments and pip/virtualenv caches live under
`$TMPDIR/saps-competition-<job-id>-<task-index>/` on the assigned node. The wrapper
requests 50 GB of local temporary disk per node with `--tmp=50G`. These files are
removed by Slurm when the job ends and rebuilt when resuming; saved results stay
in the run directory. This avoids filling the shared scratch file-count quota
with separate Python installations for every task. The existing Poetry
environment is still used to launch the runner.

Older runs may still have `task_*/env` directories on shared storage. Delete those
environment directories only after their jobs have stopped; retain the result
directories for resume. Updating the wrapper affects newly submitted jobs.

Dataset inputs use the repository's shared `.saps/outputs/cache`, across runs,
tasks, frameworks, uploads, and statistics tracing. Nodes using the same shared
checkout reuse the same cache. Set `SAPS_CACHE_DIR` before launching the runner
to use a different shared directory; it is independent of `--saps-dir` and chunk
selection. Cache files are named by content hash, and downloads are verified
before being published atomically. Concurrent requests for the same cache file
wait on a file lock and reuse the first completed download; different datasets
can download in parallel.

Benchmark runs use the manifest's recorded digest without comparing source paths
or freshness hashes against the benchmark environment. Freshness is checked by
the dataset refresh/upload workflow. Downloaded files still have their checksums
verified before entering the shared cache. Missing manifest entries or unavailable
prepared data fail setup with an instruction to run `--cache-datasets`; normal
runs never regenerate cacheable inputs or write manifest metadata.

Generators marked `cacheable = False` still assemble benchmark inputs during
setup from their shared source datasets. Those transformations must preserve
sparsity and use prepared data. The G-CARE downloader reads graph matrices,
queries, and ground-truth counts. The shell generator wraps its returned arrays
and metadata in a `DataInstance` for the storage backend to cache. The regular
subgraph generator assembles each query from that cached input without calling
the downloader or reading source files.
Older G-CARE caches need a one-time refresh from the repository root:

```bash
poetry run ./bin/run_benchmark.py --cache-datasets --re '^subgraph_gcare_graph$'
```

Run this in the dataset-upload environment with its configured storage backend.
The shell places G-CARE source files under the storage backend's cache directory
in `gcare/`.

SuiteSparse source downloads used during cache preparation also share this cache,
under `suitesparse/<group>/<name>`. A lock and atomic publication let workers reuse
one completed source download across its RHS selections.

Each finishing task refreshes the combined files with the results saved so far.
To rebuild the highest-numbered Slurm run, run from the repository root:

```bash
poetry run ./scripts/combine_competition_results.py
```

The script also works from `scripts/` as `poetry run ./combine_competition_results.py`.
To select an earlier run explicitly:

```bash
poetry run ./scripts/combine_competition_results.py \
  --run-directory competition/run_12345
```

The combined JSON follows `metadata.json`'s
`benchmarks → generators → datasets → results` hierarchy. Dataset names are
matched using metadata's `asv_ids` and `asv_param` values, rather than parameter
positions. Each result contains a metric, value, status, statistics, samples,
and references to its framework, machine, source file, and diagnostics.
Framework definitions, source information (including commit and environment),
and diagnostics appear once in top-level tables. Machine references resolve
through `machines.json`. Source file paths are relative to `run_directory`.
Use `--metadata PATH` to combine against a different metadata file.

Diagnostics retain ASV's `errcode`, `stderr`, parameter names, start time, and
duration. ASV reports these per benchmark invocation, potentially covering
multiple datasets; those results share a diagnostic reference. Resume preserves
the diagnostics and machine attribution for retained measurements. Older ASV
files can still be combined, but their missing diagnostics cannot be recovered;
the machine listing uses whatever machine identity those files recorded.
Unselected parameters are omitted; recorded skips appear with `status: "skipped"`.
Non-finite values become JSON `null` rather than nonstandard `NaN` literals.

Direct runs using `competition.config.json` write beneath
`competition/run_local/task_0/`; combine those with
`--run-directory competition/run_local`.

Competition runs filter out dataset/metric entries that already have saved
results for each environment. To continue a Slurm run, submit the same array shape
and configuration with its existing run directory:

```bash
sbatch --array=0-255 scripts/run-competition.slurm --resume competition/run_12345
```

Only missing or null results run again. Results are saved after each environment;
work interrupted before it was saved runs again. New jobs use their own
submission-directory logs while continuing results in the original run directory.
For direct runner use, pass `--resume` with the same `--results-dir` and `--machine`.
The competition config enables this automatically. Results from another commit,
environment, or benchmark version are not reused.

## Configuration

The runner auto-detects `saps.conf.json` in the current directory, or you can pass one explicitly:

```bash
poetry run ./bin/run_benchmark.py --config path/to/saps.conf.json
```

The config file can set runner options and supported ASV environment fields. A small config for running only one framework in the current environment looks like:

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

### SAPS Config Overview

`saps.conf.json` is not a complete ASV config file. It is a small SAPS runner
config that is translated into an in-memory ASV config by `bin/run_benchmark.py`.
SAPS fills in benchmark-suite details such as the project name, repository path,
benchmark directory, HTML output directory, dataset cache, manifest path, and
remote storage defaults.

Every `bin/run_benchmark.py` CLI option except `--config` can also be supplied
in config by using its argparse destination name. In practice, replace hyphens
with underscores: `remote_storage_backend` for `--remote-storage-backend`,
`cache_datasets` for `--cache-datasets`, `chunk_count` for `--chunk-count`, and
`metrics` for `--metrics`. CLI values always take precedence over config values.

The ASV fields SAPS supports directly are:

- `environment_type`: ASV environment type for normal benchmark runs. The default is `"virtualenv"`. Use `"existing:same"` to run in the current Poetry environment, which is often easiest while developing a framework wrapper. Tracing and dataset caching always use the current environment; metadata generation runs directly in the environment used to invoke `bin/generate_metadata.py`.
- `install_command`: ASV install command list. The default reinstalls the project into each ASV environment with `python -mpip install {build_dir} --force-reinstall`.
- `pythons`: Python versions ASV should use when constructing environments.
- `saps_dir`: Directory where SAPS writes runner-owned outputs such as machine files and HTML. The SAPS default is `.saps`; dataset inputs use a shared cache independently of this setting.
- `env_dir`: Directory where ASV creates benchmark environments. The SAPS default is `.saps/results`.
- `results_dir`: Directory where ASV writes benchmark results. The SAPS default is `.saps/outputs/results`.
- `matrix`: ASV environment matrix. This is the main field most users customize.
- `include`: ASV explicit environment list. Use this when each environment needs its own dependency set, such as competition runs.
- `exclude`: ASV matrix exclusion rules.

The `matrix` field has two common sections:

- `req`: Python package version requirements. ASV builds environments for the Cartesian product of these versions. For example, `"numpy": ["2.3"]` pins NumPy to that version in generated environments.
- `env_nobuild`: Environment variables that do not require rebuilding the package. Values are lists because ASV treats them as matrix entries.

The `include` field is useful when `matrix` would create unwanted combinations.
Each `include` entry describes one explicit environment. Unlike `matrix`, values
inside `include.req`, `include.env`, and `include.env_nobuild` are scalars, not
lists:

```json
{
  "matrix": {},
  "exclude": [{"env_nobuild": {"SAPS_FRAMEWORK": null}}],
  "include": [
    {
      "python": "3.12",
      "req": {"numpy": "2.3", "scipy": "1.17.1"},
      "env_nobuild": {"SAPS_FRAMEWORK": "frameworks/saps_scipy.py"}
    }
  ]
}
```

The empty `matrix` plus `exclude` rule prevents ASV from also running its base
environment. See the [ASV config reference](https://asv.readthedocs.io/en/stable/asv.conf.json.html)
for the full upstream meaning of `matrix`, `include`, and `exclude`.

Important `env_nobuild` entries:

- `SAPS_FRAMEWORK`: One or more framework wrapper files to benchmark. Relative paths are converted to absolute paths from the current working directory before child processes run.
- `SAPS_REPO_ROOT`: Repository root used by freshness discovery in child processes. Use the repository root path, or `"."` when running from the repository root.

The runner owns these values and normally you should not set them in `saps.conf.json`:

- `REMOTE_STORAGE_BACKEND` and `REMOTE_STORAGE_BUCKET`: set from `remote_storage_backend` / `remote_storage_bucket`, CLI flags, or built-in defaults.
- `SAPS_CACHE_DIR`: uses the existing environment value, or defaults to `.saps/outputs/cache` under the repository root. Passed to workers as an absolute path shared across runs and tasks.
- `SAPS_MANIFEST_PATH`: set to the repository `manifest.json`.
- `SAPS_TAGGER_STATS_DIR` and `SAPS_STATISTICS_PATH`: set during `--trace-statistics`.
- ASV `project`, `repo`, `branches`, `benchmark_dir`, and `html_dir`: derived from the repository and `.saps/outputs`.

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
  --metrics time \
  --quick \
  --timeout 30
```

Freshness tests check that generated artifacts still match the source code and metadata in the repository:

- `metadata.json` matches benchmark, generator, and dataset metadata.
- `statistics.json` contains current trace-derived tags for datasets selected by the `trace` tag.
- `manifest.json` records current dataset freshness.
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
poetry run ./bin/generate_metadata.py
```

Trace statistics are generated by running selected benchmarks with the tagger framework:

```bash
poetry run ./bin/run_benchmark.py \
  --trace-statistics \
  --tag trace \
  --timeout 30 \
  --show-stderr
```

Datasets get traced when their generated metadata has the `trace` tag. Tracing executes the benchmark with `frameworks/saps_tagger.py`, records which array operations and sparsity-relevant behaviors were observed, and writes those derived tags to `statistics.json`. Fold fresh trace-derived tags into `metadata.json` with:

```bash
poetry run ./bin/generate_metadata.py --statistics statistics.json
```

Dataset caching and statistics tracing append diagnostics to
`.saps/outputs/results/diagnostics.log`.

## Contributing Benchmarks

If you want to add a benchmark or generator, start with [CONTRIBUTING.md](CONTRIBUTING.md). It describes how to claim a benchmark, what counts as a benchmark function, what metadata and correctness evidence are required, how freshness works, how to choose dataset tags, and how the generative AI disclosure policy applies.
