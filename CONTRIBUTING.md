# SparseAutoschedulingBenchmark: Contributing Guide

Thank you for your interest in contributing! Please read the following guidelines to help us maintain a high-quality, collaborative codebase.

## Code of Conduct

We adhere to the [Python Code of Conduct](https://policies.python.org/python.org/code-of-conduct/).

## Collaboration Practices

For those who are new to the process of contributing code, welcome! We value your contribution, and are excited to work with you. GitHub's [pull request guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) will walk you through how to file a PR.

Most importantly: Before implementing a benchmark, claim it! File a GitHub issue describing which application you want to benchmark, include links to relevant source code, and assign yourself to the issue if possible so that others know you're working on that benchmark.

Please follow the [SciML Collaborative Practices](https://docs.sciml.ai/ColPrac/stable/) and [GitHub Collaborative Practices](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes) guides to help make your PR easier to review.

In this repo, please use the convention <initials>/<branch-name> for pull request branch names, e.g. ms/scheduler-pass.
This way in bash when you type your initials git checkout ms/ and <tab> you can see all your branches. We will use other names for special purposes.

## Benchmark/Dataset Contribution Process

Before implementing a benchmark, claim it. Pick an unclaimed benchmark issue, or open a new issue describing the application you want to add. Include links to the paper, source code, pseudocode, textbook section, or other primary reference you plan to translate. Be specific: line numbers, figure numbers, and section numbers help reviewers understand what you are implementing.

After you claim a benchmark:

1. Comment on the issue registering your intent to implement it.
2. Mark or ask a maintainer to mark the issue as claimed.
3. Open a draft pull request early, even if it only has a small hard-coded input.
4. Request review when the benchmark function and initial tests are ready.
5. Add larger or more realistic generators after the benchmark function is reviewed. This may be a separate PR.

Benchmark claims expire after one month without visible progress, so others can pick up stalled work.

## What Counts As A Benchmark

A complete benchmark contribution includes:

- A benchmark function: the sparse array program being measured.
- One or more generators: deterministic ways to produce inputs for that benchmark.
- Datasets: named generator configurations representing specific benchmark runs.
- Metadata: authors, references, motivation, concepts, suites, and AI disclosure.
- Correctness tests: evidence that the benchmark computes the intended result.
- A rationale: evidence that the problem and selected data are representative or useful.

Good benchmarks are adapted from real applications, papers, textbooks, or established benchmark suites. The goal is not to create tiny kernels in isolation; the goal is to preserve the shape of realistic sparse array programs in a form that many frameworks can run.

## Benchmark Function Requirements

Benchmark functions should be plain Python functions whose first argument is the framework wrapper, conventionally named `xp`. The wrapper represents the sparse array framework being tested, such as NumPy, SciPy, pydata/sparse, or another implementation.

Benchmark functions should:

- Use Array API style operations through `xp`.
- Use basic Python syntax, including `for`, `while`, and `if` when needed.
- Avoid framework-specific shortcuts that only one implementation can support.
- Avoid file I/O, threads, networking, global mutable state, recursion, and non-determinism.
- Convert input `BinsparseFormat` values to framework arrays during setup, not inside the measured function body.
- Return framework arrays that SAPS can convert back to `BinsparseFormat`.

Prefer clear translations over clever rewrites. If the original application uses a sparse matrix expression, preserve that structure unless a small adaptation is needed to fit the Array API style.

## Generators And Datasets

A `Generator` owns the logic for producing related datasets. A `Dataset` is one named run configuration for a generator. Generators should produce `DataInstance` objects containing:

- `inputs`: input arrays in `BinsparseFormat`.
- `meta`: non-array scalar or structured metadata needed by the benchmark.
- `ref_outputs`: expected output arrays, when available.
- `ref_meta`: expected non-array output metadata, when available.

Generators should be deterministic. If random data is needed, use fixed seeds and record enough metadata to reproduce the dataset. Large generated datasets should be cacheable through the storage backend unless there is a specific reason to opt out.

Concrete generators must be reachable from a benchmark so metadata, freshness checks, and dataset caching can discover them. If a generator is intentionally standalone data plumbing rather than part of a runnable benchmark, wrap it in a small `ShellBenchmark`: implement only the `generator` property and let the shell expose the generator under a `{generator.name}_shell` benchmark record.

## Correctness Tests

Every benchmark needs correctness evidence. Acceptable approaches include:

- Compare against a trusted package or reference implementation.
- Provide at least three hand-checkable examples covering different cases.
- Write an output checker when exact output equality is not the right criterion.

Prefer tests that exercise edge cases: empty inputs, singletons, disconnected graph components, repeated labels, symmetric/asymmetric structure, dense fallbacks, and shape corner cases when they are relevant to the problem.

## Metadata, Tags, And Concepts

Datasets represent one run of a benchmark. Generators group similar data-generation logic. Benchmarks are the algorithms being run.

Tags can be applied to datasets, generators, or benchmarks. If a generator or benchmark is tagged, its datasets inherit that tag.

Common suite tags include:

- `test`: datasets that run in CI and cover dense/sparse cases, different sizes, and code paths.
- `standard`: canonical datasets in the suite.
- `trace`: datasets whose operation behavior should be traced into `statistics.json`.
- `dynamic`: datasets with random seeds or generated kernels.
- `AI`: AI-generated datasets.

Use ACM CCS XML to describe problem domains. Paste the XML into `concepts`, and SAPS generates lowercase hyphenated topic tags such as `applied-computing`, `physical-sciences-and-engineering`, or `aerospace`.

Every benchmark, generator, and dataset provides `concepts`. Use `"<ccs2012></ccs2012>"` as a stub until you have a classification. You can use the [ACM CCS 2012 generator](https://dl.acm.org/ccs/ccs.cfm), copy its XML output, and paste it in as `concepts`:

```python
@property
def concepts(self) -> str:
    return """
    <ccs2012>
      <concept>
        <concept_desc>Applied computing~Physical sciences and engineering~Aerospace</concept_desc>
      </concept>
    </ccs2012>
    """
```

Manual suite tags are written to `metadata.json` under `suites`. Topic tags generated from ACM CCS XML are written under `topics`. Trace-derived tags are written to the parallel `statistics.json` hierarchy.

## Problem Quality Tags

These tags describe the general character of a problem, and are generated programatically by running --trace-statistics:

- `high-dimensional`: 5 or more dimensions in a tensor.
- `tensor`: 3 or more dimensions in a tensor.
- `large-query`: 5 or more operands on one line.
- `elementary-ops`: PEMDAS-only.
- `transcendental-ops`: contains sin, cos, pow, exp, or related operations.
- `shape-ops`: reshape, concat, transpose, squeeze, or similar operations.
- `linalg-ops`: contains `xp.linalg` or solver-like operations. `dot` is okay.
- `fancy-ops`: min, max, and, or, shift, or similar operations.
- `index-ops`: contains indexing.
- `nonzero-fill`: uses a fill value other than zero.
- `iterative`: loops over a matrix or repeats until convergence.
- `dense`: exclusively dense problems.
- `hypersparse`: contains hypersparsity, such as `nnz << n` for a dimension.
- `dynamic-sparsity`: sparse-sparse interactions may change the sparsity pattern.

## Freshness

SAPS records freshness so generated artifacts can be checked against the code that produced them. Freshness includes:

- The source file containing the benchmark, generator, or dataset class.
- The hash of local source files imported by that class.
- External dependency module names.
- Installed dependency versions recorded in `dependency_versions`.

Freshness is checked for:

- `metadata.json`: benchmark, generator, and dataset metadata.
- `statistics.json`: trace-derived tags for datasets selected by the `trace` suite.
- `manifest.json`: cached dataset digests and freshness records.
- Remote storage: every manifest record must point to an object that exists in the configured backend.

After changing benchmark code, generator code, metadata, dependency imports, or storage behavior, regenerate the affected artifacts:

```bash
poetry run ./bin/run_benchmark.py --generate-metadata
poetry run ./bin/run_benchmark.py --trace-statistics --tag trace --timeout 30 --show-stderr
poetry run ./bin/run_benchmark.py --cache-datasets
```

Do not edit freshness hashes by hand.

## Refresh Workflow And Dataset Uploads

The repository has a manually dispatched GitHub Actions workflow named `refresh` that can regenerate the large derived artifacts in CI. Use it when a PR changes benchmark metadata, trace behavior, generator output, dataset freshness, dependency versions, or storage behavior enough that local regeneration is inconvenient or likely to differ from CI.

The refresh workflow does three things:

- Builds `metadata.json` once and shares it with later jobs.
- Runs trace statistics in four chunks, then merges the chunks into `statistics.json`.
- Runs dataset caching/upload in four chunks, then merges the chunks into `manifest.json`.

The data jobs use the configured S3 backend and bucket from `.github/workflows/refresh.yml`. Public dataset reads do not need credentials, but uploads do need the repository AWS secrets. The workflow uploads generated JSON as GitHub Actions artifacts:

- `saps-metadata`: generated `metadata.json`.
- `saps-statistics-final`: merged `statistics.json`.
- `saps-manifest-final`: merged `manifest.json`.

After the workflow finishes, download the final artifacts, replace the corresponding files in the PR, and run the freshness tests locally:

```bash
poetry install --with test
poetry run pytest tests/test_freshness.py
```

Do not commit `.saps/outputs/cache`, `.saps/outputs/results`, or trace scratch output. Only commit the source changes and the refreshed JSON files.

If you run the uploader locally, use the same test-pinned environment that CI uses:

```bash
poetry install --with test
REMOTE_STORAGE_BACKEND=s3 \
REMOTE_STORAGE_BUCKET=sparse-array-programming-suite \
poetry run ./bin/run_benchmark.py --cache-datasets
```

Dataset digests should be content hashes of serialized data. Freshness records and dependency versions belong in manifest metadata; they should not force a new remote object when the data bytes have not changed.

## Tracing

Tracing runs selected benchmark cases with `frameworks/saps_tagger.py`. The tagger records observed array operations and sparsity-relevant behavior, then writes those derived tags to `statistics.json`.

A dataset is selected for tracing when the benchmark, generator, or dataset has the `trace` suite tag. Use `trace` for representative datasets that should describe the benchmark's operation mix. Avoid tracing huge datasets unless their scale is necessary for the behavior being observed.

Tracing should not change the semantics of a benchmark. It is a metadata pass over selected benchmark executions.

## Generative AI Policy

Generative AI tools may be used to clarify concepts, find bugs, explain errors, or help write tests. Do not use generative AI to write the benchmark function itself. The validity of this benchmark suite depends on benchmark functions being human translations of real applications by contributors who understand the source problem.

Benchmark files must disclose:

- The benchmark authors.
- An assertion that generative AI was not used to construct the benchmark function.
- How generative AI was used, if at all, for tests, generators, documentation, debugging, or other supporting code.

Written communication in issues, pull requests, reviews, presentations, and project channels should reflect your own understanding. If something is unclear, say so directly.

## Credit And Authorship

The code is MIT licensed. Contributors are listed with the code they write. Contributors whose benchmarks are merged are acknowledged by the project. Contributors who make serious attempts that do not merge may still be credited separately.

For research-credit contributors, course-specific requirements, grading, and paper authorship policies are set by the instructors or maintainers. As a general guideline, contributors who complete at least three benchmark issues may be invited to contribute text to a future paper and be listed as authors, subject to the active course or project policy.

## Finding Benchmark Ideas

Good sources include:

- Application source code on GitHub or Kaggle.
- Existing sparse, tensor, graph, solver, simulation, or machine-learning benchmark suites.
- Textbooks and papers with clear pseudocode.
- Reference implementations from application domains.

Useful starting points:

- How to read a paper: <https://web.stanford.edu/class/cs245/readings/how-to-read-a-paper.pdf>
- GraphBLAS math: <https://www.mit.edu/~kepner/GraphBLAS/GraphBLAS-Math-release.pdf>
- Einsum notes: <https://rockt.ai/2018/04/30/einsum>
- NumPy learning resources: <https://numpy.org/learn/>
- CombBLAS examples: <https://github.com/PASSIONLab/CombBLAS/tree/master>
- Cyclops Tensor Framework examples: <https://github.com/cyclops-community/ctf/tree/master/examples>
- LAGraph algorithms: <https://github.com/GraphBLAS/LAGraph/tree/stable/src/algorithm>
- GraphBLAS notebooks: <https://github.com/python-graphblas/python-graphblas/tree/main/notebooks>
- Iterative methods: <https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf>

## Pre-commit Hooks

Pull requests must pass some formatting, linting, and typing checks before we can merge them. These checks can be run automatically before you make commits, which is why they are sometimes called "pre-commit hooks". We use [pre-commit](https://pre-commit.com/) to run these checks.

To run pre-commit hooks manually:
```bash
poetry run pre-commit run -a
```

## Testing

SparseAutoschedulingBenchmark uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run the tests:

```bash
poetry install --with test
poetry run pytest
```

- Tests are located in the `tests/` directory at the project root.
- Write thorough tests for your new features and bug fixes.

### Optional Static Type Checking

Pytest runs mypy to check for type errors, so you shouldn't need to run it manually. In case you do need to run mypy manually, you can do so with:

```bash
poetry run mypy ./src/
```

### Regression Tests
pytest-regression is used to ensure that compiler outputs remain consistent across changes, and to better understand the impacts of compiler changes on the test outputs. To regenerate regression test outputs, run pytest with the `--regen-all` flag. Those who are curious can consult the [`pytest-regression` docs](https://pytest-regressions.readthedocs.io/en/latest/overview.html#using-data-regression).

**If you find an error or unclear section, please fix it or open an issue.**
