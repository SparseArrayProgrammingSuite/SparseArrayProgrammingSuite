# SparseApplicationBenchmark

Sparse array programming frameworks, such as [SciPy](https://scipy.org) or [pydata/sparse](https://sparse.pydata.org/en/stable/), are getting more advanced. Because the performance and optimization strategies for sparse frameworks depends heavily on the input sparsity patterns and programs, we need realistic applications to make informed design decisions. This benchmark suite consists of several applications written entirely using collective operations (such as +, *, sum, or reduce) over sparse arrays. The programs are adapted from real-world applications using a straightforward translation to an [array-programming](https://en.wikipedia.org/wiki/Array_programming#Array_languages) style. The standard form for our benchmark functions is vanilla python code using only [Array-API]( https://data-apis.org/array-api/latest/API_specification/) functions, with minimal looping or control flow.

We take great inspiration from the great benchmarks in the database community, such as [pandasbench](https://arxiv.org/abs/2506.02345), [Join Order Benchmark](https://dl.acm.org/doi/10.14778/2850583.2850594), or [TPC-H](https://www.tpc.org/tpch/)

## Contributing
Start with the [policy doc](https://docs.google.com/document/d/1N5gElU3Z_URG-K4HTdLlf_H1lpI42dH9xVdeBs4ierA/edit?usp=sharing), which describes the policies and processes by which you can contribute benchmarks to the repo.

Most importantly: Before implementing a benchmark, claim it! Select or create your own github issue describing which application you want to benchmark, with links to relevant source code, and assign yourself to the issue if possible so that others know you're working on that benchmark. You can only claim a benchmark for a maximum of 1 month, after which others can claim it.

Once you're on board, see [CONTRIBUTING.md](CONTRIBUTING.md) for software guidelines, development setup, and best practices.

## Installation

SparseApplicationBenchmark uses [poetry](https://python-poetry.org/) for packaging. To install for
development, clone the repository and run:
```bash
poetry install --extras test
```
to install the current project and dev dependencies.

# Organization

## Dataset tagging

- Datasets represent one "run" of a benchmark.
- Generators group similar generation logic.
- Benchmarks are the algorithms being run.
- Tags can be applied to datasets, generators, or benchmarks. If a generator or benchmark is tagged, the dataset inherits that tag.

### Test sets

- `test`: datasets that run in CI and cover dense/sparse cases, different sizes, and code paths.
- `standard`: canonical datasets in the suite.
- `dynamic`: datasets with random seeds or random kernels.
- `AI`: AI-generated datasets.

### Domains

Each dataset should be tagged with a problem domain from the [ACM Computing Classification System](https://dl.acm.org/ccs), converted to lowercase with hyphens, for example:

- `applied-computing`
- `physical-sciences-and-engineering`
- `aerospace`

Manual benchmark suite tags are written to metadata verbatim under `suites`. Topic tags generated from ACM CCS XML are written under `topics`, and trace-derived tags are written under `statistics`; benchmark selection builds its tag set from all three fields.

Every benchmark must provide `ccs_xml`. You can use the [ACM CCS 2012 generator](https://dl.acm.org/ccs/ccs.cfm), copy its XML output, and paste it into a benchmark as `ccs_xml`. SAPS converts each `concept_desc` path component into `topics`:

```python
@property
def ccs_xml(self) -> str:
    return """
    <ccs2012>
      <concept>
        <concept_desc>Applied computing~Physical sciences and engineering~Aerospace</concept_desc>
      </concept>
    </ccs2012>
    """
```

Run `poetry run ./bin/run_benchmark.py --generate-metadata` after changing benchmark metadata, then run `poetry run ./bin/run_benchmark.py --generate-topics` to replace the generated `topics` field without changing `suites` or generated `statistics`.

### Problem qualities

Tags describe the general character of a problem:

- `high-dimensional`: 5 or more dimensions in a tensor.
- `tensor`: 3 or more dimensions in a tensor.
- `large-query`: 5 or more operands on one line.
- `elementary-ops`: PEMDAS-only.
- `transcendental-ops`: contains sin, cos, pow, exp, etc.
- `shape-ops`: reshape, concat, or similar operations.
- `linalg-ops`: contains `np.linalg` or solver-like operations (`dot` is okay).
- `fancy-ops`: min, max, and, or, shift.
- `index-ops`: contains indexing.
- `nonzero-fill`: uses a fill value other than zero.
- `iterative`: loops over a matrix.
- `dense`: exclusively dense problems.
- `hypersparse`: contains hypersparsity (`nnz << n` for a dimension).
- `dynamic-sparsity`: sparse-sparse interactions.
