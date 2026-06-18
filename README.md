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
poetry install --with test
```
to install the current project and dev dependencies.



For example, there should be a "default" set of benchmarks, perhaps an "extended" set, or maybe a "pathological" set. At the very least, the default set of inputs should be specified in addition to more precise inputs.

It would be great to also have a set of really small example datasets just for running the benchmarks in CI to check they still work

Datasets should be understood as one "run" of a benchmark
Generators are just a code convenience thing to group similar generation logic together
Benchmarks are the algorithm being run.

Ultimately, each dataset needs to be tagged. if a dataset generator or benchmark is tagged instead, the dataset should be understood as "inheriting" the tag.

Each dataset should be tagged with one of the following test sets (headliner test sets that we report on)

    test - a tag for datasets that run in CI, we should have enough test datasets to get code coverage on all paths of generators and benchmarks (eg. dense and sparse, diff sizes, etc., whatever you have to do)
    standard - a tag for datasets in the standard, canonical, saps suite
    Future work tags in this group:
        dynamic - a tag for datasets that have random seeds, and/or random kernels (currently not used but we should document it for future work)
        AI - A tag for an AI-generated suite

Domains: each dataset should be tagged with a problem domain (that is, which college university department would handle this data or we can make tags from https://dl.acm.org/ccs) what follows is a list of examples

    computational chemistry
    image processing
    healthcare
    physics
    data analytics

Problem Qualities: what "kind" of problem is it, what is the general character of the problem

    high-dimensional
        more than 5 dims in a tensor
    tensor
        more than 3 dims in a tensor
    large-query
        more than 5 operands on one line
    elementary-ops
        pemdas-only
    transcendental-ops
        contains sin, cos, pow, exp, etc...
    shape-ops
        reshape, cat
    linalg-ops
        contains np.linalg stuff (in particular, solvers. dot is okay)
    fancy-ops
        min, max, and, or, shift
    index-ops
        contains indexing
    nonzero-fill
        any tensor uses a fill value other than zero
    iterative
        for loop over a matrix
    dense
        exclusively dense problem
    hypersparse
        contains hypersparsity (nnz << n for any dimension n in the tensor)
    dynamic-sparsity
        sparse-sparse interactions
