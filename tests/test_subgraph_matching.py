from pathlib import Path

from sparseappbench.benchmarks.subgraph_matching import benchmark_subgraph_matching, gcare_human
from sparseappbench.frameworks.numpy_framework import NumpyFramework
from sparseappbench.frameworks.sparse_framework import PyDataSparseFramework
from sparseappbench.binsparse_format import BinsparseFormat


def test_human_all():
    queries = gcare_human()
    xp = PyDataSparseFramework()
    results = benchmark_subgraph_matching(xp, queries)
    results = xp.from_benchmark(results)

    ground_truth_dir = Path('./data/gcare/ground_truth') / 'human'

    gts = []
    test_names = []
    for gt_file in ground_truth_dir.rglob('*.txt'):
        with open(gt_file, 'r') as f:
            test_names.append(f'{gt_file.parent}, {gt_file.stem}')
            gts.append(int(f.readline()))

    for i in range(len(results)):
        print(
            f'Test {test_names[i]}: Result = {results[i]}, Ground Truth = {gts[i]}, is_equal: {results[i] == gts[i]}')


if __name__ == '__main__':
    test_human_all()
