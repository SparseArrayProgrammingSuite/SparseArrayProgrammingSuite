import sparseappbench.benchmarks.subgraph_matching as subgraph_matching
from sparseappbench.frameworks.numpy_framework import NumpyFramework
from sparseappbench.frameworks.sparse_framework import PyDataSparseFramework


def test_human_subset():
    xp = PyDataSparseFramework()
    subgraph_matching.xp = xp
    dataset = subgraph_matching.SubgraphGCareGenerator().datasets[0]
    (matrices, meta) = subgraph_matching.SubgraphGCareGenerator().generate(dataset)
    # print(f'Meta: {meta}')

    results = subgraph_matching.SubgraphMatching().benchmark(matrices, meta)
    results = xp.from_binsparse(results)

    for i in range(len(results)):
        res = results[i]
        gt = meta['gts'][i]
        # print(f'Test {meta["names"][i]}: Result = {res}, Ground Truth = {gt}')
        assert res == gt, f'Test {meta["names"][i]} incorrect: Result = {res}, Ground Truth = {gt}'

if __name__ == "__main__":
    test_human_subset()