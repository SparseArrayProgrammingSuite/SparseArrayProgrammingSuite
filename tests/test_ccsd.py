from types import SimpleNamespace

import pytest

import numpy as np

from binsparse.conversions import from_numpy, to_numpy

from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_smart import SmartSparseFramework
from saps.benchmark import Param
from saps.benchmarks.ccsd import CCSDBenchmark, CCSDSyntheticGenerator

# |T| printed by the C++ CTF reference, tests/cpp_reference/coupled_cluster/ccsd.cxx,
# for no=4 and nv=6, to six decimal places.
CTF_NORM = 380638.269079


@pytest.mark.parametrize("framework", [NumpyFramework, SmartSparseFramework])
@pytest.mark.parametrize("name", ["small", "medium", "large"])
def test_framework_outputs_pass_the_bound_check(framework, name):
    benchmark = CCSDBenchmark()
    generator = CCSDSyntheticGenerator()
    dataset = next(d for d in generator.datasets if d.name == name)
    param = Param(generator, dataset)
    benchmark.setup(param, use_cache=False, xp=framework())
    benchmark.run(param)
    benchmark.teardown(param)


def test_reference_reproduces_the_ctf_norm():
    generator = CCSDSyntheticGenerator()
    problem = generator.generate(generator.datasets[0])
    T1, T2, T1_error, T2_error = (to_numpy(t) for t in problem.ref_outputs)
    norm = np.linalg.norm(T1) + np.linalg.norm(T2)
    tolerance = 5e-7 + np.linalg.norm(T1_error) + np.linalg.norm(T2_error)
    # The two norms' own rounding is far below the printed precision.
    assert abs(norm - CTF_NORM) <= tolerance + 1e-12 * norm


def test_check_rejects_a_sign_flip_that_keeps_the_norm():
    benchmark = CCSDBenchmark()
    generator = CCSDSyntheticGenerator()
    problem = generator.generate(generator.datasets[0])
    benchmark._ref_outputs = problem.ref_outputs
    T1, T2 = (to_numpy(t) for t in problem.ref_outputs[:2])
    T2_error = to_numpy(problem.ref_outputs[3])
    param = SimpleNamespace(dataset=SimpleNamespace(name="small"))

    nudged = T2 + 0.5 * T2_error
    benchmark._output = [from_numpy(T1), from_numpy(nudged)]
    benchmark.check(param)

    flipped = T2.copy()
    flipped[0, 1, 0, 1] = -flipped[0, 1, 0, 1]
    assert flipped[0, 1, 0, 1] != 0
    benchmark._output = [from_numpy(T1), from_numpy(flipped)]
    with pytest.raises(AssertionError, match="leaves its error bound"):
        benchmark.check(param)
