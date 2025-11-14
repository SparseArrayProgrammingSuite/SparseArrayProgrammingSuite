import argparse
import time

from .benchmarks.cg import (
    benchmark_cg,
    dg_cg_sparse_1,
    dg_cg_sparse_2,
    dg_cg_sparse_3,
    dg_cg_sparse_4,
    dg_cg_sparse_5,
    dg_cg_sparse_6,
    dg_cg_sparse_7,
    dg_cg_sparse_8,
)
from .benchmarks.jacobi import (
    benchmark_jacobi,
    dg_jacobi_sparse_1,
    dg_jacobi_sparse_2,
    dg_jacobi_sparse_3,
    dg_jacobi_sparse_4,
    dg_jacobi_sparse_5,
    dg_jacobi_sparse_6,
    dg_jacobi_sparse_7,
    dg_jacobi_sparse_8,
)
from .benchmarks.matmul import (
    benchmark_matmul,
    dg_matmul_dense_large,
    dg_matmul_dense_small,
    dg_matmul_sparse_large,
    dg_matmul_sparse_small,
)
from .benchmarks.preconditioned_cg import (
    benchmark_block_jacobi_cg,
    benchmark_jacobi_cg,
    dg_precond_cg_sparse_1,
    dg_precond_cg_sparse_2,
    dg_precond_cg_sparse_3,
    dg_precond_cg_sparse_4,
    dg_precond_cg_sparse_5,
    dg_precond_cg_sparse_6,
    dg_precond_cg_sparse_7,
    dg_precond_cg_sparse_8,
)
from .frameworks.checker_framework import CheckerFramework
from .frameworks.numpy_framework import NumpyFramework

FRAMEWORK_DICT = {"numpy": NumpyFramework(), "checker": CheckerFramework()}
BENCHMARK_DICT = {
    "matmul": benchmark_matmul,
    "jacobi": benchmark_jacobi,
    "cg": benchmark_cg,
    "jacobi_cg": benchmark_jacobi_cg,
    "block_jacobi_cg": benchmark_block_jacobi_cg,
}
DATA_GENERATOR_DICT = {
    "matmul": {
        "matmul_dense_small": dg_matmul_dense_small,
        "matmul_dense_large": dg_matmul_dense_large,
        "matmul_sparse_small": dg_matmul_sparse_small,
        "matmul_sparse_large": dg_matmul_sparse_large,
    },
    "jacobi": {
        "jacobi_sparse_1": dg_jacobi_sparse_1,
        "jacobi_sparse_2": dg_jacobi_sparse_2,
        "jacobi_sparse_3": dg_jacobi_sparse_3,
        "jacobi_sparse_4": dg_jacobi_sparse_4,
        "jacobi_sparse_5": dg_jacobi_sparse_5,
        "jacobi_sparse_6": dg_jacobi_sparse_6,
        "jacobi_sparse_7": dg_jacobi_sparse_7,
        "jacobi_sparse_8": dg_jacobi_sparse_8,
    },
    "cg": {
        "cg_sparse_1": dg_cg_sparse_1,
        "cg_sparse_2": dg_cg_sparse_2,
        "cg_sparse_3": dg_cg_sparse_3,
        "cg_sparse_4": dg_cg_sparse_4,
        "cg_sparse_5": dg_cg_sparse_5,
        "cg_sparse_6": dg_cg_sparse_6,
        "cg_sparse_7": dg_cg_sparse_7,
        "cg_sparse_8": dg_cg_sparse_8,
    },
    "jacobi_cg": {
        "jacobi_cg_sparse_1": dg_precond_cg_sparse_1,
        "jacobi_cg_sparse_2": dg_precond_cg_sparse_2,
        "jacobi_cg_sparse_3": dg_precond_cg_sparse_3,
        "jacobi_cg_sparse_4": dg_precond_cg_sparse_4,
        "jacobi_cg_sparse_5": dg_precond_cg_sparse_5,
        "jacobi_cg_sparse_6": dg_precond_cg_sparse_6,
        "jacobi_cg_sparse_7": dg_precond_cg_sparse_7,
        "jacobi_cg_sparse_8": dg_precond_cg_sparse_8,
    },
    "block_jacobi_cg": {
        "block_jacobi_cg_sparse_1": dg_precond_cg_sparse_1,
        "block_jacobi_cg_sparse_2": dg_precond_cg_sparse_2,
        "block_jacobi_cg_sparse_3": dg_precond_cg_sparse_3,
        "block_jacobi_cg_sparse_4": dg_precond_cg_sparse_4,
        "block_jacobi_cg_sparse_5": dg_precond_cg_sparse_5,
        "block_jacobi_cg_sparse_6": dg_precond_cg_sparse_6,
        "block_jacobi_cg_sparse_7": dg_precond_cg_sparse_7,
        "block_jacobi_cg_sparse_8": dg_precond_cg_sparse_8,
    },
}


def run_benchmark(framework, benchmark_function, benchmark_data_generator, iters):
    execution_times = []
    for _ in range(iters):
        data = benchmark_data_generator()
        start = time.perf_counter()
        benchmark_function(framework, *data)
        end = time.perf_counter()
        duration = end - start
        execution_times.append(duration)
    print(
        f"Benchmark took an average of {sum(execution_times) / len(execution_times)}\
             seconds"
    )
    return execution_times


def save_benchmark_results(
    results_folder, execution_times, framework, benchmark, data_generator
):
    filename = f"{results_folder}/{framework}_{benchmark}_{data_generator}.csv"
    with open(filename, "w") as f:
        f.write("Framework,Benchmark,Data Generator,Iteration,ExecutionTime\n")
        for i, execution_time in enumerate(execution_times):
            f.write(f"{framework},{benchmark},{data_generator},{i},{execution_time}\n")


# This function allows either command line arguments or direct function calls to
# run benchmarks. Note that if frameworks, benchmarks, or data_generators are
# provided directly, the corresponding command line arguments are ignored. Further,
# if a user benchmark is provided, data generators must also be provided directly.
def main(
    frameworks=None,
    framework_names=None,
    benchmarks=None,
    benchmark_names=None,
    data_generators=None,
    data_generator_names=None,
    iters=None,
    results_folder=None,
    args=None,
):
    collected_frameworks = FRAMEWORK_DICT.copy()
    if frameworks is not None:
        for framework_name, framework in frameworks.items():
            collected_frameworks[framework_name] = framework
    frameworks = collected_frameworks

    collected_benchmarks = BENCHMARK_DICT.copy()
    if benchmarks is not None:
        for benchmark_name, benchmark in benchmarks.items():
            collected_benchmarks[benchmark_name] = benchmark
    benchmarks = collected_benchmarks

    collected_data_generators = {
        benchmark_name: generators.copy()
        for benchmark_name, generators in DATA_GENERATOR_DICT.copy().items()
    }
    if data_generators is not None:
        for benchmark_name, generators in data_generators.items():
            for generator_name, generator in generators.items():
                collected_data_generators[benchmark_name][generator_name] = generator
    data_generators = collected_data_generators

    parser = argparse.ArgumentParser(description="Run sparse autoscheduling benchmark")
    parser.add_argument(
        "--framework",
        default=["all"],
        nargs="*",
        help="Execution framework(s) to use",
    )
    parser.add_argument(
        "--benchmark",
        default=["all"],
        nargs="*",
        help="Benchmark(s) to run",
    )
    parser.add_argument(
        "--data-generator",
        default=["all"],
        nargs="*",
        help="Data generator(s) to use",
    )
    parser.add_argument(
        "--iterations",
        default=5,
        type=int,
        help="Number of iterations to run for each benchmark",
    )
    parser.add_argument(
        "--results-folder", default="results", help="Folder to save results"
    )
    args = parser.parse_args(args)

    if framework_names is None:
        if args.framework == ["all"]:
            framework_names = list(FRAMEWORK_DICT.keys())
        else:
            framework_names = args.framework

    if benchmark_names is None:
        if args.benchmark == ["all"]:
            benchmark_names = list(BENCHMARK_DICT.keys())
        else:
            benchmark_names = args.benchmark

    if data_generator_names is None:
        if args.data_generator == ["all"]:
            data_generator_names = [
                generator_name
                for generators in collected_data_generators.values()
                for generator_name in generators
            ]
        else:
            data_generator_names = args.data_generator

    if results_folder is None:
        results_folder = args.results_folder

    if iters is None:
        iters = args.iterations

    data_generator_names = set(data_generator_names)
    for framework_name in framework_names:
        framework = frameworks[framework_name]
        for benchmark_name in benchmark_names:
            benchmark = benchmarks[benchmark_name]
            for data_generator_name, data_generator in data_generators[
                benchmark_name
            ].items():
                if data_generator_name not in data_generator_names:
                    continue
                print(
                    f"Running benchmark {benchmark_name} with framework\
                          {framework_name} and data generator {data_generator_name}"
                )
                execution_times = run_benchmark(
                    framework, benchmark, data_generator, iters
                )
                save_benchmark_results(
                    results_folder,
                    execution_times,
                    framework_name,
                    benchmark_name,
                    data_generator_name,
                )
