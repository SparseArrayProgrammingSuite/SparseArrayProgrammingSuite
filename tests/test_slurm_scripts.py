from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Slurm scripts require a POSIX shell and filesystem"
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("upload_chunks", [None, "3"])
def test_refresh_logs_use_invocation_directory(tmp_path, upload_chunks):
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    submission = tmp_path / "submitted from 50%"
    submission.mkdir()
    shutil.copy(ROOT / "scripts/submit-refresh-jobs.sh", scripts)
    setup = scripts / "ensure-poetry-env.sh"
    setup.write_text("#!/usr/bin/env bash\nexit 0\n")
    setup.chmod(0o755)
    record = tmp_path / "submissions.jsonl"
    capture = (
        "import json, os, sys; "
        'f=open(os.environ["SAPS_TEST_SUBMISSIONS"], "a"); '
        'f.write(json.dumps(sys.argv[1:])+"\\n"); f.close(); print("12345")'
    )
    shell_env = tmp_path / "shell-env"
    shell_env.write_text(
        'aws() { if [[ "$1 $2" == "configure list-profiles" ]]; then '
        'printf "dataset-upload\\n"; fi; }\n'
        "poetry() { return 0; }\n"
        "sbatch() { "
        + shlex.quote(sys.executable)
        + " -c "
        + shlex.quote(capture)
        + ' "$@"; }\n'
    )
    env = {
        **os.environ,
        "BASH_ENV": str(shell_env),
        "SAPS_TEST_SUBMISSIONS": str(record),
    }
    env.pop("SAPS_UPLOAD_CHUNK_COUNT", None)
    if upload_chunks is not None:
        env["SAPS_UPLOAD_CHUNK_COUNT"] = upload_chunks
    subprocess.run(
        ["bash", str(scripts / "submit-refresh-jobs.sh")],
        cwd=submission,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    submissions = [json.loads(line) for line in record.read_text().splitlines()]
    names = ["upload-%A_%a.log", "trace-%A_%a.log", "finalize-metadata-%j.log"]
    assert len(submissions) == len(names)
    assert f"--array=0-{int(upload_chunks or '8') - 1}" in submissions[0]
    assert "--dependency=afterok:12345" in submissions[1]
    for args, name in zip(submissions, names, strict=True):
        expected = str(submission.resolve()).replace("%", "%%") + "/" + name
        assert args[args.index("--output") + 1] == expected
        assert args[args.index("--chdir") + 1] == str(scripts.parent.resolve())


COMPETITION_SCRIPTS = {"competition-cpu.slurm": "cpu", "competition-gpu.slurm": "gpu"}


@pytest.mark.parametrize("script_name,device", COMPETITION_SCRIPTS.items())
@pytest.mark.parametrize("forwarded", [[], ["--tag", "suite-train", "--re", "hosvd"]])
def test_competition_resume_uses_original_task_directory(
    tmp_path, script_name, device, forwarded
):
    run_root = tmp_path / "old run" / "run_12345"
    run_root.mkdir(parents=True)
    record = tmp_path / "commands.jsonl"
    capture = (
        "import json, os, sys; "
        'f=open(os.environ["SAPS_TEST_COMMANDS"], "a"); '
        'f.write(json.dumps({"args": sys.argv[1:], '
        '"pip_cache": os.environ["PIP_CACHE_DIR"], '
        '"virtualenv_cache": os.environ["VIRTUALENV_OVERRIDE_APP_DATA"]})'
        '+"\\n"); f.close()'
    )
    shell_env = tmp_path / "shell-env"
    shell_env.write_text(
        "poetry() { "
        + shlex.quote(sys.executable)
        + " -c "
        + shlex.quote(capture)
        + ' "$@"; }\n'
    )
    env = {
        **os.environ,
        "BASH_ENV": str(shell_env),
        "SAPS_TEST_COMMANDS": str(record),
        "SAPS_REPO_DIRECTORY": str(ROOT),
        "SLURM_ARRAY_JOB_ID": "99999",
        "SLURM_ARRAY_TASK_ID": "2",
        "SLURM_ARRAY_TASK_COUNT": "5",
        "TMPDIR": str(tmp_path / "local scratch"),
    }
    env.pop("SAPS_COMPETITION_ARGS", None)
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / script_name),
            *forwarded[:2],
            "--resume",
            str(run_root),
            *forwarded[2:],
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    run, combine = [call["args"] for call in calls]
    task_scratch = tmp_path / "local scratch" / "saps-competition-99999-2"
    assert task_scratch.is_dir()
    assert run[run.index("--env-dir") + 1] == str(task_scratch / "env")
    for call in calls:
        assert call["pip_cache"] == str(task_scratch / "pip-cache")
        assert call["virtualenv_cache"] == str(task_scratch / "virtualenv-cache")
    # CPU tasks keep the original task_<index> layout of older runs.
    task_label = "2" if device == "cpu" else f"{device}_2"
    task_directory = str(run_root.resolve() / f"task_{task_label}")
    assert "--resume" in run
    assert run[run.index("--saps-dir") + 1] == task_directory
    assert run[run.index("--results-dir") + 1] == task_directory + "/results"
    assert run[run.index("--machine") + 1] == f"run_12345-task-{task_label}"
    assert run[run.index("--device") + 1] == device
    # Wrapper arguments other than --resume reach the runner unchanged.
    assert run[len(run) - len(forwarded) :] == forwarded
    assert combine[:2] == ["run", "./scripts/combine_competition_results.py"]
    assert combine[combine.index("--run-directory") + 1] == str(run_root.resolve())
    assert "--output" not in combine
    for script in (ROOT / "scripts").glob("*.slurm"):
        text = script.read_text()
        assert "#SBATCH --output=" in text
        assert "#SBATCH --output=/dev/null" not in text
        assert "exec >" not in text


@pytest.mark.parametrize(
    "script_name,commands",
    [
        *(
            (script_name, ["run_benchmark.py", "combine_competition_results.py"])
            for script_name in COMPETITION_SCRIPTS
        ),
        ("upload-dataset.slurm", ["run_benchmark.py"]),
        ("trace-statistics.slurm", ["run_benchmark.py"]),
        ("finalize-metadata.slurm", ["merge_statistics.py", "generate_metadata.py"]),
    ],
)
def test_slurm_submission_from_scripts_directory(tmp_path, script_name, commands):
    # Slurm runs a copied script, so its own location cannot identify the repo.
    spooled_script = tmp_path / "slurm_script"
    shutil.copy(ROOT / "scripts" / script_name, spooled_script)
    record = tmp_path / "commands"
    shell_env = tmp_path / "shell-env"
    shell_env.write_text(
        "poetry() {\n"
        '  [[ "$PWD" == "$SAPS_TEST_ROOT" ]] || return 90\n'
        '  [[ "$1" == run && -f "$2" ]] || return 91\n'
        '  if [[ "$SAPS_TEST_SCRIPT" == upload-dataset.slurm ]]; then\n'
        '    [[ "$3 $4 $5" == "--cache-datasets --chunk-count 5" ]] || return 92\n'
        '    [[ "$6 $7" == "--chunk-index 0" ]] || return 93\n'
        "  fi\n"
        '  printf "%s\\n" "$2" >> "$SAPS_TEST_COMMANDS"\n'
        "}\n"
    )
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "statistics-0.json").write_text("{}")
    env = {
        **os.environ,
        "BASH_ENV": str(shell_env),
        "SAPS_TEST_ROOT": str(ROOT),
        "SAPS_TEST_COMMANDS": str(record),
        "SAPS_TEST_SCRIPT": script_name,
        "SLURM_SUBMIT_DIR": str(ROOT / "scripts"),
        "SLURM_ARRAY_JOB_ID": "12345",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_ARRAY_TASK_COUNT": "5",
        "SAPS_TRACE_CHUNK_COUNT": "1",
        "SAPS_TRACE_OUTPUT_DIR": str(trace_dir),
        "TMPDIR": str(tmp_path / "local scratch"),
    }
    env.pop("SAPS_REPO_DIRECTORY", None)
    env.pop("SAPS_COMPETITION_ARGS", None)
    subprocess.run(
        ["bash", str(spooled_script)],
        cwd=ROOT / "scripts",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert record.read_text().splitlines() == [
        f"./scripts/{name}"
        if name == "combine_competition_results.py"
        else f"./bin/{name}"
        for name in commands
    ]


@pytest.mark.parametrize("gpu_frameworks", [False, True])
def test_submit_competition_shares_run_between_cpu_and_gpu_arrays(
    tmp_path, gpu_frameworks
):
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(ROOT / "scripts/submit-competition.sh", scripts)
    include = [{"env_nobuild": {"SAPS_FRAMEWORK": "frameworks/saps_numpy.py"}}]
    if gpu_frameworks:
        include.append({"env_nobuild": {"SAPS_DEVICE": "gpu"}})
    config = tmp_path / "competition.config.json"
    config.write_text(json.dumps({"include": include}))
    record = tmp_path / "submissions.jsonl"
    # Each fake submission prints the next job id: 100, 101, ...
    capture = (
        "import json, os, sys; "
        'path=os.environ["SAPS_TEST_SUBMISSIONS"]; '
        "n=len(open(path).readlines()) if os.path.exists(path) else 0; "
        'f=open(path, "a"); f.write(json.dumps(sys.argv[1:])+"\\n"); f.close(); '
        "print(100 + n)"
    )
    shell_env = tmp_path / "shell-env"
    shell_env.write_text(
        "sbatch() { "
        + shlex.quote(sys.executable)
        + " -c "
        + shlex.quote(capture)
        + ' "$@"; }\n'
    )
    env = {
        **os.environ,
        "BASH_ENV": str(shell_env),
        "SAPS_TEST_SUBMISSIONS": str(record),
        "SAPS_COMPETITION_CONFIG": str(config),
        "SAPS_CHUNK_COUNT": "8",
    }
    subprocess.run(
        [
            "bash",
            str(scripts / "submit-competition.sh"),
            "--tag",
            "suite-train",
            "--after",
            "42",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    *arrays, combine = [json.loads(line) for line in record.read_text().splitlines()]
    devices = ["cpu", "gpu"] if gpu_frameworks else ["cpu"]
    run_directory = scripts.parent.resolve() / "competition" / "run_100"
    assert run_directory.is_dir()
    assert len(arrays) == len(devices)
    for device, args in zip(devices, arrays, strict=True):
        wrapper = args.index(str(scripts / f"competition-{device}.slurm"))
        assert "--array=0-7" in args
        assert "--dependency=afterok:42" in args
        assert args[-2:] == ["--tag", "suite-train"]
        # The GPU array joins the run directory named after the CPU array.
        if device == "gpu":
            assert args[wrapper + 1 : wrapper + 3] == ["--resume", str(run_directory)]
        else:
            assert "--resume" not in args
    job_ids = ":".join(str(100 + i) for i in range(len(arrays)))
    assert f"--dependency=afterany:{job_ids}" in combine
    assert str(run_directory) in combine[combine.index("--wrap") + 1]


def test_submit_competition_cancels_earlier_arrays_when_a_submission_fails(tmp_path):
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(ROOT / "scripts/submit-competition.sh", scripts)
    config = tmp_path / "competition.config.json"
    config.write_text(
        json.dumps({"include": [{"env_nobuild": {"SAPS_DEVICE": "gpu"}}]})
    )
    cancelled = tmp_path / "cancelled"
    # The CPU array submits as job 100; the GPU array is rejected.
    shell_env = tmp_path / "shell-env"
    shell_env.write_text(
        'sbatch() { if [[ -e "$SAPS_TEST_SUBMITTED" ]]; then return 1; fi; '
        'touch "$SAPS_TEST_SUBMITTED"; echo 100; }\n'
        'scancel() { echo "$@" >> "$SAPS_TEST_CANCELLED"; }\n'
    )
    env = {
        **os.environ,
        "BASH_ENV": str(shell_env),
        "SAPS_TEST_SUBMITTED": str(tmp_path / "submitted"),
        "SAPS_TEST_CANCELLED": str(cancelled),
        "SAPS_COMPETITION_CONFIG": str(config),
    }
    result = subprocess.run(
        ["bash", str(scripts / "submit-competition.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert cancelled.read_text().split() == ["100"]
