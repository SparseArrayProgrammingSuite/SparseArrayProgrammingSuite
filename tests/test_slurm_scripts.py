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


def test_refresh_logs_use_invocation_directory(tmp_path):
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
    subprocess.run(
        ["bash", str(scripts / "submit-refresh-jobs.sh")],
        cwd=submission,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    submissions = [json.loads(line) for line in record.read_text().splitlines()]
    names = ["upload-%j.log", "trace-%A_%a.log", "finalize-metadata-%j.log"]
    assert len(submissions) == len(names)
    for args, name in zip(submissions, names, strict=True):
        expected = str(submission.resolve()).replace("%", "%%") + "/" + name
        assert args[args.index("--output") + 1] == expected
        assert args[args.index("--chdir") + 1] == str(scripts.parent.resolve())


def test_competition_resume_uses_original_task_directory(tmp_path):
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
            str(ROOT / "scripts/run-competition.slurm"),
            "--resume",
            str(run_root),
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
    task_directory = str(run_root.resolve() / "task_2")
    assert "--resume" in run
    assert run[run.index("--saps-dir") + 1] == task_directory
    assert run[run.index("--results-dir") + 1] == task_directory + "/results"
    assert run[run.index("--machine") + 1] == "run_12345-task-2"
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
        (
            "run-competition.slurm",
            ["run_benchmark.py", "combine_competition_results.py"],
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
