"""Worker progress and per-dataset outcomes retained alongside ASV results."""

import itertools
import json
import logging
import math
import os
import sys

STATUS_PREFIX = "saps: status "


class MeasurementTimeout(TimeoutError):
    """Warmup and measurements exceeded their elapsed-time allowance."""


def report_status(benchmark: str, dataset: str, phase: str, **details) -> None:
    message = STATUS_PREFIX + json.dumps(
        {"benchmark": benchmark, "dataset": dataset, "phase": phase, **details}
    )
    print(message, file=sys.stderr, flush=True)
    if os.environ.get("SAPS_LOG_PATH"):
        logging.getLogger("saps.work").info(message)


def parameter_statuses(params, values, output, selected=None) -> list[dict]:
    # ASV combines exit codes across parameters, but labels each parameter's
    # output separately. Use those sections to attribute timeouts correctly.
    sections: dict[str, list[str]] = {"": []}
    lines = sections[""]
    for line in (output or "").splitlines():
        if line.startswith("For parameters: "):
            lines = sections.setdefault(line.removeprefix("For parameters: "), [])
        else:
            lines.append(line)

    records = []
    for index, parameters in enumerate(itertools.product(*params)):
        if selected is not None and index not in selected:
            continue
        phase = "setup"
        process_timed_out = False
        measurement_timeout = None
        for line in sections.get(", ".join(parameters), sections[""]):
            if line.startswith("asv: benchmark timed out (timeout "):
                process_timed_out = True
            if not line.startswith(STATUS_PREFIX):
                continue
            try:
                event = json.loads(line[len(STATUS_PREFIX) :])
            except json.JSONDecodeError:
                continue  # A process can die partway through a marker.
            if not isinstance(event, dict):
                continue
            if event.get("phase") in ("setup", "measurement", "teardown", "finished"):
                phase = event["phase"]
            if event.get("status") == "measurement_timeout":
                measurement_timeout = event

        value = values[index]
        record: dict = {"parameters": list(parameters), "phase": phase}
        if measurement_timeout is not None:
            record.update(
                status="measurement_timeout",
                timeout_source="measurement",
                elapsed_seconds=measurement_timeout.get("elapsed_seconds"),
                timeout_seconds=measurement_timeout.get("timeout_seconds"),
            )
        elif process_timed_out:
            record.update(
                status="teardown_timeout"
                if phase == "finished"
                else phase + "_timeout",
                timeout_source="process",
            )
        elif value is None:
            record["status"] = "failed"
        elif math.isnan(value):
            record["status"] = "skipped"
        elif math.isinf(value):
            record.update(status="failed", reason="nonfinite")
        else:
            record["status"] = "ok"
        records.append(record)
    return records
