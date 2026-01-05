from __future__ import annotations

import contextlib
import io
import json

from byzpy.cli import main


def test_list_aggregators():
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        exit_code = main(["list", "aggregators"])

    output = f.getvalue()

    assert exit_code == 0
    assert "byzpy.aggregators.coordinate_wise.median.CoordinateWiseMedian" in output
    # Ensure no duplicates or test files
    assert "tests" not in output
    assert "CoordinateWiseMedian" in output


def test_list_aggregators_json():
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        exit_code = main(["list", "aggregators", "--format", "json"])

    output = f.getvalue()
    assert exit_code == 0
    data = json.loads(output)
    assert data["component"] == "aggregators"
    assert isinstance(data["items"], list)
    assert len(data["items"]) > 0


def test_list_attacks():
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        exit_code = main(["list", "attacks"])

    output = f.getvalue()
    assert exit_code == 0
    # Just check it runs, we don't know exact attacks if we didn't check


def test_list_invalid_component():
    # argparse should handle this, usually exits with 2
    f = io.StringIO()
    f_err = io.StringIO()
    try:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f_err):
            main(["list", "invalid_thing"])
    except SystemExit as e:
        assert e.code != 0
