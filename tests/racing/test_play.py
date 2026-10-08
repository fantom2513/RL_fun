import os
import subprocess
import sys

import numpy as np
import pytest

from rl_fun.racing.play import action_from_keys, main, parse_args


def test_left_key_steers_left_and_up_key_accelerates():
    action = action_from_keys(left=True, right=False, up=True, down=False)

    assert action.dtype == np.float32
    assert action.tolist() == [1.0, 1.0]


def test_opposite_keys_cancel_out():
    action = action_from_keys(left=True, right=True, up=True, down=True)

    assert action.tolist() == [0.0, 0.0]


def test_right_and_down_keys_steer_right_and_brake():
    action = action_from_keys(left=False, right=True, up=False, down=True)

    assert action.tolist() == [-1.0, -1.0]


def test_parse_args_defaults():
    arguments = parse_args([])

    assert arguments.track == "oval"
    assert arguments.max_steps is None
    assert arguments.max_frames is None


def test_parse_args_reads_options():
    arguments = parse_args(["wavy", "--max-steps", "50", "--max-frames", "2"])

    assert (arguments.track, arguments.max_steps, arguments.max_frames) == ("wavy", 50, 2)


def test_main_runs_a_few_frames_in_a_dummy_window():
    assert main(["--max-frames", "3"]) == 0


def test_main_rejects_unknown_track():
    with pytest.raises(SystemExit) as error:
        main(["no-such-track", "--max-frames", "1"])

    assert error.value.code == 2


def test_script_help_is_in_russian():
    result = subprocess.run(
        [sys.executable, "scripts/play.py", "--help"],
        capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=60, check=False,
    )

    assert result.returncode == 0
    assert "трасс" in result.stdout.lower()
