from __future__ import annotations

import torch

from byzpy.engine.parameter_server.runner import ParameterServerRunner


def test_parameter_server_runner_mean():
    grads = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
    ]
    runner = ParameterServerRunner(worker_grad_fns=[lambda g=g: g for g in grads])
    runner.start()
    try:
        out = runner.run_round()
        assert torch.allclose(out, sum(grads) / len(grads))
    finally:
        runner.stop()
