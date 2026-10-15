"""Model-free checks for Euler integration on the caller's time grid."""

import importlib.util
from pathlib import Path

import pytest
import torch

SCHEDULER_PATH = (
    Path(__file__).resolve().parents[1]
    / "kimia_infer/models/detokenizer/flow_matching/scheduler.py"
)
spec = importlib.util.spec_from_file_location("flow_matching_scheduler", SCHEDULER_PATH)
scheduler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler_module)
StreamingFlowMatchingScheduler = scheduler_module.StreamingFlowMatchingScheduler


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_step_broadcasts_over_state_dimensions(batch_size):
    scheduler = StreamingFlowMatchingScheduler(timesteps=4)
    state = torch.zeros(batch_size, 5, 7, dtype=torch.float64)
    velocity = torch.arange(batch_size, dtype=torch.float64).view(-1, 1, 1) + 1
    result = scheduler.step(state, velocity.expand_as(state))
    expected = state + (1 - scheduler.sigma_min) / 4 * velocity
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_sample_integrates_each_time_interval(batch_size):
    scheduler = StreamingFlowMatchingScheduler(timesteps=3)
    state = torch.zeros(batch_size, 5, 7, dtype=torch.float64)
    time_grid = torch.tensor([0.0, 0.2, 0.7, 1.0], dtype=torch.float64)
    evaluated_times = []

    def velocity(t, x):
        evaluated_times.append(float(t))
        return torch.full_like(x, 2 * t + 1)

    result = scheduler.sample(velocity, time_grid, state)
    # Explicit Euler: .2 * 1 + .5 * 1.4 + .3 * 2.4 = 1.62.
    torch.testing.assert_close(result, torch.full_like(state, 1.62))
    assert evaluated_times == time_grid[:-1].tolist()


@pytest.mark.parametrize("time_grid", [[0.0, 1.0], [0.4, 0.5, 0.8], [1.0, 0.7, 0.0]])
def test_constant_velocity_obeys_grid_endpoints(time_grid):
    scheduler = StreamingFlowMatchingScheduler(timesteps=15)
    state = torch.randn(2, 3, 7, dtype=torch.float64)
    time_grid = torch.tensor(time_grid, dtype=torch.float64)
    result = scheduler.sample(lambda t, x: torch.ones_like(x), time_grid, state)
    torch.testing.assert_close(result, state + time_grid[-1] - time_grid[0])


def test_single_time_point_requires_no_model_evaluation():
    scheduler = StreamingFlowMatchingScheduler(timesteps=15)
    state = torch.randn(2, 3, 7)

    def unexpected_evaluation(t, x):
        raise AssertionError("A grid with no intervals must not evaluate the velocity")

    result = scheduler.sample(unexpected_evaluation, torch.tensor([0.4]), state)
    torch.testing.assert_close(result, state)
