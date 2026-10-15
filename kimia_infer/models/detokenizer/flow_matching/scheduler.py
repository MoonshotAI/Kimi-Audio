import torch
from abc import abstractmethod, ABC

try:
    from torchdyn.core import NeuralODE

    NEURALODE_INSTALLED = True
except ImportError:
    NEURALODE_INSTALLED = False


class SchedulerBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def set_timesteps(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def add_noise(self):
        pass


class StreamingFlowMatchingScheduler(SchedulerBase):
    def __init__(
        self,
        timesteps=1000,
        sigma_min=1e-4,
    ) -> None:
        super().__init__()

        self.sigma_min = sigma_min
        self.timesteps = timesteps
        self.t_min = 0
        self.t_max = 1 - self.sigma_min

        self.neural_ode = None

    def set_timesteps(self, timesteps=15):
        self.timesteps = timesteps

    def step(self, xt, predicted_v):

        h = (self.t_max - self.t_min) / self.timesteps

        xt = xt + h * predicted_v
        return xt

    def sample(self, ode_wrapper, time_steps, xt, verbose=False, x0=None):
        if verbose:
            gt_v = x0 - xt

        # time_steps contains integration endpoints, as in NeuralODE. Evaluate
        # the velocity at the left endpoint of each interval, never after the
        # final endpoint, and retain non-uniform (or descending) grid spacing.
        for t, t_next in zip(time_steps[:-1], time_steps[1:]):
            h = (t_next - t).to(device=xt.device, dtype=xt.dtype)
            predicted_v = ode_wrapper(t, xt)
            if verbose:
                dist = torch.mean(torch.nn.functional.l1_loss(gt_v, predicted_v))
                print("Time: {}, Distance: {}".format(t, dist))
            xt = xt + h * predicted_v
        return xt

    def sample_by_neuralode(self, ode_wrapper, time_steps, xt, verbose=False, x0=None):
        if not NEURALODE_INSTALLED:
            raise ImportError("NeuralODE is not installed, please install it first.")

        if self.neural_ode is None:
            self.neural_ode = NeuralODE(
                ode_wrapper,
                solver="euler",
                sensitivity="adjoint",
                atol=self.sigma_min,
                rtol=self.sigma_min,
            )

        eval_points, traj = self.neural_ode(xt, time_steps)
        return traj[-1]

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ):
        ut = original_samples - (1 - self.sigma_min) * noise  # 和ut的梯度没关系
        t_unsqueeze = timesteps.unsqueeze(1).unsqueeze(1).float() / self.timesteps
        x_noisy = (
            t_unsqueeze * original_samples
            + (1.0 - (1 - self.sigma_min) * t_unsqueeze) * noise
        )
        return x_noisy, ut
