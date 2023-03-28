
from ast import Dict
from typing import Any, Optional, Tuple
import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.wrappers.compatibility import LegacyEnv
from gymnasium.utils.step_api_compatibility import convert_to_done_step_api


class EnvBackCompability(gym.Env):
    r"""A wrapper which can transform an environment from the new API to the old API.

    Old step API refers to step() method returning (observation, reward, done, info), and reset() only retuning the observation.
    New step API refers to step() method returning (observation, reward, terminated, truncated, info) and reset() returning (observation, info).
    (Refer to docs for details on the API change)

    Known limitations:
    - Environments that use `self.np_random` might not work as expected.
    """

    def __init__(self, new_env):
        """A wrapper which converts new-style envs to valid old envs.

        Args:
            old_env: the env to wrap, implemented with the new API
        """
        self.metadata = getattr(new_env, "metadata", {"render_modes": []})
        self.env = new_env

        self.observation_space = new_env.observation_space
        self.action_space = new_env.action_space

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[LegacyEnv, dict]:
        """Resets the environment.

        Args:
            seed: the seed to reset the environment with

        Returns:
            (observation)
        """
        if seed is not None:
            self.env.seed(seed)

        if self.render_mode == "human":
            self.render()

        obs, _ = self.env.reset()

        return obs

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, info)
        """
        r = self.env.step(action)

        if self.env.render_mode == "human":
            self.render()

        return convert_to_done_step_api(r)  # type: ignore

    def render(self, mode: Optional[str] = None) -> Any:
        """Renders the environment.

        Args:
            mode: override the render mode
        Returns:
            The rendering of the environment, depending on the render mode
        """
        if mode is not None:
            assert self.env.render_mode == mode
        return self.env.render()

    def close(self):
        """Closes the environment."""
        self.env.close()

    def __str__(self):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)
