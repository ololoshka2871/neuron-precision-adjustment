from gymnasium.envs.registration import register

register(
    id="gym_quarz/QuartzEnv-v6",
    entry_point="gym_quarz.envs:QuartzEnv6",
)