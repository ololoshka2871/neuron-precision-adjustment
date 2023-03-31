from gymnasium.envs.registration import register

register(
    id="gym_quarz/QuartzEnv-v4",
    entry_point="gym_quarz.envs:QuartzEnv4",
)

register(
    id="gym_quarz/QuartzEnv-v5",
    entry_point="gym_quarz.envs:QuartzEnv5",
)
