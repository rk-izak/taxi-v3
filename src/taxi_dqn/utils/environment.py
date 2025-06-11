import gymnasium as gym


class Environment:
    """Used for storage of methods regarding the gym env."""
    @staticmethod
    def make_env(seed: int | None, render: bool):
        env = gym.make("Taxi-v3", render_mode="human" if render else None)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return env

    @staticmethod
    def set_state(env: gym.Env, taxi_row: int, taxi_col: int,
                passenger_idx: int, dest_idx: int) -> int:
        state = env.unwrapped.encode(taxi_row, taxi_col, passenger_idx, dest_idx)
        env.unwrapped.s = state
        return state
