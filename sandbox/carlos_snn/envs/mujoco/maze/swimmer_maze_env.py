from sandbox.carlos_snn.envs.mujoco.maze.fast_maze_env import FastMazeEnv  # %^&*&^%
from sandbox.carlos_snn.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.core.serializable import Serializable


class SwimmerMazeEnv(FastMazeEnv, Serializable):

    # MODEL_CLASS = normalize(SwimmerEnv)
    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 3
    MAZE_MAKE_CONTACTS = True

    # # this is not needed, but on without the stub method I can't run a SwimmerMazeEnv anymore!
    # def __init__(self, **kwargs):
    #     Serializable.quick_init(self, locals())
    #     super(SwimmerMazeEnv, self).__init__(**kwargs)

