# ISCA patch of dino_wm/env/__init__.py: guard the eager pointmaze import
# (pulls gym.envs.mujoco -> mujoco_py, not installed here). PushT-only port;
# registration entry_points stay lazy, so the other envs remain registered.
from gym.envs.registration import register

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

try:
    from .pointmaze import U_MAZE

    register(
        id='point_maze',
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec': U_MAZE,
            'reward_type': 'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
        }
    )
except Exception as e:  # mujoco_py not installed (pusht-only port)
    print(f"[env] point_maze disabled: {e}")

register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
