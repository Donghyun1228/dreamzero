"""
URGym environment definition.
"""
# ruff: noqa

import time

import gym
import numpy as np

try:
    from realworld.ur_controller.ur import URClient
except ModuleNotFoundError:
    from ur_controller.ur import URClient


def wait_for_obs(ur_client):
    """Fetches an observation from the URClient."""
    obs = ur_client.get_observation()
    while obs is None:
        print("Waiting for observations...")
        obs = ur_client.get_observation()
        time.sleep(1)
    return obs


def convert_obs(obs, im_size):
    """Preprocesses image and proprio observations."""
    image_obs = np.transpose(obs["image"], (2, 0, 1))
    wrist_image_obs = np.transpose(obs["wrist_image"], (2, 0, 1))

    return {
        "image": image_obs,
        "wrist_image": wrist_image_obs,
        "full_image": obs["full_image"],
        "joint_position": obs["joint_position"],
        "cartesian_position": obs["cartesian_position"],
        "gripper_position": obs["gripper_position"],
        "proprio": obs["proprio"],
    }


def null_obs(img_size):
    """Returns a dummy observation with all-zero image and proprio."""
    return {
        "image": np.zeros((3, img_size, img_size), dtype=np.uint8),
        "wrist_image": np.zeros((3, img_size, img_size), dtype=np.uint8),
        "full_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "joint_position": np.zeros((6,), dtype=np.float64),
        "cartesian_position": np.zeros((6,), dtype=np.float64),
        "gripper_position": np.zeros((1,), dtype=np.float64),
        "proprio": np.zeros((7,), dtype=np.float64),
    }


class URGym(gym.Env):
    """
    A Gym environment for the UR controller provided by:
    """

    def __init__(
        self,
        ur_client: URClient,
        cfg: dict,
        im_size: int = 224,
        blocking: bool = True,
    ):
        self.ur_client = ur_client
        self.im_size = im_size
        self.blocking = blocking
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=np.zeros((3, im_size, im_size)),
                    high=255 * np.ones((3, im_size, im_size)),
                    dtype=np.uint8,
                ),
                "wrist_image": gym.spaces.Box(
                    low=np.zeros((3, im_size, im_size)),
                    high=255 * np.ones((3, im_size, im_size)),
                    dtype=np.uint8,
                ),
                "full_image": gym.spaces.Box(
                    low=np.zeros((480, 640, 3)),
                    high=255 * np.ones((480, 640, 3)),
                    dtype=np.uint8,
                ),
                "joint_position": gym.spaces.Box(low=-2 * np.ones((6,)), high=2 * np.ones((6,)), dtype=np.float64),
                "cartesian_position": gym.spaces.Box(low=-2 * np.ones((6,)), high=2 * np.ones((6,)), dtype=np.float64),
                "gripper_position": gym.spaces.Box(low=np.zeros((1,)), high=np.ones((1,)), dtype=np.float64),
                "proprio": gym.spaces.Box(low=-2 * np.ones((7,)), high=2 * np.ones((7,)), dtype=np.float64),
            }
        )
        self.action_space = gym.spaces.Box(low=-2 * np.ones((7,)), high=2 * np.ones((7,)), dtype=np.float64)
        self.cfg = cfg

    def step(self, action, *, require_wrist=True):
        self.ur_client.step_action(action, blocking=self.blocking, action_space=self.cfg.action_space)

        raw_obs = self.ur_client.get_observation(require_wrist=require_wrist)
        truncated = False
        if raw_obs is None:
            # this indicates a loss of connection with the server
            # due to an exception in the last step so end the trajectory
            truncated = True
            obs = null_obs(self.im_size)  # obs with all zeros
        else:
            obs = convert_obs(raw_obs, self.im_size)

        return obs, 0, False, truncated, {}

    def stop(self):
        self.ur_client.stop()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ur_client.reset()
        self.move_to_start_state()

        raw_obs = wait_for_obs(self.ur_client)
        obs = convert_obs(raw_obs, self.im_size)

        return obs, {}

    def get_observation(self):
        raw_obs = wait_for_obs(self.ur_client)
        obs = convert_obs(raw_obs, self.im_size)
        return obs

    def move_to_start_state(self):
        successful = False
        while not successful:
            try:
                self.ur_client.move(np.concatenate([self.cfg.init_ee_pos, self.cfg.init_ee_rotvec]), blocking=True)
                successful = True
            except Exception as e:
                print(e)
