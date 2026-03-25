"""Utils for evaluating policies in real-world BridgeData V2 environments."""
# ruff: noqa

import atexit
import os
import sys
import time

import cv2
import imageio
import numpy as np
import torch

sys.path.append(".")
try:
    from realworld.snumpr_env import URGym
    from realworld.ur_controller.ur import URClient
except ModuleNotFoundError:
    from snumpr_env import URGym
    from ur_controller.ur import URClient

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
UR_PROPRIO_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: f"{x:0.2f}"})


def get_ur_env_params(cfg):
    """Gets (mostly default) environment parameters for the UR environment."""
    env_params = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [
            [0.1, -0.15, -0.1, -1.57, 0],
            [0.45, 0.25, 0.18, 1.57, 0],
        ],
        "action_clipping": "xyz",
        "catch_environment_except": False,
        "start_state": [0.6918, -0.1736, 0.6794, 2.214, 2.221, 0.003],  # pose when reset is called
        "skip_move_to_neutral": False,
        "return_full_image": False,
        "camera_topics": [{"name": "/blue/image_raw"}],
    }
    env_params["override_workspace_boundaries"] = cfg.bounds
    env_params["camera_topics"] = cfg.camera_topics
    env_params["return_full_image"] = True
    return env_params


def get_ur_env(cfg, model=None):
    """Get UR control environment."""
    # Set up the UR environment parameters
    env_params = get_ur_env_params(cfg)
    start_state = np.concatenate([cfg.init_ee_pos, cfg.init_ee_rotvec])
    env_params["start_state"] = list(start_state)

    # Set up the UR client
    ur_client = URClient(host=cfg.host_ip, ur_ip=cfg.ur_ip, port=cfg.port)
    print(f"[System] Registering safe exit for URClient connected to {cfg.ur_ip}")
    atexit.register(ur_client.stop)

    ur_client.init(env_params)
    env = URGym(
        ur_client,
        cfg=cfg,
        blocking=cfg.blocking,
    )
    return env


def get_next_task_label(task_label):
    """Prompt the user to input the next task."""
    if task_label == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the task name: ")
        task_label = user_input
    else:
        user_input = input("Enter the task name (or leave blank to repeat the previous task): ")
        if user_input == "":
            pass  # Do nothing -> Let task_label be the same
        else:
            task_label = user_input
    print(f"Task: {task_label}")
    return task_label


def save_rollout_video(rollout_images, fps, save_dir=""):
    """Saves an MP4 replay of an episode."""
    if save_dir == "":
        save_dir = "./rollouts"
    else:
        save_dir = os.path.join("./rollouts", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    rollout_name = input("Enter the name of current rollout: ")
    if rollout_name == "":
        print("Skip rollout saving")
        return
    mp4_path = f"{save_dir}/{rollout_name}-{DATE_TIME}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")


def save_rollout_data(rollout_orig_images, rollout_images, rollout_states, rollout_actions, idx):
    """
    Saves rollout data from an episode.

    Args:
        rollout_orig_images (list): Original rollout images (before preprocessing).
        rollout_images (list): Preprocessed images.
        rollout_states (list): Proprioceptive states.
        rollout_actions (list): Predicted actions.
        idx (int): Episode index.
    """
    os.makedirs("./rollouts", exist_ok=True)
    path = f"./rollouts/rollout-{DATE_TIME}-{idx+1}.npz"
    # Convert lists to numpy arrays
    orig_images_array = np.array(rollout_orig_images)
    images_array = np.array(rollout_images)
    states_array = np.array(rollout_states)
    actions_array = np.array(rollout_actions)
    # Save to a single .npz file
    np.savez(path, orig_images=orig_images_array, images=images_array, states=states_array, actions=actions_array)
    print(f"Saved rollout data at path {path}")


def resize_image(img: np.ndarray, resize_size: tuple[int, int]) -> np.ndarray:
    """
    Resize uint8 HxWxC image to resize_size using Lanczos interpolation.
    """
    assert isinstance(resize_size, tuple) and len(resize_size) == 2
    h, w = resize_size

    if img.dtype != np.uint8:
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)

    # cv2.resize: dsize = (width, height)
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    resized = np.clip(np.round(resized), 0, 255).astype(np.uint8)
    return resized


def get_preprocessed_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    w = 250
    cropped_img = obs["full_image"][:, w : w + 720 :, :]
    cropped_img = resize_image(cropped_img, resize_size)
    return cropped_img
