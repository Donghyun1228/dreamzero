# ruff: noqa

import contextlib
import dataclasses
import faulthandler
import os
import re
import signal
import time
import uuid
from dataclasses import field
from typing import Dict, List

import cv2
import numpy as np
import tqdm
import draccus

from eval_utils.policy_client import WebsocketClientPolicy

try:
    from realworld.snumpr_utils import get_ur_env
except ModuleNotFoundError:
    from snumpr_utils import get_ur_env

faulthandler.enable()

# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class GenerateConfig:
    # Rollout parameters
    max_timesteps: int = 1200
    open_loop_horizon: int = 24  # Match the 24-step action chunk used by the UR10e policy
    policy_frame_offsets: List[int] = field(default_factory=lambda: [-23, -16, -8, 0])
    policy_frame_fps: float = 30.0

    # Remote policy server
    remote_host: str = "127.0.0.1"
    remote_port: int = 5000

    # ---------------------------------------------------------------------------------------------
    # UR environment parameters
    # ---------------------------------------------------------------------------------------------
    host_ip: str = "147.47.190.120"
    ur_ip: str = "147.46.76.176"
    port: int = 5556

    init_ee_pos: List[float] = field(default_factory=lambda: [0.6918, -0.1736, 0.6794])
    init_ee_rotvec: List[float] = field(default_factory=lambda: [2.214, 2.221, 0.003])

    bounds: List[List[float]] = field(
        default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/blue/image_raw"}])

    blocking: bool = False
    control_frequency: float = 30
    action_space: str = "joint_position"

    # ---------------------------------------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------------------------------------
    save_data: bool = False


# -------------------------------------------------------------------------------------------------
# Ctrl+C protection during blocking policy calls
# -------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """
    Prevent Ctrl+C from interrupting a blocking policy server call.
    The interrupt is delayed and re-raised afterward.
    """
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


# -------------------------------------------------------------------------------------------------
# Main rollout loop
# -------------------------------------------------------------------------------------------------


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip(np.round(image), 0, 255).astype(np.uint8)
    return image


def _resize_for_server(image: np.ndarray, image_resolution) -> np.ndarray:
    image = _to_hwc_uint8(image)
    if image_resolution is None:
        return image

    target_h, target_w = image_resolution
    if image.shape[:2] == (target_h, target_w):
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _resize_video_for_server(video: np.ndarray, image_resolution) -> np.ndarray:
    video = np.asarray(video)
    if video.ndim == 3:
        return _resize_for_server(video, image_resolution)
    if video.ndim == 4:
        return np.stack([_resize_for_server(frame, image_resolution) for frame in video], axis=0)
    raise ValueError(f"Expected video with 3 or 4 dimensions, got shape {video.shape}")


def _extract_actions(inference_result) -> np.ndarray:
    actions = inference_result["actions"] if isinstance(inference_result, dict) else inference_result
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None]
    if actions.ndim != 2:
        raise ValueError(f"Expected a 2D action chunk, got shape {actions.shape}")
    return actions


def _build_request_data(obs, camera_frames: dict, instruction: str, session_id: str, server_metadata: dict) -> dict:
    image_resolution = server_metadata.get("image_resolution")
    needs_wrist_camera = bool(server_metadata.get("needs_wrist_camera", True))
    needs_session_id = bool(server_metadata.get("needs_session_id", False))

    request_data = {
        "video.cam_third_person": _resize_video_for_server(camera_frames["video.cam_third_person"], image_resolution),
        "state.joint_pos": np.asarray(obs["joint_position"], dtype=np.float64),
        "state.gripper_pos": np.asarray(obs["gripper_position"], dtype=np.float64),
        "annotation.task": instruction,
    }

    if needs_wrist_camera:
        request_data["video.cam_wrist"] = _resize_video_for_server(camera_frames["video.cam_wrist"], image_resolution)
    if needs_session_id:
        request_data["session_id"] = session_id

    return request_data


def _should_query_policy(pred_action_chunk, actions_from_chunk_completed: int, open_loop_horizon: int) -> bool:
    if pred_action_chunk is None:
        return True
    if actions_from_chunk_completed >= len(pred_action_chunk):
        return True
    return actions_from_chunk_completed >= open_loop_horizon


def _binarize_gripper(action: np.ndarray) -> np.ndarray:
    gripper_cmd = 1.0 if float(action[-1]) > 0.5 else 0.0
    return np.concatenate([action[:-1], np.array([gripper_cmd], dtype=np.float32)], axis=0)


@draccus.wrap()
def main(cfg: GenerateConfig):
    # Initialize environment
    env = get_ur_env(cfg)
    print("Created the SNUMPR environment")

    # Connect to policy server
    policy_client = WebsocketClientPolicy(cfg.remote_host, cfg.remote_port)
    server_metadata = policy_client.get_server_metadata()
    print(f"Connected to policy server with metadata: {server_metadata}")
    needs_wrist_camera = bool(server_metadata.get("needs_wrist_camera", True))

    server_action_space = server_metadata.get("action_space")
    if server_action_space is not None and server_action_space != "joint_position":
        raise ValueError(f"Expected a joint-position policy server, got action_space={server_action_space!r}")

    instruction = "do something"
    exp_name = input("Enter experiment name: ").strip()
    policy_offsets_seconds = [offset / cfg.policy_frame_fps for offset in cfg.policy_frame_offsets]

    while True:
        new_instruction = input("Enter instruction (leave empty to reuse): ").strip()
        if new_instruction:
            instruction = new_instruction

        # Reset environment
        env.reset()
        input(f"Instruction: {instruction}\nPress Enter to start rollout")

        obs, _ = env.reset()
        session_id = str(uuid.uuid4())
        require_wrist = needs_wrist_camera

        actions_from_chunk_completed = 0
        pred_action_chunk = None
        is_first_policy_query = True
        expected_action_dim = int(obs["joint_position"].shape[-1] + obs["gripper_position"].shape[-1])

        with prevent_keyboard_interrupt():
            policy_client.reset({"session_id": session_id})

        print("Running rollout... Press Ctrl+C to stop early.")
        bar = tqdm.tqdm(range(cfg.max_timesteps))

        for _ in bar:
            try:
                step_start = time.time()

                # Query policy if needed
                if _should_query_policy(pred_action_chunk, actions_from_chunk_completed, cfg.open_loop_horizon):
                    actions_from_chunk_completed = 0
                    camera_frames = env.ur_client.get_policy_camera_frames(
                        offsets_seconds=policy_offsets_seconds,
                        include_wrist=needs_wrist_camera,
                        first_only=is_first_policy_query,
                    )

                    request_data = _build_request_data(
                        obs=obs,
                        camera_frames=camera_frames,
                        instruction=instruction,
                        session_id=session_id,
                        server_metadata=server_metadata,
                    )

                    with prevent_keyboard_interrupt():
                        inference_result = policy_client.infer(request_data)

                    pred_action_chunk = _extract_actions(inference_result)

                    if pred_action_chunk.shape[-1] != expected_action_dim:
                        raise ValueError(
                            f"Expected actions with {expected_action_dim} dims "
                            f"({obs['joint_position'].shape[-1]} joints + 1 gripper), "
                            f"got shape {pred_action_chunk.shape}"
                        )
                    is_first_policy_query = False

                # Select action from the predicted chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                action = _binarize_gripper(action)

                obs, _, _, _, _ = env.step(action, require_wrist=require_wrist)
                # Enforce control frequency
                elapsed = time.time() - step_start
                sleep_time = max(0.0, 1.0 / cfg.control_frequency - elapsed)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\nRollout interrupted early by user.")
                break

        # -----------------------------------------------------------------------------------------
        # Save rollout video
        # -----------------------------------------------------------------------------------------
        cleaned_instruction = re.sub(r"\s+", "_", re.sub(r"\bthe\b", "", instruction, flags=re.IGNORECASE)).strip("_")

        save_dir = os.path.join("./rollouts", exp_name, cleaned_instruction)
        os.makedirs(save_dir, exist_ok=True)

        rollout_name = input("Enter rollout name (empty to skip saving): ").strip()
        if not rollout_name:
            env.ur_client.end_rollout(save_path=None)
            policy_client.reset({"session_id": session_id})
            print("Skipping rollout saving.")
            continue

        timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
        mp4_path = os.path.join(save_dir, f"{rollout_name}-{timestamp}.mp4")

        env.ur_client.end_rollout(save_path=mp4_path)
        print(f"Saved rollout video to: {mp4_path}")
        policy_client.reset({"session_id": session_id})

        # Reset gripper after rollout
        env.ur_client.gripper.move(
            position=env.ur_client.gripper.get_open_position(),
            speed=64,
            force=1,
        )
        time.sleep(0.1)


if __name__ == "__main__":
    main()
