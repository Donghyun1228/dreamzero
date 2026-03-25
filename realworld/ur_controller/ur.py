"""
URClient definition.
"""
# ruff: noqa

import threading
import time

import cv2
import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R

try:
    import realworld.ur_controller.robotiq as robotiq
    from realworld.ur_controller.camera import ZMQCameraSubscriber
except ModuleNotFoundError:
    import ur_controller.robotiq as robotiq
    from ur_controller.camera import ZMQCameraSubscriber


class URClient:
    def __init__(self, host: str = "localhost", ur_ip: str = "localhost", port: int = 5556):
        """
        Args:
            host: camera publisher host ip
            ur_ip: UR controller ip
            port: (kept for interface compatibility, not used directly)
        """
        self.rtde_c = rtde_control.RTDEControlInterface(ur_ip, 100)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ur_ip, 30)

        self.gripper = robotiq.RobotiqGripper()
        self.gripper.connect(ur_ip, 63352)
        self.gripper.activate()

        self._pose_lock = threading.Lock()
        self.target_pose = None
        self.target_joint = None
        self._control_running = True

        # Flag to distinguish idle vs active servo control
        self._control_active = False
        self._control_mode = "cartesian_position"

        # Servo parameters
        self.servo_accel = 0.0
        self.servo_speed = 0.0
        self.servo_lookahead = 0.03
        self.servo_gain = 300
        self.max_joint_command_delta = 0.2
        self._last_gripper_command_is_open = None

        # Initialize target to actual pose once before starting control loop
        self._sync_target_to_actual()

        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

        self.image_subscriber = ZMQCameraSubscriber(
            host=host,
            port=10005,
            ur_client=self,
            # topic_type="RGB",
        )
        self.wrist_image_subscriber = ZMQCameraSubscriber(
            host=host,
            port=10006,
            # topic_type="RGB",
        )

        print(f"URClient connected to {host}:{port}")

    def _flush_receive_buffer(self):
        print("Flushing receive buffer...")
        for _ in range(1000):
            self.rtde_r.getActualTCPPose()
        print("Receive buffer flushed.")

    def _sync_target_to_actual(self, flush=True):
        """Read current TCP pose and set it as the servo target."""
        if flush:
            self._flush_receive_buffer()

        actual_pose = self.rtde_r.getActualTCPPose()
        with self._pose_lock:
            self.target_pose = list(actual_pose)

        return actual_pose

    def _sync_joint_target_to_actual(self):
        actual_joint = self.rtde_r.getActualQ()
        with self._pose_lock:
            self.target_joint = list(actual_joint)
        return actual_joint

    def _get_gripper_position(self):
        try:
            gripper_position = float(self.gripper.get_current_position())
        except Exception:
            gripper_position = float(self.gripper.get_open_position())

        open_position = float(self.gripper.get_open_position())
        closed_position = float(self.gripper.get_closed_position())
        denom = max(closed_position - open_position, 1.0)
        normalized = (gripper_position - open_position) / denom
        return np.array([np.clip(normalized, 0.0, 1.0)], dtype=np.float64)

    def _command_gripper(self, gripper_value: float):
        desired_open = float(gripper_value) > 0.5
        if self._last_gripper_command_is_open == desired_open:
            return

        gripper_command = (
            self.gripper.get_open_position() if desired_open else self.gripper.get_closed_position()
        )
        self.gripper.move(position=gripper_command, speed=64, force=1)
        self._last_gripper_command_is_open = desired_open

    def _control_loop(self, frequency=100.0):
        """
        High-frequency control loop to send servo commands to the robot.
        """
        dt = 1.0 / frequency

        print(f"[Control Loop] Started at {frequency}Hz")

        while self._control_running:
            # If not active, do nothing this cycle
            if not self._control_active:
                time.sleep(dt)
                continue

            start_time = time.time()
            with self._pose_lock:
                control_mode = self._control_mode
                cmd_pose = None if self.target_pose is None else list(self.target_pose)
                cmd_joint = None if self.target_joint is None else list(self.target_joint)

            try:
                if control_mode == "joint_position":
                    current_joint = np.array(self.rtde_r.getActualQ(), dtype=np.float32)
                    if cmd_joint is None:
                        cmd_joint = current_joint.tolist()
                    cmd_joint_np = np.asarray(cmd_joint, dtype=np.float32)
                    clipped_joint = current_joint + np.clip(
                        cmd_joint_np - current_joint,
                        -self.max_joint_command_delta,
                        self.max_joint_command_delta,
                    )
                    self.rtde_c.servoJ(
                        clipped_joint.tolist(),
                        self.servo_speed,
                        self.servo_accel,
                        dt,
                        self.servo_lookahead,
                        self.servo_gain,
                    )
                else:
                    current_pose = self.rtde_r.getActualTCPPose()
                    if cmd_pose is None:
                        cmd_pose = list(current_pose)

                    current_np = np.array(current_pose[:3], dtype=np.float32)
                    cmd_np = np.array(cmd_pose[:3], dtype=np.float32)
                    diff = np.linalg.norm(current_np - cmd_np)

                    # Safety: if target drifts too far from actual, resync to actual pose
                    if diff > 0.15:
                        print(f"[SafeGuard] Drift detected ({diff:.3f}m). Re-syncing.")
                        current_pose = self._sync_target_to_actual(flush=False)
                        cmd_pose = list(current_pose)

                    self.rtde_c.servoL(
                        cmd_pose,
                        self.servo_speed,
                        self.servo_accel,
                        dt,
                        self.servo_lookahead,
                        self.servo_gain,
                    )
            except Exception as e:
                print("[servo loop] error:", e)

            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def init(self, env_params: dict, image_size: int = 224):
        """
        Initialize the environment.
            env_params: dict of env params
            image_size: the size of the cropped image to return
        """
        self.env_params = env_params
        self.image_size = image_size

    def _crop_exterior_bgr(self, full_image):
        w = 250
        if full_image.shape[1] >= w + 720:
            return full_image[:, w : w + 720, :]
        return full_image

    def _crop_wrist_bgr(self, full_image):
        if full_image.shape[1] >= 480:
            return full_image[:, :480, :]
        return full_image

    def _to_rgb(self, image_bgr):
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _format_observation_image(self, full_image, camera_type: str):
        if camera_type == "wrist":
            cropped = self._crop_wrist_bgr(full_image)
        else:
            cropped = self._crop_exterior_bgr(full_image)
        resized = cv2.resize(cropped, (self.image_size, self.image_size))
        return self._to_rgb(resized)

    def _format_policy_image(self, full_image, camera_type: str):
        if camera_type == "wrist":
            cropped = self._crop_wrist_bgr(full_image)
        else:
            cropped = self._crop_exterior_bgr(full_image)
        return self._to_rgb(cropped)

    def _get_policy_anchor_timestamp(self, include_wrist=True):
        timestamps = []
        for subscriber in (self.image_subscriber, self.wrist_image_subscriber if include_wrist else None):
            if subscriber is None:
                continue
            timestamp, _ = subscriber.get_latest_frame()
            if timestamp is not None:
                timestamps.append(float(timestamp))

        if not timestamps:
            raise RuntimeError("No camera frames available for policy query")

        return min(timestamps)

    def get_policy_camera_frames(self, offsets_seconds, include_wrist=True, first_only=False):
        anchor_timestamp = self._get_policy_anchor_timestamp(include_wrist=include_wrist)
        if first_only:
            target_timestamps = [anchor_timestamp]
        else:
            target_timestamps = [anchor_timestamp + float(offset) for offset in offsets_seconds]

        third_person_raw = self.image_subscriber.sample_frames_nearest(target_timestamps)
        third_person = [self._format_policy_image(frame, camera_type="third_person") for frame in third_person_raw]

        if include_wrist:
            try:
                wrist_raw = self.wrist_image_subscriber.sample_frames_nearest(target_timestamps)
                wrist = [self._format_policy_image(frame, camera_type="wrist") for frame in wrist_raw]
            except RuntimeError:
                wrist = [frame.copy() for frame in third_person]
        else:
            wrist = [frame.copy() for frame in third_person]

        if first_only:
            return {
                "video.cam_third_person": third_person[-1],
                "video.cam_wrist": wrist[-1],
            }

        return {
            "video.cam_third_person": np.stack(third_person, axis=0),
            "video.cam_wrist": np.stack(wrist, axis=0),
        }

    def get_observation(self, *, require_wrist=True):
        _, full_image = self.image_subscriber.get_latest_frame()

        if full_image is None:
            print("Null Obs!!!")
            return None
        image = self._format_observation_image(full_image, camera_type="third_person")

        if require_wrist:
            _, wrist_full = self.wrist_image_subscriber.get_latest_frame()
            if wrist_full is None:
                wrist_image = image  # fallback
            else:
                wrist_image = self._format_observation_image(wrist_full, camera_type="wrist")
        else:
            wrist_image = image

        full_image = self._to_rgb(full_image)
        joint_position = np.array(self.rtde_r.getActualQ(), dtype=np.float64)
        cartesian_position = np.array(self.rtde_r.getActualTCPPose(), dtype=np.float64)
        gripper_position = self._get_gripper_position()
        proprio = np.concatenate([joint_position, gripper_position], axis=0)
        return {
            "image": image,
            "wrist_image": wrist_image,
            "full_image": full_image,
            "joint_position": joint_position,
            "cartesian_position": cartesian_position,
            "gripper_position": gripper_position,
            "proprio": proprio,
        }

    def stop(self):
        print("[URClient] Stopping ...")
        self._control_running = False
        self.image_subscriber.stop()
        self.wrist_image_subscriber.stop()
        self.image_subscriber._thread.join(timeout=1.0)
        self.wrist_image_subscriber._thread.join(timeout=1.0)
        try:
            self.rtde_c.servoStop()
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
        except Exception as e:
            print(f"[URClient] Cleanup error: {e}")
        print("[URClient] Disconnected.")

    def reset(self):
        # Put the control loop into idle state
        self._flush_receive_buffer()
        self._control_active = False
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass
        with self._pose_lock:
            self.target_pose = None
            self.target_joint = None
            self._control_mode = "cartesian_position"
        self._last_gripper_command_is_open = None
        self.image_subscriber.clear_recent_history(keep_latest=True)
        self.wrist_image_subscriber.clear_recent_history(keep_latest=True)

        self.image_subscriber.start_recording()

    def end_rollout(self, save_path):
        self.image_subscriber.stop_recording(save_path)

        print("[URClient] Waiting for recording to finish...")
        self.image_subscriber._recording_done_event.wait()
        print("[URClient] Recording finished.")

    def _rotvec2rpy(self, rotvec):
        """Convert rotation vector to roll-pitch-yaw."""
        return R.from_rotvec(rotvec).as_euler("xyz")

    def _rpy2rotvec(self, rpy):
        """Convert roll-pitch-yaw to rotation vector."""
        return R.from_euler("xyz", rpy).as_rotvec()

    def step_action(
        self,
        action,
        speed=0.5,  # m/s (unused here, kept for interface compatibility)
        acceleration=0.5,  # m/s^2 (unused here)
        blocking=True,
        action_space="cartesian_delta_pose",
    ):
        action = np.asarray(action, dtype=np.float32)
        self._command_gripper(action[-1])

        if action_space == "joint_position":
            target_joint = action[:-1]
            current_joint = np.asarray(self.rtde_r.getActualQ(), dtype=np.float32)
            if target_joint.shape != current_joint.shape:
                raise ValueError(
                    f"Joint-position action shape mismatch: expected {current_joint.shape}, got {target_joint.shape}"
                )

            if not self._control_active or self._control_mode != "joint_position":
                print("[URClient] Waking up joint servo control.")
                try:
                    self.rtde_c.servoStop()
                except Exception:
                    pass
                self._sync_joint_target_to_actual()
                self._control_active = True

            with self._pose_lock:
                self.target_joint = target_joint.tolist()
                self._control_mode = "joint_position"
            return

        zero_action = np.asarray(
            [
                0.0001729056,
                0.0003461923,
                0.0010509108,
                0.0009982494,
                0.0004280202,
                0.0007946129,
            ],
            dtype=np.float32,
        )

        # If coming from idle, sync target to current pose and enable control
        if not self._control_active or self._control_mode != "cartesian_position":
            print("[URClient] Waking up cartesian servo control.")
            try:
                self.rtde_c.servoStop()
            except Exception:
                pass
            self._sync_target_to_actual()
            self._control_active = True

        current_pose = np.array(self.rtde_r.getActualTCPPose(), dtype=np.float32)  # [x,y,z,rx,ry,rz]
        current_rpy = self._rotvec2rpy(current_pose[3:6])

        # Clip deltas to avoid overly large jumps per step
        delta_xyz = np.clip(action[0:3], -0.1, 0.1)
        delta_rpy = np.clip(action[3:6], -0.1, 0.1)

        new_rpy = current_rpy + delta_rpy
        new_rotvec = self._rpy2rotvec(new_rpy)

        new_pose = current_pose.copy()
        new_pose[0:3] += delta_xyz
        new_pose[3:6] = new_rotvec

        with self._pose_lock:
            self.target_pose = new_pose.tolist()
            self._control_mode = "cartesian_position"

        if np.all(np.abs(action[:-1] - zero_action) < 1e-4):
            return

    def move(self, pose, speed=0.1, acceleration=0.1, blocking=True):
        pose = np.asarray(pose, dtype=np.float32)

        # Pause the control loop while doing a blocking moveL
        self._control_active = False

        try:
            self.rtde_c.servoStop()
        except Exception:
            pass

        try:
            self.rtde_c.moveL(pose.tolist(), speed, acceleration, asynchronous=(not blocking))
        except Exception as e:
            print("[move] moveL error:", e)

        try:
            self.gripper.move_and_wait_for_pos(
                position=self.gripper.get_open_position(),
                speed=64,
                force=1,
            )
            self._last_gripper_command_is_open = True
        except Exception as e:
            print("[move] gripper error:", e)

        # After move, sync target to actual so the servo loop continues smoothly
        self._sync_target_to_actual()
        print("Target pose: " + ", ".join(f"{v:.4f}" for v in self.target_pose))

        if np.linalg.norm(np.array(self.rtde_r.getActualTCPPose()) - pose) > 1e-2:
            print("[URClient] Warning: move command did not reach target pose accurately.")
            input("PRESS ENTER TO CONTINUE...")
            input("DO NOT FORGET TO CHECK THE ROBOT SAFETY!")
