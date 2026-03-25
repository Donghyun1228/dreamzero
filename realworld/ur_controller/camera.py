# ruff: noqa

import base64
from collections import deque
import json
import os
import pickle
import subprocess
import threading
import time
import queue

import cv2
import imageio
import numpy as np
import zmq


class ZMQCameraSubscriber:
    def __init__(self, host, port, save_video_path=None, fps=30, ur_client=None, history_size=256):
        self._host, self._port = host, port
        self.save_video_path = save_video_path
        self.fps = fps

        self._init_subscriber()

        self.latest_image = None
        self.latest_timestamp = None
        self._lock = threading.Lock()
        self._running = True
        self._recent_frames = deque(maxlen=history_size)

        # 메모리 버퍼 (녹화 중 여기에만 저장)
        self._frame_buffer = []
        self._frame_times_buffer = []

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        self._recording = False
        self.ur_client = ur_client

        # 백그라운드 저장 상태 관리
        self._is_saving = False
        self._save_thread = None

        # [복구] 외부에서 저장이 끝났는지 기다릴 수 있도록 이벤트 객체 복구
        self._recording_done_event = threading.Event()
        self._recording_done_event.set()  # 초기 상태는 '작업 완료' 상태로 둠

    def _init_subscriber(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        print(f"tcp://{self._host}:{self._port}")
        self.socket.connect(f"tcp://{self._host}:{self._port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"rgb_image")

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        socks = poller.poll(500)

        if socks:
            print("[OK] Received a message from publisher!")
            return True
        print("[WARN] No message received (publisher not sending or wrong address?)")
        return False

    def stop(self):
        print(f"Closing the subscriber socket in {self._host}:{self._port}.")
        self._running = False
        self.socket.close()
        self.context.term()

    def get_latest_frame(self):
        with self._lock:
            if self.latest_image is None:
                return None, None
            return self.latest_timestamp, self.latest_image.copy()

    def clear_recent_history(self, keep_latest=True):
        with self._lock:
            latest = None
            if keep_latest and self.latest_image is not None and self.latest_timestamp is not None:
                latest = (self.latest_timestamp, self.latest_image)
            self._recent_frames.clear()
            if latest is not None:
                self._recent_frames.append(latest)

    def sample_frames_nearest(self, target_timestamps):
        with self._lock:
            history = list(self._recent_frames)

        if not history:
            raise RuntimeError(f"No camera frames available for {self._host}:{self._port}")

        times = np.asarray([ts for ts, _ in history], dtype=np.float64)
        frames = [frame for _, frame in history]

        sampled = []
        for target_ts in target_timestamps:
            idx = int(np.argmin(np.abs(times - target_ts)))
            sampled.append(frames[idx].copy())
        return sampled

    def _recv_loop(self):
        while self._running:
            try:
                if self.socket.poll(100) == 0:
                    continue

                raw = self.socket.recv()
                data = pickle.loads(raw.lstrip(b"rgb_image "))

                img = cv2.imdecode(
                    np.frombuffer(base64.b64decode(data["rgb_image"]), np.uint8),
                    cv2.IMREAD_COLOR,
                )
                recv_time = time.time()

                with self._lock:
                    self.latest_image = img
                    self.latest_timestamp = recv_time
                    self._recent_frames.append((recv_time, img))

                # 녹화 중일 때: 리스트에 원본 그대로 추가 (빠름)
                if self._recording:
                    self._frame_buffer.append(img)
                    self._frame_times_buffer.append(recv_time)

            except Exception:
                time.sleep(0.01)

    def start_recording(self):
        if self._is_saving:
            print("[WARN] Previous video is still saving! Waiting is recommended.")

        self._frame_buffer = []
        self._frame_times_buffer = []

        # [복구] 녹화 시작 시 '완료 안 됨' 상태로 변경
        self._recording_done_event.clear()

        self._recording = True
        print("[Recorder] Recording started (Memory Buffer Mode)")

    def stop_recording(self, final_path=None):
        """
        녹화를 중단하고 백그라운드 스레드에서 저장을 시작합니다.
        메인 스레드는 즉시 반환됩니다.
        """
        self._recording = False

        # 데이터 이동 (참조 복사 후 원본 초기화)
        frames = self._frame_buffer
        times = self._frame_times_buffer

        self._frame_buffer = []
        self._frame_times_buffer = []

        # 백그라운드 저장 시작
        self._save_thread = threading.Thread(
            target=self._async_save_task, args=(frames, times, final_path), daemon=True
        )
        self._save_thread.start()
        print(f"[Recorder] Stop requested. Background saving started for {len(frames)} frames...")

    def _async_save_task(self, frames, times, final_path):
        self._is_saving = True
        try:
            if not frames:
                print("[WARN] No frames to save.")
                return
            if not final_path:
                print("[Recorder] Recording stopped without saving.")
                return

            # FPS 계산
            if len(times) > 1:
                real_fps = 1.0 / np.mean(np.diff(times))
            else:
                real_fps = self.fps

            print(f"[BG-Save] Saving... Real FPS: {real_fps:.2f}")

            os.makedirs(os.path.dirname(final_path), exist_ok=True)

            # 1. 원본 저장 (고압축 적용 CRF 30)
            writer = imageio.get_writer(
                final_path,
                fps=real_fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=None,
                ffmpeg_params=[
                    "-preset",
                    "superfast",  # 저장 속도 빠름
                    "-crf",
                    "30",  # [핵심] 압축률 높임 (기본 23 -> 30)
                ],
            )

            for img in frames:
                w = 250
                # Crop logic
                if img.shape[1] >= w + 720:
                    crop = img[:, w : w + 720, :]
                else:
                    crop = img

                # BGR -> RGB
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                writer.append_data(crop)

            writer.close()
            print(f"[BG-Save] Original saved: {final_path}")

            # 2. 2배속 저장 (고압축 적용)
            dirpath = os.path.dirname(final_path)
            basename = os.path.basename(final_path)
            speedup_path = os.path.join(dirpath, "speedup", basename)
            os.makedirs(os.path.dirname(speedup_path), exist_ok=True)

            reencode_with_speedup(
                input_path=final_path, output_path=speedup_path, fps=real_fps, writer_fps=real_fps, speedup=2.0
            )
            print(f"[BG-Save] 2x Speedup saved: {speedup_path}")

        except Exception as e:
            print(f"[BG-Save Error] {e}")
        finally:
            self._is_saving = False
            # [복구] 저장 작업이 모두 끝나면 이벤트 set (대기하던 외부 코드 실행 재개)
            self._recording_done_event.set()


class ZMQCameraOnDemandSubscriber:
    def __init__(self, host, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"rgb_image")

    def recv(self, timeout_ms=100):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        socks = poller.poll(timeout_ms)
        if not socks:
            return None

        raw = self.socket.recv()
        data = pickle.loads(raw.lstrip(b"rgb_image "))
        encoded = np.frombuffer(base64.b64decode(data["rgb_image"]), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


def reencode_with_speedup(
    input_path,
    output_path,
    fps,
    writer_fps=30,
    speedup=2.0,
    speed_text="x2",
):
    if fps <= 0:
        fps = 30
    scale = (writer_fps / fps) * (1 / speedup)

    drawtext = (
        "drawtext=text='%s':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.5:boxborderw=6:x=10:y=10"
    ) % speed_text

    vf = f"setpts={scale}*PTS,{drawtext}"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "quiet",
        "-i",
        input_path,
        "-vf",
        vf,
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "superfast",  # 속도 빠르게
        "-crf",
        "30",  # [핵심] 고압축 설정
        output_path,
    ]
    subprocess.run(cmd, check=True)


def reencode_original(
    input_path,
    output_path,
    fps,
    writer_fps=30,
):
    if fps <= 0:
        fps = 30
    scale = writer_fps / fps

    vf = f"setpts={scale}*PTS"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "quiet",
        "-i",
        input_path,
        "-vf",
        vf,
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "superfast",
        "-crf",
        "30",  # [핵심] 고압축 설정
        output_path,
    ]
    subprocess.run(cmd, check=True)
