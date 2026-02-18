# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
from dataclasses import dataclass

import imageio
import numpy as np

from cosmos_transfer2._src.imaginaire.utils import log


@dataclass
class VideoData:
    frames: np.ndarray  # Shape: [B, H, W, C]
    fps: int
    duration: int  # in seconds


def get_video_filepaths(input_dir: str) -> list[str]:
    """Get a list of filepaths for all videos in the input directory."""
    paths = glob.glob(f"{input_dir}/**/*.mp4", recursive=True)
    paths += glob.glob(f"{input_dir}/**/*.avi", recursive=True)
    paths += glob.glob(f"{input_dir}/**/*.mov", recursive=True)
    paths = sorted(paths)
    log.debug(f"Found {len(paths)} videos")
    return paths


def read_video(filepath: str) -> VideoData:
    """Read a video file and extract its frames and metadata."""
    try:
        reader = imageio.get_reader(filepath, "ffmpeg")
    except Exception as e:
        raise ValueError(f"Failed to read video file: {filepath}") from e

    # Extract metadata from the video file
    try:
        metadata = reader.get_meta_data()
        fps = metadata.get("fps")
        duration = metadata.get("duration")
    except Exception as e:
        reader.close()
        raise ValueError(f"Failed to extract metadata from video file: {filepath}") from e

    # Extract frames from the video file
    try:
        frames = np.array([frame for frame in reader])
    except Exception as e:
        raise ValueError(f"Failed to extract frames from video file: {filepath}") from e
    finally:
        reader.close()

    return VideoData(frames=frames, fps=fps, duration=duration)


def save_video(filepath: str, frames: np.ndarray, fps: int) -> None:
    """Save a video file from a sequence of frames using VP9 codec for NIMS compatibility."""
    try:
        writer = imageio.get_writer(
            filepath,
            fps=fps,
            codec='libvpx-vp9',
            quality=None,
            bitrate=0,
            output_params=[
                '-f', 'mp4',
                '-crf', '30',  # Quality setting (0-63, lower is better quality)
                '-deadline', 'realtime',  # Optimize for real-time encoding
                '-cpu-used', '4',  # Speed/quality tradeoff (0-8, higher is faster)
                '-row-mt', '1',  # Enable row-based multi-threading
            ]
        )
        for frame in frames:
            # Handle single channel frames by repeating across 3 channels
            if len(frame.shape) == 2:
                frame = frame[:, :, None].repeat(3, axis=2)
            writer.append_data(frame)
    except Exception as e:
        raise ValueError(f"Failed to save video file to {filepath}") from e
    finally:
        writer.close()
