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

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union

import torch

from cosmos_transfer2._src.imaginaire.flags import INTERNAL
from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import get_augmentor_for_eval
from cosmos_transfer2._src.imaginaire.utils.distributed import get_rank, get_world_size, broadcast, barrier
from cosmos_transfer2._src.transfer2.inference.utils import (
    get_t5_from_prompt,
    normalized_float_to_uint8,
    read_and_process_control_input,
    read_and_process_image_context,
    read_and_process_video,
    read_and_resize_input,
    reshape_output_video_to_input_resolution,
    uint8_to_normalized_float,
    detect_aspect_ratio,
    INTER_LINEAR,
    INTER_AREA,
    INTER_NEAREST,
)


class ControlVideo2WorldInference:
    """
    Handles the Control2Video inference process, including model loading, data preparation,
    and video transfer from an input video and text prompt.
    """

    def __init__(
        self,
        registered_exp_name: str,
        checkpoint_paths: Union[str, list[str]],
        s3_credential_path: str,
        exp_override_opts: Optional[list[str]] = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        cache_dir: Optional[str] = None,
        skip_load_model: bool = False,
        base_load_from: Optional[str] = None,
    ):
        """
        Initializes the ControlVideo2WorldInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path.

        Args:
            registered_exp_name (str): Name of the experiment configuration.
            checkpoint_paths (Union[str, list[str]]): Single checkpoint path or List of checkpoint paths for multi-branch models.
            s3_credential_path (str): Path to S3 credentials file for ckpt & negative embedding (if loading from S3).
            exp_override_opts (list[str]): List of experiment override options.
            process_group (torch.distributed.ProcessGroup): Process group for distributed training.
            cache_dir (str): Cache directory for storing pre-computed embeddings.
            skip_load_model (bool): Whether to skip loading model from checkpoint for multi-control models.
        """
        self.registered_exp_name = registered_exp_name
        self.checkpoint_path = checkpoint_paths if isinstance(checkpoint_paths, str) else checkpoint_paths[0]
        self.s3_credential_path = s3_credential_path
        self.cache_dir = cache_dir

        if exp_override_opts is None:
            exp_override_opts = []
        # no need to load base model separately at inference
        exp_override_opts.append("model.config.base_load_from=null")
        if not INTERNAL:
            exp_override_opts.append("~data_train")
        # Load the model and config. Each trained model's config is composed by
        # loading a pre-registered experiment config, and then (optionally) overriding with some command-line
        # arguments. That is done in experiment_list.py. Here we simply replicate that process.
        model, config = load_model_from_checkpoint(
            experiment_name=self.registered_exp_name,
            s3_checkpoint_dir=self.checkpoint_path,
            config_file="cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py",
            load_ema_to_reg=True,
            local_cache_dir=(
                cache_dir if not checkpoint_paths else None
            ),  # for multi-control models, need to load other branches before caching
            experiment_opts=exp_override_opts,
        )
        if (
            isinstance(checkpoint_paths, list) and len(checkpoint_paths) > 1 and not skip_load_model
        ):  # load other branches for multi-control models
            load_from_local = False
            if cache_dir is not None:
                # build a unique path for s3checkpoint dir
                local_s3_ckpt_fp = os.path.join(
                    cache_dir,
                    self.checkpoint_path.split("s3://")[1],
                    "torch_model",
                    f"_rank_{distributed.get_rank()}.pt",
                )
                if os.path.exists(local_s3_ckpt_fp):
                    load_from_local = True

            if load_from_local:
                log.info(f"Loading model cached locally from {local_s3_ckpt_fp}")
                model.load_state_dict(easy_io.load(local_s3_ckpt_fp))
            else:
                model.load_multi_branch_checkpoints(checkpoint_paths=checkpoint_paths)
                if cache_dir is not None:
                    log.info(f"Caching model state dict to {local_s3_ckpt_fp}")
                    easy_io.dump(model.state_dict(), local_s3_ckpt_fp)

        if base_load_from is not None:
            log.info(f"Loading base model from {base_load_from}")
            model.config.base_load_from = {
                "load_path": base_load_from,
                "credentials": s3_credential_path,
            }
            model.load_base_model()

        self.text_encoder_class = model.text_encoder_class

        if process_group is not None:
            log.info("Enabling CP in base model\n")
            model.net.enable_context_parallel(process_group)

        self.model = model
        self.config = config
        self.batch_size = 1

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prev_output: torch.Tensor,
        text_embedding: torch.Tensor,
        fps: int,
        negative_prompt: str = None,
        control_weight: str = "1.0",
        image_context: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.

            image_context (torch.Tensor, optional): Image context tensor for conditioning. Can be (B, C, H, W).

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = prev_output.shape
        input_key = "video" if T > 1 else "images"

        data_batch = {
            "dataset_name": "video_data",
            input_key: prev_output.squeeze(2),
            "t5_text_embeddings": text_embedding,  # positive prompt embedding. Name has t5 but also supports Reason1.
            "fps": torch.randint(16, 32, (self.batch_size,)).cuda(),  # Random FPS (might be used by model)
            "padding_mask": torch.zeros(self.batch_size, 1, H, W).cuda(),  # Padding mask (assumed no padding here)
            "num_conditional_frames": 1,  # Specify that the first frame is conditional
            "control_weight": [float(w) for w in control_weight.split(",")],
            "input_video": video,
        }

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        # Add image context
        if image_context is not None:
            data_batch["image_context"] = image_context.cuda().to(dtype=torch.bfloat16).contiguous()

        # Handle negative prompts for classifier-free guidance
        if negative_prompt is not None:
            assert self.neg_t5_embeddings is not None, "Negative prompt embedding is not computed."
            data_batch["neg_t5_text_embeddings"] = self.neg_t5_embeddings

        return data_batch

    def _get_num_chunks(
        self, input_frames: torch.Tensor, num_video_frames_per_chunk: int, num_conditional_frames: int
    ) -> tuple[int, int, int]:
        """
        Get the number of chunks for chunk-wise long video generation.
        """
        # Frame number settting for chunk-wise long video generation
        num_total_frames = input_frames.shape[1]
        num_frames_per_chunk = num_video_frames_per_chunk - num_conditional_frames
        if num_video_frames_per_chunk == 1:
            num_chunks = 1
        else:
            num_generated_frames_vid2vid = num_total_frames - num_video_frames_per_chunk
            num_chunks = 1 + num_generated_frames_vid2vid // num_frames_per_chunk
            if num_generated_frames_vid2vid % num_frames_per_chunk != 0:
                num_chunks += 1

        return num_total_frames, num_chunks, num_frames_per_chunk

    def _pad_input_frames(
        self,
        input_frames: torch.Tensor,
        num_total_frames: int,
        num_video_frames_per_chunk: int,
        padding_mode: str = "reflect",
    ) -> torch.Tensor:
        """
        Pad input frames if total frames is less than chunk size
        """
        if num_total_frames < num_video_frames_per_chunk:
            # Check whether the input_frames is empty. If so, there is nothing to pad.
            if num_total_frames == 0:
                raise ValueError("No input frames; cannot pad. Verify that video frame counts match.")
            if padding_mode == "repeat":
                last_frame = input_frames[:, -1:, :, :]  # Get the last frame
                padding = last_frame.repeat(1, num_video_frames_per_chunk - num_total_frames, 1, 1)
                input_frames = torch.cat([input_frames, padding], dim=1)
            elif padding_mode == "reflect":
                while input_frames.shape[1] < num_video_frames_per_chunk:
                    padding = min(input_frames.shape[1] - 1, num_video_frames_per_chunk - input_frames.shape[1])
                    padding_frames = input_frames.flip(dims=[1])[:, :padding, :, :]
                    input_frames = torch.cat([input_frames, padding_frames], dim=1)
            else:
                raise ValueError(f"Invalid padding mode: {padding_mode}")
        return input_frames

    def _batch_load_videos(
        self,
        video_paths: dict[str, str],
        resolution: str,
        max_frames: int | None = None,
    ) -> dict[str, tuple[torch.Tensor, int, str, tuple[int, int]]]:
        """
        Load multiple videos concurrently using thread pool to overlap hardware decoder initialization.
        
        Args:
            video_paths: Dictionary mapping identifier to video path.
            resolution: Target resolution for all videos.
            max_frames: Maximum frames for main video (None for controls).
            
        Returns:
            Dictionary mapping identifier to (frames, fps, aspect_ratio, original_hw).
        """
        results = {}
        
        # Modality to interpolation mapping
        modality_interpolation = {
            "main": INTER_AREA,
            "edge": INTER_LINEAR,
            "vis": INTER_AREA,
            "depth": INTER_LINEAR,
            "seg": INTER_NEAREST,
            "inpaint": INTER_LINEAR,
            "edge_mask": INTER_LINEAR,
            "vis_mask": INTER_LINEAR,
            "depth_mask": INTER_LINEAR,
            "seg_mask": INTER_LINEAR,
            "inpaint_mask": INTER_LINEAR,
        }
        
        def load_single_video(key: str, path: str) -> tuple[str, tuple]:
            """Load a single video file."""
            try:
                interpolation = modality_interpolation.get(key, INTER_AREA)
                # For main video, use max_frames; for controls, load all frames
                num_frames = max_frames if key == "main" else None
                
                frames, fps, aspect_ratio, original_hw = read_and_resize_input(
                    path,
                    num_total_frames=num_frames if num_frames else 5000,
                    interpolation=interpolation,
                    resolution=resolution,
                )
                return key, (frames, fps, aspect_ratio, original_hw)
            except Exception as e:
                log.warning(f"Failed to load video {key} from {path}: {e}")
                return key, None
        
        # Load all videos concurrently
        with ThreadPoolExecutor(max_workers=len(video_paths)) as executor:
            futures = {
                executor.submit(load_single_video, key, path): key 
                for key, path in video_paths.items()
            }
            
            for future in as_completed(futures):
                key, result = future.result()
                if result is not None:
                    results[key] = result
        
        return results

    def _load_inputs_sequential(
        self,
        video_path: str,
        resolution: str,
        max_frames: int | None,
        hint_key: list[str],
        seg_control_prompt: str | None,
        input_control_video_paths: dict[str, str] | None,
        image_context_path: Optional[str],
        context_frame_idx: int | None,
    ) -> tuple[torch.Tensor, int, str, tuple[int, int], torch.Tensor | None, dict[str, torch.Tensor]]:
        """
        Original sequential loading approach - loads video, image context, and control inputs one by one.
        Each video file incurs the hardware decoder initialization overhead (~400ms).
        """
        log.info("Rank 0: Using SEQUENTIAL loading (COSMOS_BATCHED_VIDEO_LOADING=0)...")
        
        # Load main video
        torch.cuda.nvtx.range_push("read_and_process_video")
        input_frames, fps, aspect_ratio, original_hw = read_and_process_video(
            video_path, resolution=resolution, max_frames=max_frames
        )
        torch.cuda.nvtx.range_pop()
        
        if input_frames.shape[1] == 0:
            raise ValueError("Input video is empty")

        # Process image context if provided
        log.info("Rank 0: Processing image context if available...")
        actual_image_context_path = image_context_path
        if context_frame_idx is not None:
            actual_image_context_path = video_path
            log.info(f"Using context frame index: {context_frame_idx} from video path: {video_path}")
        
        torch.cuda.nvtx.range_push("read_and_process_image_context")
        image_context = read_and_process_image_context(
            actual_image_context_path,
            resolution=(VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]),
            resize=True,
            context_frame_idx=context_frame_idx,
        )
        torch.cuda.nvtx.range_pop()
        
        # Load control inputs sequentially
        log.info("Rank 0: Loading control inputs...")
        torch.cuda.nvtx.range_push("rank0_read_and_process_control_input")
        control_input_dict = read_and_process_control_input(
            video_path=video_path,
            input_control_paths=input_control_video_paths,
            hint_key=hint_key,
            resolution=resolution,
            seg_control_prompt=seg_control_prompt,
            preloaded_frames=input_frames,  # Pass preloaded frames to avoid re-reading video for depth/seg
        )
        torch.cuda.nvtx.range_pop()
        
        return input_frames, fps, aspect_ratio, original_hw, image_context, control_input_dict

    def _load_inputs_batched(
        self,
        video_path: str,
        resolution: str,
        max_frames: int | None,
        hint_key: list[str],
        seg_control_prompt: str | None,
        input_control_video_paths: dict[str, str] | None,
        image_context_path: Optional[str],
        context_frame_idx: int | None,
    ) -> tuple[torch.Tensor, int, str, tuple[int, int], torch.Tensor | None, dict[str, torch.Tensor]]:
        """
        Batched concurrent loading approach - loads all video files in parallel using ThreadPoolExecutor.
        Hardware decoder initialization overhead is overlapped across all files.
        """
        log.info("Rank 0: Using BATCHED loading (COSMOS_BATCHED_VIDEO_LOADING=1)...")
        
        # -------- Collect all video paths for batched loading --------
        torch.cuda.nvtx.range_push("rank0_batch_load_videos")
        log.info("Rank 0: Collecting video paths for batched loading...")
        
        video_paths_to_load = {"main": video_path}
        
        # Add control video paths that exist
        if input_control_video_paths:
            for modality in hint_key:
                control_path = input_control_video_paths.get(modality)
                if control_path and os.path.exists(control_path):
                    video_paths_to_load[modality] = control_path
                
                # Also check for mask files
                mask_path = input_control_video_paths.get(f"{modality}_mask")
                if mask_path and os.path.exists(mask_path):
                    video_paths_to_load[f"{modality}_mask"] = mask_path
        
        log.info(f"Rank 0: Batched loading {len(video_paths_to_load)} video(s) concurrently: {list(video_paths_to_load.keys())}")
        
        # Load all videos concurrently
        loaded_videos = self._batch_load_videos(
            video_paths_to_load,
            resolution=resolution,
            max_frames=max_frames,
        )
        torch.cuda.nvtx.range_pop()
        
        # Extract main video results
        if "main" not in loaded_videos:
            raise ValueError("Failed to load main input video")
        
        input_frames, fps, aspect_ratio, original_hw = loaded_videos["main"]
        
        if input_frames.shape[1] == 0:
            raise ValueError("Input video is empty")
        
        # Build control_input_dict from loaded control videos
        control_input_dict = {}
        for modality in hint_key:
            control_key = f"control_input_{modality}"
            if modality in loaded_videos:
                control_input_dict[control_key] = loaded_videos[modality][0]  # Just the frames tensor
            
            # Handle masks
            mask_key = f"{modality}_mask"
            if mask_key in loaded_videos:
                mask_frames = loaded_videos[mask_key][0]
                control_input_dict[f"{control_key}_mask"] = (mask_frames[:1] > 127.5).to(torch.bool)

        # Process image context if provided
        log.info("Rank 0: Processing image context if available...")
        actual_image_context_path = image_context_path
        if context_frame_idx is not None:
            actual_image_context_path = video_path
            log.info(f"Using context frame index: {context_frame_idx} from video path: {video_path}")
        
        torch.cuda.nvtx.range_push("rank0_read_and_process_image_context")
        image_context = read_and_process_image_context(
            actual_image_context_path,
            resolution=(VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]),
            resize=True,
            context_frame_idx=context_frame_idx,
        )
        torch.cuda.nvtx.range_pop()
        
        # For modalities not loaded from files, compute on-the-fly if needed
        # (depth from preloaded frames, seg via SAM2, edge/vis via augmentor)
        torch.cuda.nvtx.range_push("rank0_compute_missing_controls")
        for modality in hint_key:
            control_key = f"control_input_{modality}"
            if control_key not in control_input_dict:
                # Need to compute this modality on-the-fly
                if modality == "depth":
                    log.info("Computing depth on-the-fly from preloaded frames...")
                    fallback_dict = read_and_process_control_input(
                        video_path=video_path,
                        input_control_paths={},
                        hint_key=["depth"],
                        resolution=resolution,
                        seg_control_prompt=None,
                        preloaded_frames=input_frames,
                    )
                    if "control_input_depth" in fallback_dict:
                        control_input_dict["control_input_depth"] = fallback_dict["control_input_depth"]
                elif modality == "seg" and seg_control_prompt:
                    log.info("Computing segmentation on-the-fly...")
                    fallback_dict = read_and_process_control_input(
                        video_path=video_path,
                        input_control_paths={},
                        hint_key=["seg"],
                        resolution=resolution,
                        seg_control_prompt=seg_control_prompt,
                        preloaded_frames=input_frames,
                    )
                    if "control_input_seg" in fallback_dict:
                        control_input_dict["control_input_seg"] = fallback_dict["control_input_seg"]
                # edge/vis will be computed by augmentor, no need to handle here
        torch.cuda.nvtx.range_pop()
        
        return input_frames, fps, aspect_ratio, original_hw, image_context, control_input_dict

    def _load_and_broadcast_inputs(
        self,
        video_path: str,
        resolution: str = "720",
        max_frames: int | None = None,
        hint_key: list[str] = ["edge"],
        seg_control_prompt: str | None = None,
        input_control_video_paths: dict[str, str] | None = None,
        image_context_path: Optional[str] = None,
        context_frame_idx: int | None = None,
    ) -> tuple[torch.Tensor, int, str, tuple[int, int], torch.Tensor | None, dict[str, torch.Tensor]]:
        """
        Load video, image context, and control inputs on rank 0 and broadcast to all ranks.
        
        Uses environment variable COSMOS_BATCHED_VIDEO_LOADING to choose loading strategy:
        - COSMOS_BATCHED_VIDEO_LOADING=1 (default): Batched concurrent loading - overlaps hardware
          decoder initialization overhead across all video files.
        - COSMOS_BATCHED_VIDEO_LOADING=0: Sequential loading - loads videos one by one (original behavior).

        Args:
            video_path: Path to the input video file.
            resolution: Target resolution (e.g., "720", "480").
            max_frames: Maximum number of frames to read from the video.
            hint_key: List of control modalities to process.
            seg_control_prompt: Text prompt for SAM2 segmentation.
            input_control_video_paths: Dictionary mapping modality to file path.
            image_context_path: Path to image file to use as image context.
            context_frame_idx: Frame index of the input video to use as image context.

        Returns:
            input_frames: Processed video tensor (C, T, H, W).
            fps: Frames per second of the original input video.
            aspect_ratio: Aspect ratio of the original input video.
            original_hw: Original height and width of the input video.
            image_context: Image context tensor or None.
            control_input_dict: Dictionary mapping control input keys to tensors.
        """
        rank = get_rank()
        world_size = get_world_size()
        
        # Check environment variable for loading strategy
        # Default to sequential loading (0), set to 1 for batched loading with broadcast
        use_batched_loading = os.environ.get("COSMOS_BATCHED_VIDEO_LOADING", "0").lower() not in ("0", "false", "no")
        
        # -------- Sequential loading (original behavior) - each rank loads its own data --------
        if not use_batched_loading:
            log.info(f"Rank {rank}: Using SEQUENTIAL loading without broadcast (COSMOS_BATCHED_VIDEO_LOADING=0)...")
            return self._load_inputs_sequential(
                video_path=video_path,
                resolution=resolution,
                max_frames=max_frames,
                hint_key=hint_key,
                seg_control_prompt=seg_control_prompt,
                input_control_video_paths=input_control_video_paths,
                image_context_path=image_context_path,
                context_frame_idx=context_frame_idx,
            )
        
        # -------- Batched loading with broadcast - load on rank 0 and broadcast to all ranks --------
        log.info(f"Rank {rank}: Using BATCHED loading with broadcast (COSMOS_BATCHED_VIDEO_LOADING=1)...")
        
        # Initialize placeholders for data that will be broadcast
        input_frames = None
        fps = None
        aspect_ratio = None
        original_hw = None
        image_context = None
        control_input_dict = {}
        
        if rank == 0:
            input_frames, fps, aspect_ratio, original_hw, image_context, control_input_dict = self._load_inputs_batched(
                video_path=video_path,
                resolution=resolution,
                max_frames=max_frames,
                hint_key=hint_key,
                seg_control_prompt=seg_control_prompt,
                input_control_video_paths=input_control_video_paths,
                image_context_path=image_context_path,
                context_frame_idx=context_frame_idx,
            )
        
        # -------- Broadcast data to all ranks --------
        if world_size > 1:
            log.info(f"Rank {rank}: Broadcasting loaded data from rank 0...")
            torch.cuda.nvtx.range_push("broadcast_inputs")
            
            # First broadcast metadata (fps, aspect_ratio, original_hw)
            # We need to send these as tensors
            if rank == 0:
                metadata = torch.tensor([
                    fps,
                    int(original_hw[0]),
                    int(original_hw[1]),
                ], dtype=torch.int64, device='cuda')
                # Encode aspect_ratio as index
                aspect_ratio_map = {"16,9": 0, "4,3": 1, "1,1": 2, "3,4": 3, "9,16": 4}
                aspect_ratio_idx = aspect_ratio_map.get(aspect_ratio, 0)
                metadata = torch.cat([metadata, torch.tensor([aspect_ratio_idx], dtype=torch.int64, device='cuda')])
            else:
                metadata = torch.zeros(4, dtype=torch.int64, device='cuda')
            
            broadcast(metadata, src=0)
            
            if rank != 0:
                fps = int(metadata[0].item())
                original_hw = (int(metadata[1].item()), int(metadata[2].item()))
                aspect_ratio_reverse_map = {0: "16,9", 1: "4,3", 2: "1,1", 3: "3,4", 4: "9,16"}
                aspect_ratio = aspect_ratio_reverse_map[int(metadata[3].item())]
            
            # Broadcast input_frames shape first, then the tensor
            if rank == 0:
                frames_shape = torch.tensor(list(input_frames.shape), dtype=torch.int64, device='cuda')
            else:
                frames_shape = torch.zeros(4, dtype=torch.int64, device='cuda')  # C, T, H, W
            
            broadcast(frames_shape, src=0)
            
            if rank != 0:
                input_frames = torch.zeros(
                    tuple(frames_shape.tolist()),
                    dtype=torch.uint8,
                    device='cuda'
                )
            else:
                input_frames = input_frames.cuda()
            
            # Broadcast the actual input frames
            # Convert to float for broadcast (uint8 may have issues with some NCCL versions)
            # Ensure tensor is contiguous before broadcast
            input_frames_float = input_frames.float().contiguous()
            broadcast(input_frames_float, src=0)
            input_frames = input_frames_float.to(torch.uint8).cpu()
            
            # Broadcast image_context if it exists
            if rank == 0:
                has_image_context = torch.tensor([1 if image_context is not None else 0], dtype=torch.int64, device='cuda')
            else:
                has_image_context = torch.zeros(1, dtype=torch.int64, device='cuda')
            
            broadcast(has_image_context, src=0)
            
            if has_image_context.item() == 1:
                if rank == 0:
                    ic_shape = torch.tensor(list(image_context.shape), dtype=torch.int64, device='cuda')
                else:
                    ic_shape = torch.zeros(4, dtype=torch.int64, device='cuda')  # B, C, H, W
                
                broadcast(ic_shape, src=0)
                
                if rank != 0:
                    image_context = torch.zeros(
                        tuple(ic_shape.tolist()),
                        dtype=torch.bfloat16,
                        device='cuda'
                    )
                else:
                    image_context = image_context.cuda().contiguous()
                
                broadcast(image_context, src=0)
            
            # Broadcast control inputs
            if rank == 0:
                num_controls = torch.tensor([len(control_input_dict)], dtype=torch.int64, device='cuda')
                # Encode control keys as indices
                control_key_map = {
                    "control_input_edge": 0, "control_input_vis": 1, "control_input_depth": 2,
                    "control_input_seg": 3, "control_input_inpaint": 4, "control_input_hdmap_bbox": 5,
                    "control_input_inpaint_mask": 6, "control_input_edge_mask": 7,
                    "control_input_vis_mask": 8, "control_input_depth_mask": 9, "control_input_seg_mask": 10,
                }
                control_keys_tensor = torch.tensor(
                    [control_key_map.get(k, -1) for k in control_input_dict.keys()],
                    dtype=torch.int64, device='cuda'
                )
            else:
                num_controls = torch.zeros(1, dtype=torch.int64, device='cuda')
            
            broadcast(num_controls, src=0)
            
            if num_controls.item() > 0:
                if rank != 0:
                    control_keys_tensor = torch.zeros(int(num_controls.item()), dtype=torch.int64, device='cuda')
                else:
                    # Ensure contiguous before broadcast
                    control_keys_tensor = control_keys_tensor.contiguous()
                
                broadcast(control_keys_tensor, src=0)
                
                control_key_reverse_map = {
                    0: "control_input_edge", 1: "control_input_vis", 2: "control_input_depth",
                    3: "control_input_seg", 4: "control_input_inpaint", 5: "control_input_hdmap_bbox",
                    6: "control_input_inpaint_mask", 7: "control_input_edge_mask",
                    8: "control_input_vis_mask", 9: "control_input_depth_mask", 10: "control_input_seg_mask",
                }
                
                if rank == 0:
                    control_keys_list = list(control_input_dict.keys())
                else:
                    control_keys_list = [control_key_reverse_map[int(k.item())] for k in control_keys_tensor]
                    control_input_dict = {}
                
                for i, key in enumerate(control_keys_list):
                    if rank == 0:
                        ctrl_tensor = control_input_dict[key]
                        if ctrl_tensor is None:
                            ctrl_shape = torch.tensor([-1, -1, -1, -1], dtype=torch.int64, device='cuda')
                        else:
                            ctrl_shape = torch.tensor(list(ctrl_tensor.shape), dtype=torch.int64, device='cuda')
                    else:
                        ctrl_shape = torch.zeros(4, dtype=torch.int64, device='cuda')
                    
                    broadcast(ctrl_shape, src=0)
                    
                    if ctrl_shape[0].item() == -1:
                        # None tensor
                        if rank != 0:
                            control_input_dict[key] = None
                        continue
                    
                    if rank != 0:
                        # Determine dtype based on key (masks are bool)
                        if "mask" in key:
                            ctrl_tensor = torch.zeros(
                                tuple(ctrl_shape.tolist()),
                                dtype=torch.bool,
                                device='cuda'
                            )
                        else:
                            ctrl_tensor = torch.zeros(
                                tuple(ctrl_shape.tolist()),
                                dtype=torch.uint8,
                                device='cuda'
                            )
                    else:
                        ctrl_tensor = ctrl_tensor.cuda()
                    
                    # Broadcast as float, convert back
                    # Ensure tensor is contiguous before broadcast
                    ctrl_float = ctrl_tensor.float().contiguous()
                    broadcast(ctrl_float, src=0)
                    
                    if "mask" in key:
                        ctrl_tensor = ctrl_float.bool().cpu()
                    else:
                        ctrl_tensor = ctrl_float.to(torch.uint8).cpu()
                    
                    if rank != 0:
                        control_input_dict[key] = ctrl_tensor
                    else:
                        control_input_dict[key] = ctrl_tensor
            
            torch.cuda.nvtx.range_pop()
        
        # Ensure all ranks are synchronized before continuing
        barrier()
        
        return input_frames, fps, aspect_ratio, original_hw, image_context, control_input_dict

    @torch.no_grad()
    def generate_image2world_from_embeddings(
        self,
        text_embeddings: torch.Tensor,
        video_path: str,
        guidance: int = 7,
        seed: int = 1,
        resolution: str = "720",
        num_conditional_frames: int = 1,
        num_video_frames_per_chunk: int = 93,
        num_steps: int = 35,
        control_weight: str = "1.0",
        sigma_max: float | None = None,
        hint_key: list[str] = ["edge"],
        preset_edge_threshold: str = "medium",
        preset_blur_strength: str = "medium",
        seg_control_prompt: str | None = None,
        input_control_video_paths: dict[str, str] | None = None,
        show_control_condition: bool = False,
        show_input: bool = False,
        image_context_path: Optional[str] = None,
        keep_input_resolution: bool = True,
        negative_prompt: str | None = None,
        max_frames: int | None = None,
        context_frame_idx: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], int, tuple[int, int]]:
        torch.cuda.nvtx.range_push("generate_image2world_from_embeddings")
        assert negative_prompt is None or self.neg_t5_embeddings is not None, "Negative prompt embedding is not computed."
        
        # --------Input processing--------
        # Load video, image context, and control inputs on rank 0 and broadcast to all ranks.
        # This avoids redundant I/O and prevents multiple CUDA context initializations.
        log.info("Loading and broadcasting inputs across ranks...")
        torch.cuda.nvtx.range_push("load_and_broadcast_inputs")
        input_frames, fps, aspect_ratio, original_hw, image_context, control_input_dict = self._load_and_broadcast_inputs(
            video_path=video_path,
            resolution=resolution,
            max_frames=max_frames,
            hint_key=hint_key,
            seg_control_prompt=seg_control_prompt,
            input_control_video_paths=input_control_video_paths,
            image_context_path=image_context_path,
            context_frame_idx=context_frame_idx,
        )
        torch.cuda.nvtx.range_pop()

        # -------- Stuff to handle chunk-wise long video generation --------
        torch.cuda.nvtx.range_push("prepare_chunks")
        num_total_frames, num_chunks, num_frames_per_chunk = self._get_num_chunks(
            input_frames, num_video_frames_per_chunk, num_conditional_frames
        )
        # Pad input frames if total frames is less than chunk size
        input_frames = self._pad_input_frames(input_frames, num_total_frames, num_video_frames_per_chunk)
        all_chunks, time_per_chunk = [], []
        # Initialize control_video_dict to accumulate control inputs across chunks
        control_video_dict = {}
        all_control_chunks = {key: [] for key in hint_key}
        # For first chunk, use zeros as input (after normalization it is 0)
        prev_output = torch.zeros_like(input_frames[:, :num_video_frames_per_chunk]).to(torch.uint8).cuda()[None]
        torch.cuda.nvtx.range_pop()

        # --------Start of chunk-wise long video generation--------
        for chunk_id in range(num_chunks):
            torch.cuda.nvtx.range_push(f"chunk_{chunk_id}")
            log.info(f"Generating chunk {chunk_id + 1}/{num_chunks}")
            start_time = time.perf_counter()

            # Calculate start frame for this chunk
            chunk_start_frame = chunk_id * num_frames_per_chunk
            chunk_end_frame = min(chunk_start_frame + num_video_frames_per_chunk, input_frames.shape[1])

            x_sigma_max = None
            if input_frames is not None:
                torch.cuda.nvtx.range_push("prepare_input_frames")
                cur_input_frames = input_frames[:, chunk_start_frame:chunk_end_frame]
                cur_input_frames = self._pad_input_frames(
                    cur_input_frames, cur_input_frames.shape[1], num_video_frames_per_chunk
                )
                torch.cuda.nvtx.range_pop()
                if sigma_max is not None:
                    torch.cuda.nvtx.range_push("encode_x_sigma_max")
                    x0 = uint8_to_normalized_float(cur_input_frames, dtype=torch.bfloat16)[None].cuda()
                    x0 = self.model.encode(x0).contiguous()
                    x_sigma_max = self.model.get_x_from_clean(x0, sigma_max, seed=(seed + chunk_id))
                    torch.cuda.nvtx.range_pop()

            if isinstance(text_embeddings, list):
                text_emb_idx = min(chunk_id, len(text_embeddings) - 1)
                text_embedding = text_embeddings[text_emb_idx]
            else:
                text_embedding = text_embeddings

            # Prepare the data batch with current input. Note: this doesn't include control inputs yet.
            torch.cuda.nvtx.range_push("get_data_batch_input")
            data_batch = self._get_data_batch_input(
                cur_input_frames,
                prev_output,
                text_embedding,
                fps,
                negative_prompt=negative_prompt,
                control_weight=control_weight,
                image_context=image_context,
            )
            torch.cuda.nvtx.range_pop()

            # Process control inputs as specified in the hint_key list.
            # If pre-computed control inputs are provided, load them into the data batch.
            torch.cuda.nvtx.range_push("process_control_inputs")
            for k, v in control_input_dict.items():
                cur_control_input = v[:, chunk_start_frame:chunk_end_frame]
                data_batch[k] = self._pad_input_frames(
                    cur_control_input, cur_control_input.shape[1], num_video_frames_per_chunk
                )
                if k == "control_input_inpaint_mask":
                    data_batch["control_input_inpaint"] = cur_input_frames
            torch.cuda.nvtx.range_pop()
            # Otherwise, compute control inputs on-the-fly via the augmentor（applicable to edge and vis).
            torch.cuda.nvtx.range_push("get_augmentor_for_eval")
            data_batch = get_augmentor_for_eval(
                data_dict=data_batch,
                input_keys=["input_video"],
                output_keys=hint_key,
                preset_edge_threshold=preset_edge_threshold,
                preset_blur_strength=preset_blur_strength,
            )
            torch.cuda.nvtx.range_pop()

            if chunk_id == 0:
                data_batch[NUM_CONDITIONAL_FRAMES_KEY] = 0
            else:
                data_batch[NUM_CONDITIONAL_FRAMES_KEY] = (
                    1 + (num_conditional_frames - 1) // 4
                )  # tokenizer temporal compression is 4x

            random.seed(seed)
            seed = random.randint(0, 1000000)
            log.info(f"Seed: {seed}")

            # Save all tensors for debugging before generate_samples_from_batch
            # debug_dir = "debug_tensors"
            # os.makedirs(debug_dir, exist_ok=True)
            # debug_filename = os.path.join(debug_dir, f"data_batch_chunk_{chunk_id}_seed_{seed}.pt")
            # # Convert all tensors to CPU before saving to avoid device issues
            # data_batch_cpu = {}
            # for k, v in data_batch.items():
            #     if isinstance(v, torch.Tensor):
            #         data_batch_cpu[k] = v.cpu()
            #     else:
            #         data_batch_cpu[k] = v
            # torch.save(data_batch_cpu, debug_filename)
            # log.info(f"Saved debug tensors to {debug_filename}")

            # Generate and decode video
            torch.cuda.nvtx.range_push("generate_samples_from_batch")
            sample = self.model.generate_samples_from_batch(
                data_batch,
                n_sample=1,
                guidance=guidance,
                seed=seed,
                is_negative_prompt=negative_prompt is not None,
                x_sigma_max=x_sigma_max,
                sigma_max=sigma_max,
                num_steps=num_steps,
            )
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("decode")
            video = self.model.decode(sample).cpu()  # Shape: (1, C, T, H, W)
            torch.cuda.nvtx.range_pop()

            # For visualization: concatenate condition and input videos with generated video
            video_cat = video
            conditions = []
            if show_input and input_frames is not None:
                x0 = uint8_to_normalized_float(cur_input_frames, dtype=torch.bfloat16)[None]
                video_cat = torch.cat([x0, video_cat], dim=-1)

            # Accumulate control inputs for each chunk
            torch.cuda.nvtx.range_push("accumulate_control_chunks")
            for key in hint_key:
                control_input = data_batch["control_input_" + key]
                if f"control_input_{key}_mask" in data_batch:
                    control_input = (control_input + 1) / 2 * data_batch[f"control_input_{key}_mask"] * 2 - 1

                # Store control input for this chunk
                if chunk_id == 0:
                    all_control_chunks[key].append(control_input.cpu())
                else:
                    # For subsequent chunks, only append the non-overlapping frames
                    all_control_chunks[key].append(control_input[:, :, num_conditional_frames:, :, :].cpu())

                if show_control_condition:
                    conditions += [control_input.cpu()]
            torch.cuda.nvtx.range_pop()

            if show_control_condition:
                video_cat = torch.cat([*conditions, video_cat], dim=-1)

            if chunk_id == 0:
                all_chunks.append(video_cat.cpu())
            else:
                # For subsequent chunks, only append the non-overlapping frames
                all_chunks.append(video_cat[:, :, num_conditional_frames:, :, :].cpu())

            # For next chunk, use last conditional_frames as input
            if chunk_id < num_chunks - 1:  # Don't need to prepare next input for last chunk
                torch.cuda.nvtx.range_push("prepare_next_chunk_input")
                last_frames = video[:, :, -num_conditional_frames:, :, :]  # (1, C, num_conditional_frames, H, W)
                # Convert to uint8 [0, 255]
                last_frames_uint8 = normalized_float_to_uint8(last_frames)
                # Create blank frames for the rest
                blank_frames = torch.zeros(
                    (
                        1,
                        3,
                        num_video_frames_per_chunk - num_conditional_frames,
                        video.shape[-2],
                        video.shape[-1],
                    ),
                    dtype=torch.uint8,
                    device=video.device,
                )
                prev_output = torch.cat([last_frames_uint8, blank_frames], dim=2)
                torch.cuda.nvtx.range_pop()
            end_time = time.perf_counter()
            time_per_chunk.append(end_time - start_time)
            torch.cuda.nvtx.range_pop()  # chunk_{chunk_id}

        # Concatenate all chunks along time
        torch.cuda.nvtx.range_push("concatenate_chunks")
        full_video = torch.cat(all_chunks, dim=2)  # (1, C, T, H, W)
        # Keep only the original number of frames
        full_video = full_video[:, :, :num_total_frames, :, :]

        # Concatenate all control chunks and trim to original frames
        for key in hint_key:
            if all_control_chunks[key]:
                control_video_dict[key] = torch.cat(all_control_chunks[key], dim=2)  # (1, C, T, H, W)
                # Keep only the original number of frames
                control_video_dict[key] = control_video_dict[key][:, :, :num_total_frames, :, :]
        torch.cuda.nvtx.range_pop()

        if keep_input_resolution:
            torch.cuda.nvtx.range_push("reshape_output_to_input_resolution")
            # reshape output video to match the input video resolution
            full_video = reshape_output_video_to_input_resolution(
                full_video, hint_key, show_control_condition, show_input, original_hw
            )
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("reshape_control_videos_to_input_resolution")
            # Also resize control videos to match input resolution
            for key in hint_key:
                if key in control_video_dict and control_video_dict[key] is not None:
                    control_video_dict[key] = reshape_output_video_to_input_resolution(
                        control_video_dict[key], [key], False, False, original_hw
                    )
            torch.cuda.nvtx.range_pop()
        log.info(f"Average time per chunk: {sum(time_per_chunk) / len(time_per_chunk)}")
        torch.cuda.nvtx.range_pop()  # generate_image2world_from_embeddings
        return full_video, control_video_dict, fps, original_hw

    @torch.no_grad()
    def generate_img2world(
        self,
        prompt: str | torch.Tensor | list[str] | dict[str, str],
        video_path: str,
        guidance: int = 7,
        seed: int = 1,
        resolution: str = "720",
        num_conditional_frames: int = 1,
        num_video_frames_per_chunk: int = 93,
        num_steps: int = 35,
        control_weight: str = "1.0",
        sigma_max: float | None = None,
        hint_key: list[str] = ["edge"],
        preset_edge_threshold: str = "medium",
        preset_blur_strength: str = "medium",
        seg_control_prompt: str | None = None,
        input_control_video_paths: dict[str, str] | None = None,
        show_control_condition: bool = False,
        show_input: bool = False,
        image_context_path: Optional[str] = None,
        keep_input_resolution: bool = True,
        negative_prompt: str | None = None,
        max_frames: int | None = None,
        context_frame_idx: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], int, tuple[int, int]]:
        """
        Generates a video based on an input video and text prompt.
        Supports chunk-wise long video generation.

        Args:
            prompt (str): The text prompt describing the desired video content/style.
            video_path (str): Path to the input conditional video.
            guidance (int, optional): Classifier-free guidance scale. Defaults to 7.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            resolution (str, optional): Resolution of the video (720-default, 480, etc). Defaults to 720.
            image_context_path (str, optional): Path to image file to use as image context. If None, uses random frame from video. Will be ignored and use input video if context_frame_idx is provided.
            keep_input_resolution (bool, optional): Whether to keep the exact dimension of the. Defaults to True.
            negative_prompt (str, optional): Negative prompt for classifier-free guidance. Defaults to None.
            max_frames (int, optional): Maximum number of frames to read from the video. Defaults to None. 1 for image.
            context_frame_idx (int, optional): Frame index of the input video to use as image context. Defaults to None. In this case, can still use image_context_path to provide image context.
        Returns:
            torch.Tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1].
            dict[str, torch.Tensor]: Dictionary mapping hint key to the corresponding control input video tensor.
            int: Frames per second of the original input video.
            tuple[int, int]: Original height and width of the input video.

        Raises:
            ValueError: If the input video is empty or invalid.
        """
        # Get text context embeddings
        log.info("Computing prompt text embeddings...")
        if self.text_encoder_class == "T5":
            text_embeddings = get_t5_from_prompt(prompt, text_encoder_class="T5", cache_dir=self.cache_dir)
        else:
            text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                {"ai_caption": [prompt], "images": None}, input_caption_key="ai_caption"
            )
        if negative_prompt:
            log.info("Computing negative prompt text embeddings...")
            if self.text_encoder_class == "T5":
                neg_text_embeddings = get_t5_from_prompt(
                    negative_prompt, text_encoder_class="T5", cache_dir=self.cache_dir
                )
            else:
                neg_text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                    {"ai_caption": [negative_prompt], "images": None}, input_caption_key="ai_caption"
                )
            self.neg_t5_embeddings = neg_text_embeddings

        return self.generate_image2world_from_embeddings(
            text_embeddings=text_embeddings,
            video_path=video_path,
            guidance=guidance,
            seed=seed,
            resolution=resolution,
            num_conditional_frames=num_conditional_frames,
            num_video_frames_per_chunk=num_video_frames_per_chunk,
            num_steps=num_steps,
            control_weight=control_weight,
            sigma_max=sigma_max,
            hint_key=hint_key,
            preset_edge_threshold=preset_edge_threshold,
            preset_blur_strength=preset_blur_strength,
            seg_control_prompt=seg_control_prompt,
            input_control_video_paths=input_control_video_paths,
            show_control_condition=show_control_condition,
            show_input=show_input,
            image_context_path=image_context_path,
            keep_input_resolution=keep_input_resolution,
            negative_prompt=negative_prompt,
            max_frames=max_frames,
            context_frame_idx=context_frame_idx,
        )
