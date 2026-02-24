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

"""Database of released checkpoints.

The database maps checkpoint internal URIs to public URIs and associates metadata (e.g.
experiment name).

## Usage

Register a checkpoint:

```python
CheckpointConfig(
    uuid="0e8177cc-0db5-4cfd-a8a4-b820c772f4fc",
    s3=CheckpointDirS3(
        uri="s3://bucket/path/to/checkpoint",
    ),
    hf=CheckpointDirHf(
        repository="org/repo",
        revision="revision",
        subdirectory="path/to/checkpoint",
    ),
).register()
```

Checkpoints can be referenced by UUID, S3 URI, or local path. Optionally, use `get_checkpoint_uri` to validate and normalize the URI.

```python
# S3 URI
checkpoint_uri = get_checkpoint_uri("s3://bucket/path/to/checkpoint")
# UUID
checkpoint_uri = get_checkpoint_uri("0e8177cc-0db5-4cfd-a8a4-b820c772f4fc")
# Local path
checkpoint_uri = get_checkpoint_uri("/path/to/checkpoint", check_exists=True)
```

When the checkpoint is loaded, call 'download_checkpoint':

```python
from cosmos_transfer2._src.imaginaire.flags import INTERNAL

if not INTERNAL:
    from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import download_checkpoint

    checkpoint_uri = download_checkpoint(checkpoint_uri)

load_checkpoint(checkpoint_uri)
```
"""

import functools
import os
import shlex
import subprocess
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, TypeAlias

import pydantic
from typing_extensions import Self, override

from cosmos_transfer2._src.imaginaire.flags import EXPERIMENTAL_CHECKPOINTS, INTERNAL
from cosmos_transfer2._src.imaginaire.utils import log

_MINIMUM_HF_CLI_VERSION = "1.3.5"


def _is_uuid(checkpoint_uri: str) -> bool:
    """Return True if the URI is a UUID."""
    try:
        uuid.UUID(str(checkpoint_uri))
        return True
    except ValueError:
        return False


def _is_path(checkpoint_uri: str) -> bool:
    """Return True if the URI is a local path."""
    return not ("://" in checkpoint_uri or _is_uuid(checkpoint_uri))


def normalize_uri(checkpoint_uri: str) -> str:
    """Normalize checkpoint URI."""
    checkpoint_uri = checkpoint_uri.rstrip("/")
    if checkpoint_uri.startswith("s3://"):
        checkpoint_uri = checkpoint_uri.removesuffix("/model")
    return checkpoint_uri


def sanitize_uri(checkpoint_uri: str) -> str:
    """Sanitize checkpoint URI."""
    checkpoint_uri = normalize_uri(checkpoint_uri)
    if checkpoint_uri.startswith("s3://"):
        checkpoint_uri = checkpoint_uri.removeprefix("s3://").split("/", 1)[1]
        checkpoint_uri = f"s3://bucket/{checkpoint_uri}"
    return checkpoint_uri


class _CheckpointUri(pydantic.BaseModel):
    """Config for checkpoint file/directory."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    metadata: dict = pydantic.Field(default_factory=dict)
    """File metadata.

    Only used for debugging.
    """


def _validate_s3_uri(uri: str) -> str:
    """Validate and normalize S3 URI."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}. Must start with 's3://'")
    return normalize_uri(uri)


S3Uri = Annotated[str, pydantic.AfterValidator(_validate_s3_uri)]


class _CheckpointS3(_CheckpointUri):
    """Config for checkpoint on S3."""

    uri: S3Uri
    """S3 URI."""


class CheckpointFileS3(_CheckpointS3):
    """Config for checkpoint file on S3."""


class CheckpointDirS3(_CheckpointS3):
    """Config for checkpoint directory on S3."""


CheckpointS3: TypeAlias = CheckpointFileS3 | CheckpointDirS3


def _hf_download(cmd_args: list[str]) -> str:
    """Run Hugging Face CLI download command and return the local path.

    Uses a newer Hugging Face CLI version to download checkpoint. The dependency
    version is very old and not robust.
    """
    cmd = [
        "uvx",
        f"hf>={_MINIMUM_HF_CLI_VERSION}",
        "download",
        *cmd_args,
    ]
    log.info(f"{shlex.join(cmd)}")
    subprocess.check_call(cmd, text=True)
    return subprocess.check_output([*cmd, "--quiet"], text=True, env=dict(os.environ) | {"HF_HUB_OFFLINE": "1"}).strip()


class _CheckpointHf(_CheckpointUri, ABC):
    """Config for checkpoint on Hugging Face."""

    repository: str
    """Repository id (organization/repository)."""
    revision: str
    """Git revision id which can be a branch name, a tag, or a commit hash."""

    _path: str | None = None
    """Local path."""

    @abstractmethod
    def _download(self) -> str: ...

    def download(self) -> str:
        """Download checkpoint and return the local path."""
        if self._path is None:
            self._path = self._download()
        return self._path


class CheckpointFileHf(_CheckpointHf):
    """Config for checkpoint file on Hugging Face."""

    filename: str
    """File name."""

    @override
    def _download(self) -> str:
        """Return local path from NGC workspace, or raise if not found.

        Tries NGC workspace locations before failing — HuggingFace downloads
        are disabled, but the file may be available locally.
        """
        ngc_workspace = os.environ.get("NIM_WORKSPACE", "/opt/nim/workspace")
        candidates = [
            os.path.join(ngc_workspace, self.filename),
            os.path.join(ngc_workspace, "checkpoints", "diffusion", "torch", self.filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise RuntimeError(
            f"Hugging Face downloads are disabled. "
            f"Cannot download {self.filename} from {self.repository}@{self.revision}. "
            f"Searched NGC workspace paths:\n"
            + "\n".join(f"  {p}" for p in candidates)
        )


class CheckpointDirHf(_CheckpointHf):
    """Config for checkpoint directory on Hugging Face."""

    subdirectory: str = ""
    """Repository subdirectory."""
    include: tuple[str, ...] = ()
    """Include patterns.

    See https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download
    """
    exclude: tuple[str, ...] = ()
    """Exclude patterns.

    See https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download
    """

    @override
    def _download(self) -> str:
        """Download checkpoint and return the local path."""
        raise RuntimeError(
            f"Hugging Face downloads are disabled. "
            f"Cannot download directory from {self.repository}@{self.revision}"
            f"{f' (subdirectory: {self.subdirectory})' if self.subdirectory else ''}. "
            f"Please ensure all required checkpoints are available in NGC workspace."
        )


CheckpointHf: TypeAlias = CheckpointFileHf | CheckpointDirHf


class CheckpointConfig(pydantic.BaseModel):
    """Config for checkpoint."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    uuid: str
    """Checkpoint UUID."""
    name: str
    """Checkpoint name.

    Only used for debugging.
    """
    metadata: dict = pydantic.Field(default_factory=dict)
    """Checkpoint metadata.

    Only used for debugging.
    """
    experiment: str | None = None
    """Hydra experiment name."""
    config_file: str | None = None
    """Hydra config file."""

    s3: CheckpointS3
    """Config for checkpoint on S3."""
    hf: CheckpointHf
    """Config for checkpoint on Hugging Face."""

    @property
    def full_name(self) -> str:
        """Return full name for debugging."""
        return f"{self.name}({self.uuid})"

    @cached_property
    def path(self) -> str:
        """Return S3 URI or local path.

        Resolution order (NIM / NGC deployments):
        1. Internal S3 URI (NVIDIA internal only).
        2. Per-checkpoint env var override: CHECKPOINT_<UUID_WITH_UNDERSCORES>=/path
        3. NIM workspace lookup by UUID (legacy layout).
        4. NIM workspace lookup by HF filename/subdirectory (NIM standard layout).
        5. Raise RuntimeError — HuggingFace downloads are disabled.
        """
        if INTERNAL and self.s3 is not None:
            return self.s3.uri

        ngc_workspace = os.environ.get("NIM_WORKSPACE", "/opt/nim/workspace")

        # Per-checkpoint env var override: CHECKPOINT_{UUID}=/path/to/checkpoint
        env_var_name = f"CHECKPOINT_{self.uuid.replace('-', '_').upper()}"
        env_path = os.environ.get(env_var_name)
        if env_path and os.path.exists(env_path):
            return env_path

        # --- Build candidate paths ---
        is_file_checkpoint = isinstance(self.hf, CheckpointFileHf)
        candidate_paths: list[str] = []

        # Legacy layout: {workspace}/{uuid}[.ext]
        candidate_paths.append(os.path.join(ngc_workspace, self.uuid))
        candidate_paths.append(os.path.join(ngc_workspace, "checkpoints", self.uuid))
        if is_file_checkpoint:
            for ext in (".pth", ".pt", ".ckpt"):
                candidate_paths.append(os.path.join(ngc_workspace, f"{self.uuid}{ext}"))
                candidate_paths.append(os.path.join(ngc_workspace, "checkpoints", f"{self.uuid}{ext}"))

        # NIM standard layout: {workspace}/{hf.filename} or {workspace}/{hf.subdirectory}
        # Also check under checkpoints/diffusion/torch/ (NIM deployment layout for diffusion models)
        if is_file_checkpoint and isinstance(self.hf, CheckpointFileHf):
            candidate_paths.append(os.path.join(ngc_workspace, self.hf.filename))
            candidate_paths.append(os.path.join(ngc_workspace, "checkpoints", "diffusion", "torch", self.hf.filename))
        elif isinstance(self.hf, CheckpointDirHf):
            if self.hf.subdirectory:
                candidate_paths.append(os.path.join(ngc_workspace, self.hf.subdirectory))
                candidate_paths.append(os.path.join(ngc_workspace, "checkpoints", "diffusion", "torch", self.hf.subdirectory))
            else:
                # No subdirectory: the workspace root itself may be the checkpoint dir
                candidate_paths.append(ngc_workspace)

        for path in candidate_paths:
            if os.path.exists(path):
                if path == ngc_workspace:
                    # Sanity-check: workspace root must contain a config.json to be valid
                    if not os.path.exists(os.path.join(path, "config.json")):
                        continue
                return path

        raise RuntimeError(
            f"Checkpoint {self.full_name} not found in NGC workspace '{ngc_workspace}'. "
            f"HuggingFace downloads are disabled in this environment. "
            f"Searched paths:\n"
            + "\n".join(f"  {p}" for p in candidate_paths)
            + f"\nTo override, set {env_var_name}=/path/to/checkpoint."
        )

    @classmethod
    def maybe_from_uri(cls, uri: str) -> Self | None:
        """Return checkpoint config for URI if found, otherwise None."""
        uri = normalize_uri(uri)
        return _CHECKPOINTS.get(uri, None)

    @classmethod
    def from_uri(cls, uri: str) -> Self:
        """Return checkpoint config for URI if found, otherwise raise an error."""
        self = cls.maybe_from_uri(uri)
        if self is None:
            raise ValueError(
                f"Checkpoint '{uri}' not found. Set 'export COSMOS_EXPERIMENTAL_CHECKPOINTS=1' to include experimental checkpoints."
            )
        return self

    def register(self):
        """Register checkpoint config."""
        register_checkpoint(self)


_CHECKPOINTS: dict[str, CheckpointConfig] = {}
"""Mapping from checkpoint URI to checkpoint config."""


def register_checkpoint(checkpoint_config: CheckpointConfig):
    """Register checkpoint config.

    DEPRECATED: Use 'CheckpointConfig.register' instead.
    """
    if not EXPERIMENTAL_CHECKPOINTS:
        if checkpoint_config.hf.repository in ["nvidia/Cosmos-Experimental", "nvidia-cosmos-ea/Cosmos-Experimental"]:
            # Don't register experimental checkpoints. An exception will be
            # raised in CI if the checkpoint is used without
            # EXPERIMENTAL_CHECKPOINTS.
            return
    for uri in [checkpoint_config.uuid, checkpoint_config.s3.uri]:
        if uri in _CHECKPOINTS:
            raise ValueError(f"Checkpoint '{uri}' already registered.")
        _CHECKPOINTS[uri] = checkpoint_config


def get_checkpoint_uri(checkpoint_uri: str, *, check_exists: bool = False) -> str:
    """Validate and normalize checkpoint URI."""
    checkpoint_uri = normalize_uri(checkpoint_uri)
    if (checkpoint := CheckpointConfig.maybe_from_uri(checkpoint_uri)) is not None:
        return checkpoint.s3.uri
    if checkpoint_uri.startswith("hf://"):
        return checkpoint_uri
    if _is_path(checkpoint_uri):
        if check_exists:
            checkpoint_path = Path(checkpoint_uri).expanduser().absolute()
            if not checkpoint_path.exists():
                raise ValueError(f"Checkpoint '{checkpoint_path}' does not exist.")
            checkpoint_uri = str(checkpoint_path)
        return checkpoint_uri
    if INTERNAL:
        return checkpoint_uri
    raise ValueError(
        f"Checkpoint '{checkpoint_uri}' not found. Set 'export COSMOS_EXPERIMENTAL_CHECKPOINTS=1' to include experimental checkpoints."
    )


@functools.lru_cache
def _download_hf_checkpoint(checkpoint_hf: str) -> str:
    # Parse hf://org/repo/path/to/file.pth
    assert checkpoint_hf.startswith("hf://"), f"Not a HuggingFace URI: {checkpoint_hf}"
    hf_path = checkpoint_hf.removeprefix("hf://")
    # Split into repo_id (org/repo) and filename (path/to/file.pth)
    parts = hf_path.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid HuggingFace URI format: {checkpoint_hf}. Expected format: hf://org/repo/path/to/file.pth"
        )
    repo_id = "/".join(parts[:2])  # org/repo
    filename = "/".join(parts[2:])  # path/to/file.pth
    return CheckpointFileHf(
        repository=repo_id,
        revision="main",
        filename=filename,
    ).download()


@functools.lru_cache
def download_checkpoint(checkpoint_uri: str, *, check_exists: bool = True) -> str:
    """Download a checkpoint by URI and return the local path.

    This should only be used when the checkpoint is loaded. If you just need a
    URI, use 'get_checkpoint_uri' instead.

    Downloaded checkpoints are cached, so calling this multiple times will
    return the same path.

    Supports:
    - Checkpoint UUID: 0e8177cc-0db5-4cfd-a8a4-b820c772f4fc
    - S3 URI: s3://bucket/path/to/checkpoint
    - HuggingFace URI: hf://org/repo/path/to/file.pth
    - Local path: /path/to/checkpoint
    """
    if INTERNAL:
        return checkpoint_uri
    if (checkpoint := CheckpointConfig.maybe_from_uri(checkpoint_uri)) is not None:
        return checkpoint.download()
    if checkpoint_uri.startswith("hf://"):
        return _download_hf_checkpoint(checkpoint_uri)
    if check_exists and not os.path.exists(checkpoint_uri):
        raise ValueError(f"Checkpoint path {checkpoint_uri} does not exist.")
    return checkpoint_uri


get_checkpoint_path = download_checkpoint
"""DEPRECATED: Use 'download_checkpoint' instead."""
