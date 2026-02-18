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

"""Stub decord package for NIM / inference-only environments.

decord is a training dependency (video dataset loading) that must not be
installed in the inference container for legal reasons.  This stub exposes
the same public symbols so that module-level imports succeed, but raises a
clear RuntimeError if any symbol is actually called.

Usage in Dockerfile:
    COPY packages/decord-stub/decord /path/to/site-packages/decord
"""

_MSG = (
    "decord is not available in this environment. "
    "VideoReader and related symbols are training-only dependencies "
    "and cannot be used during inference."
)


class _Unavailable:
    """Placeholder that raises on instantiation."""

    _name: str = "decord symbol"

    def __init_subclass__(cls, name: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        cls._name = name

    def __init__(self, *args, **kwargs):
        raise RuntimeError(f"{self._name}: {_MSG}")

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"{self._name}: {_MSG}")


class VideoReader(_Unavailable, name="decord.VideoReader"):
    pass


class VideoLoader(_Unavailable, name="decord.VideoLoader"):
    pass


class AVReader(_Unavailable, name="decord.AVReader"):
    pass


def cpu(dev_id: int = 0):
    raise RuntimeError(f"decord.cpu: {_MSG}")


def gpu(dev_id: int = 0):
    raise RuntimeError(f"decord.gpu: {_MSG}")


# Expose a minimal ndarray stub so `import decord; decord.ndarray` doesn't
# raise AttributeError when the module is merely imported.
class _NdarrayModule:
    @staticmethod
    def cpu(dev_id: int = 0):
        raise RuntimeError(f"decord.ndarray.cpu: {_MSG}")

    @staticmethod
    def gpu(dev_id: int = 0):
        raise RuntimeError(f"decord.ndarray.gpu: {_MSG}")


ndarray = _NdarrayModule()

__version__ = "0.0.0+stub"
