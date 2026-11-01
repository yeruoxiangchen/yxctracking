#!/usr/bin/env bash
# Source from the repository root before training, inference or tests.
pose_source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export POSE_ALIGNED_WORKDIR="${1:-${POSE_ALIGNED_WORKDIR:-/data/pose-aligned-reconstruction}}"
export TMPDIR="${POSE_ALIGNED_WORKDIR}/tmp"
export XDG_CACHE_HOME="${POSE_ALIGNED_WORKDIR}/cache"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${pose_source_root}:${pose_source_root}/ReconViaGen:${pose_source_root}/ReconViaGen/wheels/vggt${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TMPDIR}" "${TORCH_HOME}" "${HF_HUB_CACHE}" "${TORCH_EXTENSIONS_DIR}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${CUDA_CACHE_PATH}"
