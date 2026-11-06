#!/usr/bin/env bash
# Run with bash; no executable bit or current-directory assumption required.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NOVA_ROOT="${NOVA_ROOT:-/cloud/nova-avatar}"
NOVA_SHARED_MODEL="${NOVA_SHARED_MODEL:-/model/HuggingFace/Wan-AI/Wan2.2-S2V-14B}"
NOVA_MODEL_DIR="${NOVA_MODEL_DIR:-$NOVA_SHARED_MODEL}"
NOVA_VENV="${NOVA_VENV:-$NOVA_ROOT/.venv}"
export HF_HOME="${HF_HOME:-$NOVA_ROOT/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$NOVA_ROOT/cache/pip}"

usage() {
  echo 'Usage: bash scripts/cloud.sh {init|check|install-deps|install-diffsynth|install-student-deps|infer|model-check|model-repair|student-prepare|student-train|student-mouth-train|student-compare|student-export|student-infer|student-mobile-export|student-mobile-fixture|student-mobile-diagnose}'
  echo 'Inference options: bash scripts/cloud.sh infer --help'
  echo 'Overrides: NOVA_ROOT, NOVA_VENV, NOVA_SHARED_MODEL, NOVA_MODEL_DIR, HF_HOME, PIP_CACHE_DIR'
}

activate_environment() {
  if [[ ! -f "$NOVA_VENV/bin/activate" ]]; then
    echo 'Environment missing; run init first.' >&2
    exit 1
  fi
  source "$NOVA_VENV/bin/activate"
}

case "${1:-help}" in
  init)
    mkdir -p "$NOVA_ROOT"/{inputs,outputs,reports,cache/pip,cache/huggingface}
    if [[ ! -f "$NOVA_VENV/bin/activate" ]]; then
      python3 -m venv --system-site-packages "$NOVA_VENV"
    fi
    activate_environment
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
    echo "Environment: $NOVA_VENV"
    ;;
  check)
    activate_environment
    python "$REPO_DIR/scripts/cloud_environment.py" --output "$NOVA_ROOT/reports" --root "$NOVA_ROOT"
    ;;
  install-deps|install-diffsynth|install-student-deps)
    activate_environment
    mkdir -p "$NOVA_ROOT/reports"
    constraints="$(mktemp "$NOVA_ROOT/reports/torch-constraints.XXXXXX")"
    # Constrain existing platform GPU packages so pip fails instead of replacing them.
    python - "$constraints" <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path
lines = []
for name in ('torch', 'torchvision', 'torchaudio'):
    try:
        lines.append(f'{name}=={metadata.version(name)}')
    except metadata.PackageNotFoundError:
        pass
if not any(line.startswith('torch==') for line in lines):
    raise SystemExit('Platform PyTorch missing; stop before dependency installation.')
Path(sys.argv[1]).write_text('\n'.join(lines) + '\n')
PY
    if [[ "$1" == install-diffsynth ]]; then
      revision="$(tr -d '\n\r' < "$REPO_DIR/configs/diffsynth-revision.txt")"
      if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
        echo 'Invalid pinned DiffSynth revision' >&2
        exit 1
      fi
      python -m pip install -c "$constraints" -r "$REPO_DIR/requirements/diffsynth.txt"
      # A commit-pinned HTTPS archive avoids Git clone/fetch TLS failures.
      python -m pip install --no-deps --retries 5 --timeout 120 \
        "diffsynth @ https://codeload.github.com/modelscope/DiffSynth-Studio/zip/$revision"
      python -c 'from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig; print("DiffSynth S2V import OK")'
    elif [[ "$1" == install-student-deps ]]; then
      python -m pip install -c "$constraints" -r "$REPO_DIR/requirements/student.txt"
    else
      python -m pip install -c "$constraints" -r "$REPO_DIR/requirements/cloud.txt"
    fi
    python "$REPO_DIR/scripts/cloud_dependencies.py"
    python -m pip freeze > "$NOVA_ROOT/reports/environment-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
    ;;
  infer)
    activate_environment
    export DIFFSYNTH_SKIP_DOWNLOAD=True
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export PYTHONUNBUFFERED=1
    mkdir -p "$NOVA_ROOT/reports"
    log="$NOVA_ROOT/reports/infer-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
    python "$REPO_DIR/scripts/infer_s2v.py" --root "$NOVA_ROOT" \
      --model-dir "$NOVA_MODEL_DIR" "${@:2}" 2>&1 | tee "$log"
    ;;
  student-prepare|student-train|student-mouth-train|student-compare|student-export|student-infer|student-mobile-export|student-mobile-fixture|student-mobile-diagnose)
    activate_environment
    script="${1//-/_}.py"
    python "$REPO_DIR/scripts/$script" "${@:2}"
    ;;
  model-check)
    activate_environment
    python "$REPO_DIR/scripts/cloud_model.py" check --model-dir "$NOVA_MODEL_DIR"
    ;;
  model-repair)
    activate_environment
    python "$REPO_DIR/scripts/cloud_model.py" repair \
      --source "$NOVA_SHARED_MODEL" --target "$NOVA_ROOT/models/Wan2.2-S2V-14B"
    ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
