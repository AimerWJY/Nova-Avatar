#!/usr/bin/env bash
# Run with bash; no executable bit or current-directory assumption required.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NOVA_ROOT="${NOVA_ROOT:-/cloud/nova-avatar}"
NOVA_SHARED_MODEL="${NOVA_SHARED_MODEL:-/model/HuggingFace/Wan-AI/Wan2.2-S2V-14B}"
NOVA_VENV="${NOVA_VENV:-$NOVA_ROOT/.venv}"
export HF_HOME="${HF_HOME:-$NOVA_ROOT/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$NOVA_ROOT/cache/pip}"

usage() {
  echo 'Usage: bash scripts/cloud.sh {init|check|install-deps|model-check|model-repair}'
  echo 'Overrides: NOVA_ROOT, NOVA_VENV, NOVA_SHARED_MODEL, HF_HOME, PIP_CACHE_DIR'
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
    mkdir -p "$NOVA_ROOT"/{models,inputs,outputs,reports,cache/pip,cache/huggingface}
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
  install-deps)
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
    python -m pip install -c "$constraints" -r "$REPO_DIR/requirements/cloud.txt"
    python "$REPO_DIR/scripts/cloud_dependencies.py"
    python -m pip freeze > "$NOVA_ROOT/reports/environment-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
    ;;
  model-check|model-repair)
    activate_environment
    python "$REPO_DIR/scripts/cloud_model.py" "${1#model-}" \
      --source "$NOVA_SHARED_MODEL" --target "$NOVA_ROOT/models/Wan2.2-S2V-14B"
    ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
