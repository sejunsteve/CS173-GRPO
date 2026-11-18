#!/bin/bash
# Run only INSIDE an existing single-GPU Scholar allocation (e.g. OnDemand Jupyter).
set -euo pipefail
cd "$(dirname "$0")"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "No Slurm allocation. Submit train.sbatch instead of using a login node." >&2
    exit 2
fi
if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi
module load conda/2024.09-py312
unset PYTHONPATH
python --version
nvidia-smi
python -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), sys.version'
if [[ ! -x .venv-scholar/bin/python ]]; then
    python -m venv .venv-scholar
fi
source .venv-scholar/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python -m pip install --no-cache-dir 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-cache-dir -r requirements.txt
python -m pip check
mkdir -p output reports
RUN_TAG="${RUN_TAG:-run1}"
RUN_DIR="output/$RUN_TAG"
REPORT_DIR="reports/$RUN_TAG"
if [[ -e "$RUN_DIR" ]]; then
    echo "Run output already exists: $RUN_DIR. Set a new RUN_TAG." >&2
    exit 2
fi
python -m pip freeze > scholar-environment.txt
python -m unittest discover -s . -p 'test_*.py' -v
if [[ ! -d data ]]; then
    python dataset.py --seed 173
fi
python dataset.py --validate-only
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory)'
python evaluate.py --output "$RUN_DIR/baseline"
python training.py --steps "${GRPO_STEPS:-130}" --output "$RUN_DIR/grpo"
python evaluate.py --adapter "$RUN_DIR/grpo/adapter" --output "$RUN_DIR/after"
python report.py --run "$RUN_DIR/grpo" --baseline "$RUN_DIR/baseline/metrics.json" --after "$RUN_DIR/after/metrics.json" --output "$REPORT_DIR"
tar -czf output/phase2-results.tar.gz "$RUN_DIR" reports scholar-environment.txt
echo 'PHASE2_FINISHED: download output/phase2-results.tar.gz through Scholar Files.'
