#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root so all relative paths resolve
cd "$(dirname "$0")"

# 1) Collect (pick one or more domains)
python -m data_pipeline.collectors.support_ticket_collector --limit 2000
# python -m data_pipeline.collectors.legal_data_collector --limit 2000
# python -m data_pipeline.collectors.medical_data_collector --limit 2000

# 2) Clean
python -m data_pipeline.cleaning.pii_scrubber
python -m data_pipeline.cleaning.deduplicator
python -m data_pipeline.cleaning.quality_filter

# 3) Format + split
python -m data_pipeline.formatting.instruction_formatter
python -m data_pipeline.formatting.dataset_splitter

# 4) Train
python training/train.py --config_dir training/config

# 5) Evaluate against the same base model used for training (override with BASE_MODEL=...)
BASE_MODEL="${BASE_MODEL:-$(python -c 'import yaml; print(yaml.safe_load(open("training/config/model_config.yaml"))["base_model"])')}"
python evaluation/benchmark_runner.py --base "$BASE_MODEL" --adapter data/checkpoints/final
python evaluation/comparison_report.py

echo "Done. See reports/comparison_report.md"