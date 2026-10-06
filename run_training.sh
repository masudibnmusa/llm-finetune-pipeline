#!/usr/bin/env bash
set -euo pipefail

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

# 5) Evaluate (set BASE_MODEL to the same model as in model_config.yaml)
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
python evaluation/benchmark_runner.py --base "$BASE_MODEL" --adapter data/checkpoints/final
python evaluation/comparison_report.py

echo "Done. See reports/comparison_report.md"