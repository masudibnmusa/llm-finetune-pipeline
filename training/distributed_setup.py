"""Helpers for multi-GPU training with Accelerate + DeepSpeed ZeRO-2 or FSDP.
Run `python training/distributed_setup.py` to write config files, then launch with
the printed commands."""
import json
from pathlib import Path

DEEPSPEED_ZERO2 = {
    "bf16": {"enabled": "auto"},
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}

FSDP_ACCELERATE = """compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_machines: 1
num_processes: 2          # number of GPUs
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_offload_params: false
"""


def write_configs(out_dir: str = "training/config"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/ds_zero2.json", "w") as f:
        json.dump(DEEPSPEED_ZERO2, f, indent=2)
    with open(f"{out_dir}/accelerate_fsdp.yaml", "w") as f:
        f.write(FSDP_ACCELERATE)
    print("Wrote ds_zero2.json and accelerate_fsdp.yaml")
    print("\nLaunch examples:")
    print("  accelerate launch --num_processes 2 training/train.py")
    print("  accelerate launch --config_file training/config/accelerate_fsdp.yaml training/train.py")
    print("\nFor DeepSpeed, add `deepspeed: training/config/ds_zero2.json` under `args:` "
          "in training_args.yaml and `pip install deepspeed`.")


if __name__ == "__main__":
    write_configs()