from transformers import EarlyStoppingCallback, TrainerCallback


class LossLoggerCallback(TrainerCallback):
    """Prints train/eval loss as training runs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            print(f"[step {state.global_step}] train_loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            print(f"[step {state.global_step}] eval_loss={logs['eval_loss']:.4f}")


def get_callbacks(patience: int = 3):
    # Early stopping requires load_best_model_at_end=True and matching eval/save steps
    return [EarlyStoppingCallback(early_stopping_patience=patience), LossLoggerCallback()]