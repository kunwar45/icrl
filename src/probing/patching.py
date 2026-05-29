"""
Activation patching for causal localization.

Protocol:
1. Run model on unsafe trajectory τ_unsafe; record residual stream at layer ℓ.
2. Run model on paired safe trajectory τ_safe; record activations at same layer.
3. Re-run τ_unsafe with residual stream at ℓ replaced by τ_safe activations.
4. Score patched trajectory with C_θ.
5. ΔC = C_θ(patched) − C_θ(original) is the causal effect at ℓ.
"""
import torch
from transformers import PreTrainedModel
from src.constraint.encoder import TrajectoryEncoder
from src.data.trajectory import Trajectory


def patch_residual_stream(
    model: PreTrainedModel,
    unsafe_trajectory: Trajectory,
    safe_trajectory: Trajectory,
    layer_idx: int,
    tokenizer,
) -> torch.Tensor:
    safe_activation = {}

    def capture_hook(module, input, output):
        safe_activation["hidden"] = output[0].detach().clone()

    def patch_hook(module, input, output):
        if "hidden" in safe_activation:
            patched = list(output)
            patched[0] = safe_activation["hidden"]
            return tuple(patched)
        return output

    hook_layer = model.model.layers[layer_idx]

    safe_text = safe_trajectory.to_text()
    safe_inputs = tokenizer(safe_text, return_tensors="pt", truncation=True, max_length=2048)
    safe_inputs = {k: v.to(next(model.parameters()).device) for k, v in safe_inputs.items()}

    with torch.no_grad():
        capture_handle = hook_layer.register_forward_hook(capture_hook)
        model(**safe_inputs, output_hidden_states=True)
        capture_handle.remove()

    unsafe_text = unsafe_trajectory.to_text()
    unsafe_inputs = tokenizer(unsafe_text, return_tensors="pt", truncation=True, max_length=2048)
    unsafe_inputs = {k: v.to(next(model.parameters()).device) for k, v in unsafe_inputs.items()}

    with torch.no_grad():
        patch_handle = hook_layer.register_forward_hook(patch_hook)
        patched_output = model(**unsafe_inputs, output_hidden_states=True)
        patch_handle.remove()

    return patched_output


def compute_patching_heatmap(
    model: PreTrainedModel,
    constraint_model: TrajectoryEncoder,
    trajectory_pairs: list[tuple[Trajectory, Trajectory]],
    tokenizer,
    n_layers: int,
) -> dict:
    layer_deltas: dict[int, list[float]] = {l: [] for l in range(n_layers)}

    for unsafe_traj, safe_traj in trajectory_pairs:
        baseline_score = constraint_model([unsafe_traj.to_text()]).item()

        for layer_idx in range(n_layers):
            patched_output = patch_residual_stream(
                model, unsafe_traj, safe_traj, layer_idx, tokenizer
            )
            patched_hidden = patched_output.hidden_states[-1][0].mean(0)
            patched_score = constraint_model.head(
                patched_hidden.float().unsqueeze(0)
            ).item()

            delta = patched_score - baseline_score
            layer_deltas[layer_idx].append(delta)

    mean_delta = {l: float(sum(v) / len(v)) for l, v in layer_deltas.items() if v}
    return {
        "mean_delta_by_layer": mean_delta,
        "raw_deltas": layer_deltas,
    }
