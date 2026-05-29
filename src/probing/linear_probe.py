"""
Layer-wise linear probing.

For each layer ℓ, train a linear probe to predict C_θ(τ1:t)
from the residual stream hidden state.
"""
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from transformers import PreTrainedModel, PreTrainedTokenizer
from omegaconf import DictConfig
from src.data.trajectory import Trajectory


class LayerWiseProbes:
    def __init__(
        self,
        cfg: DictConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.probes: dict[int, Ridge] = {}

    @torch.no_grad()
    def _extract_hidden_states(self, trajectory: Trajectory) -> dict[int, np.ndarray]:
        text = trajectory.to_text()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        self.model.eval()
        outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = {}
        for layer_idx, hs in enumerate(outputs.hidden_states[1:]):
            hidden_states[layer_idx] = hs[0].mean(0).cpu().float().numpy()

        return hidden_states

    def fit(self, trajectories: list[Trajectory], constraint_scores: list[float]):
        print(f"Extracting hidden states for {len(trajectories)} trajectories...")
        all_hidden = [self._extract_hidden_states(t) for t in trajectories]

        for layer_idx in range(self.n_layers):
            X = np.stack([h[layer_idx] for h in all_hidden])
            y = np.array(constraint_scores)
            probe = Ridge(alpha=1.0)
            probe.fit(X, y)
            self.probes[layer_idx] = probe
            if layer_idx % 8 == 0:
                print(f"  Layer {layer_idx} probe fitted")

    def evaluate(
        self, trajectories: list[Trajectory], constraint_scores: list[float]
    ) -> dict:
        all_hidden = [self._extract_hidden_states(t) for t in trajectories]
        y = np.array(constraint_scores)

        results = {}
        for layer_idx, probe in self.probes.items():
            X = np.stack([h[layer_idx] for h in all_hidden])
            y_pred = probe.predict(X)
            results[layer_idx] = float(r2_score(y, y_pred))

        return {
            "probe_r2_by_layer": results,
            "n_trajectories": len(trajectories),
        }
