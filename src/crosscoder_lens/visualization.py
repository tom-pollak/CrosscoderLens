import torch as t
from tiny_dashboard.feature_centric_dashboards import OfflineFeatureCentricDashboard
from transformers import PreTrainedTokenizer

from .model import CrossCoder


class CrossCoderVisualizer:
    """Visualizer for CrossCoder features using tiny-activation-dashboard"""

    def __init__(
        self,
        model: CrossCoder,
        tokenizer: PreTrainedTokenizer,
        n_examples: int = 10,
        top_k: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_examples = n_examples
        self.top_k = top_k

    def get_max_activation_examples(
        self, texts: list[str]
    ) -> dict[int, list[tuple[float, list[str], list[float]]]]:
        """Get examples of maximum activations for each feature"""
        max_activation_examples = {}

        # Tokenize all texts
        tokenized = [self.tokenizer.tokenize(text) for text in texts]

        # Get activations for all texts
        with t.no_grad():
            for feature_idx in range(self.model.config.dict_size):
                examples = []

                for text_idx, (text, tokens) in enumerate(
                    zip(texts, tokenized, strict=False)
                ):
                    print(f"Processing text {text_idx} of {len(texts)}")
                    # Get activations
                    input_ids = self.tokenizer.encode(text, return_tensors="pt").to(
                        self.model.config.device
                    )
                    acts = self.model.get_feature_activations(
                        self.model.encode(input_ids)
                    )[0, :, feature_idx]

                    # Get top-k activating positions
                    top_k_values, top_k_indices = t.topk(
                        acts, min(self.top_k, len(acts))
                    )

                    for val, idx in zip(top_k_values, top_k_indices):
                        if val > 0:  # Only include non-zero activations
                            # Create activation pattern
                            act_pattern = t.zeros_like(acts)
                            act_pattern[idx] = val

                            examples.append((val.item(), tokens, act_pattern.tolist()))

                # Sort by activation value and take top n_examples
                examples.sort(key=lambda x: x[0], reverse=True)
                max_activation_examples[feature_idx] = examples[: self.n_examples]

        return max_activation_examples

    def create_dashboard(
        self,
        texts: list[str],
        export_path: str | None = None,
        max_features: int | None = None,
    ) -> OfflineFeatureCentricDashboard:
        """Create and optionally export a dashboard for the model"""
        max_activation_examples = self.get_max_activation_examples(texts)

        # Create dashboard
        dashboard = OfflineFeatureCentricDashboard(
            max_activation_examples, self.tokenizer
        )

        # Export if path provided
        if export_path:
            dashboard.export_to_html(
                export_path, max_features or len(max_activation_examples)
            )

        return dashboard
