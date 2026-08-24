from datetime import datetime, timezone
from pathlib import Path
from .online import load_source
from .online_experiment import OnlineLearningExperiment
from .state import save_model, save_metadata


class OnlineSync:
    """
    Acquire mathematical experiences from a local or remote JSON source.

    Remote content is treated strictly as DATA.
    No downloaded content is imported or executed.
    """

    def __init__(
        self,
        state_dir="state",
        seed=17,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.state_dir / "model.npz"
        self.metadata_path = self.state_dir / "metadata.json"

        from .state import load_model

        self.model = load_model(self.model_path, seed=seed)

    def run(self, source, epochs=6, lr=0.1):
        experiences = load_source(source)

        experiment = OnlineLearningExperiment(
            seed=17,
            model=self.model,
        )

        result = experiment.learn(
            experiences,
            epochs=epochs,
            lr=lr,
        )

        if result["promoted"]:
            self.model = experiment.model

            save_model(
                self.model,
                self.model_path,
            )

            save_metadata(
                self.metadata_path,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "source": source,
                    "trusted_examples": result["trusted_examples"],
                    "holdout_accuracy_before":
                        result["holdout_accuracy_before"],
                    "holdout_accuracy_after":
                        result["holdout_accuracy_after"],
                    "improvement": result["improvement"],
                    "promoted": True,
                },
            )

        return result
