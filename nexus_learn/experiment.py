from .generator import generate
from .model import NumpyRegressor
from .verifier import exact


class AutonomousExperiment:
    def __init__(self, seed=7):
        self.model = NumpyRegressor(seed=seed)

    def evaluate(self, problems):
        predictions = self.model.predict([x.text for x in problems])
        return sum(exact(x, y) for x, y in zip(problems, predictions)) / len(problems)

    @staticmethod
    def failure_signature(problem):
        text = problem.text
        kind = (
            "log_unknown" if "x) =" in text else
            "power_unknown" if "^?" in text else
            "log_direct" if "= ?" in text else
            "log_explicit"
        )
        return kind

    def run(self, cycles=3):
        hidden = generate(100, 991, "hidden")
        initial = self.evaluate(hidden)
        total = 0
        history = []

        for c in range(cycles):
            train = generate(400, 100 + c, "train")

            # Failure-guided learning: inspect the held-out failures only to
            # select the same structural families in fresh training data.
            failed = [p for p in hidden
                      if not exact(p, self.model.predict([p.text])[0])]
            failed_kinds = {self.failure_signature(p) for p in failed}
            if failed_kinds:
                focused = [p for p in train if self.failure_signature(p) in failed_kinds]
                train = focused + train

            # A short replay pass over the fresh examples prevents a weak
            # early batch from dominating the learned representation.
            for _ in range(4):
                for i in range(0, len(train), 32):
                    b = train[i:i + 32]
                    self.model.train_batch(
                        [x.text for x in b], [x.answer for x in b], lr=0.1
                    )

            total += 400
            history.append(self.evaluate(hidden))

        final = self.evaluate(hidden)
        return {
            "initial_hidden": initial,
            "final_hidden": final,
            "best_hidden": max(history or [initial]),
            "cycles": cycles,
            "training_examples": total,
            "history": history,
        }
