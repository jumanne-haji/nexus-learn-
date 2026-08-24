from copy import deepcopy
from .generator import generate
from .model import NumpyRegressor
from .online import validate
from .verifier import exact

class OnlineLearningExperiment:
    def __init__(self,seed=17,model=None):
        self.model=model or NumpyRegressor(seed=seed)

    @staticmethod
    def evaluate(model,problems):
        pred=model.predict([p.text for p in problems])
        return sum(exact(p,y) for p,y in zip(problems,pred))/len(problems)

    def learn(self,experiences,epochs=6,lr=.1):
        trusted=validate(experiences)
        holdout=generate(200,31415,"holdout")
        before=self.evaluate(self.model,holdout)

        candidate=deepcopy(self.model)

        for _ in range(epochs):
            for i in range(0,len(trusted),32):
                b=trusted[i:i+32]
                if b:
                    candidate.train_batch(
                        [x.text for x in b],
                        [x.answer for x in b],
                        lr=lr)

        after=self.evaluate(candidate,holdout)
        promoted=after>before

        if promoted:
            self.model=candidate

        return {
            "trusted": len(trusted),
            "trusted_examples": len(trusted),
            "holdout_accuracy_before": before,
            "holdout_accuracy_after": after,
            "before": before,
            "after": after,
            "promoted": promoted,
            "improvement": after - before
        }
