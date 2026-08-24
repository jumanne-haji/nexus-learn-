import numpy as np
from nexus_learn.model import NumpyRegressor
from nexus_learn.generator import generate
def test_model_changes():
    m=NumpyRegressor(seed=1); p=generate(16,1)
    before=m.W1.copy()
    for _ in range(5): m.train_batch([x.text for x in p],[x.answer for x in p])
    assert (before!=m.W1).any()
def test_numeric_answers_not_classes():
    m=NumpyRegressor(seed=1); p=generate(8,2)
    for _ in range(20): m.train_batch([x.text for x in p],[x.answer for x in p])
    pred=m.predict([x.text for x in p])
    assert pred.shape==(8,)
def test_generation():
    assert len(generate(10))==10

def test_prediction_is_numeric_and_bounded():
    m=NumpyRegressor(seed=3)
    p=generate(12,4)
    pred=m.predict([x.text for x in p])
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()

def test_learning_improves_hidden_set():
    from nexus_learn.experiment import AutonomousExperiment
    r=AutonomousExperiment(seed=7).run(cycles=3)
    assert r["final_hidden"] > r["initial_hidden"]
    assert r["training_examples"] == 1200
