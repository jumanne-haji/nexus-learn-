import click
import numpy as np
from . import __version__
from .experiment import AutonomousExperiment
from .generator import generate
from .model import NumpyRegressor


@click.group()
def main():
    pass


@main.command()
def version():
    click.echo(f"nexus-learn {__version__}")
    click.echo("backend=numpy (torch optional, never imported)")


@main.command()
def benchmark():
    m = NumpyRegressor(seed=1)
    train = generate(1200, 555, "train")
    hidden = generate(200, 777, "hidden")
    for _ in range(6):
        for i in range(0, len(train), 32):
            b = train[i:i + 32]
            m.train_batch([x.text for x in b], [x.answer for x in b], lr=0.1)
    pred = m.predict([x.text for x in hidden])
    y = np.array([x.answer for x in hidden])
    acc = np.mean(np.abs(pred - y) <= 0.25)
    click.echo(f"Numerical answer accuracy: {acc:.1%}")


@main.command()
@click.option("--cycles", default=3, type=int)
def autonomous(cycles):
    r = AutonomousExperiment().run(cycles)
    click.echo("NEXUS-LEARN autonomous numerical learner")
    click.echo(f"Initial hidden accuracy: {r['initial_hidden']:.1%}")
    click.echo(f"Final hidden accuracy:   {r['final_hidden']:.1%}")
    click.echo(f"Best hidden accuracy:    {r['best_hidden']:.1%}")
    click.echo(f"Training examples:       {r['training_examples']}")
    click.echo(f"Learning improvement:    {'POSITIVE' if r['final_hidden'] > r['initial_hidden'] else 'NONE'}")


if __name__ == "__main__":
    main()
