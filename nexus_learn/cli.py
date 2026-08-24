import click
from pathlib import Path
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


@main.command("online-learn")
@click.argument("source")
@click.option("--epochs", default=4, type=int)
def online_learn(source, epochs):
    from .online import load_source
    from .online_experiment import OnlineLearningExperiment

    click.echo("NEXUS-LEARN online experience learner")
    click.echo(f"Source: {source}")

    experiences = load_source(source)

    result = OnlineLearningExperiment().learn(
        experiences,
        epochs=epochs,
    )

    click.echo(
        f"Trusted experiences: "
        f"{result['trusted_examples']}"
    )

    click.echo(
        f"Holdout before: "
        f"{result['holdout_accuracy_before']:.1%}"
    )

    click.echo(
        f"Holdout after:  "
        f"{result['holdout_accuracy_after']:.1%}"
    )

    click.echo(
        "Candidate: "
        + ("PROMOTED" if result["promoted"] else "REJECTED")
    )

    click.echo(
        f"Improvement: "
        f"{result['improvement']:+.1%}"
    )



@main.command("generate-online")
@click.option("--count",default=1200,type=int)
@click.option("--seed",default=424242,type=int)
def generate_online(count,seed):
    import json
    from dataclasses import asdict
    from .generator import generate_rich

    items=generate_rich(count,seed,"online")
    Path("examples").mkdir(exist_ok=True)

    with open("examples/online_generated.json","w",encoding="utf-8") as f:
        json.dump(
            {"experiences":[asdict(x) for x in items]},
            f,
            indent=2
        )

    click.echo(f"Generated: {len(items)}")
    click.echo(
        f"Unique experiences: "
        f"{len({(x.text,x.answer) for x in items})}"
    )

@main.command("online-sync")
@click.argument("source")
@click.option("--epochs", default=6, type=int)
@click.option("--lr", default=0.1, type=float)
@click.option("--state-dir", default="state", type=click.Path())
def online_sync(source, epochs, lr, state_dir):
    """
    Fetch mathematical experiences from SOURCE and attempt online learning.

    SOURCE may be a local JSON file or an HTTP/HTTPS JSON endpoint.
    Remote content is treated as data only.
    """

    from .online_sync import OnlineSync

    click.echo("NEXUS-LEARN online experience acquisition")
    click.echo(f"Source: {source}")

    sync = OnlineSync(state_dir=state_dir)

    result = sync.run(
        source,
        epochs=epochs,
        lr=lr,
    )

    click.echo(
        f"Trusted experiences: "
        f"{result['trusted_examples']}"
    )

    click.echo(
        f"Holdout before: "
        f"{result['holdout_accuracy_before']:.1%}"
    )

    click.echo(
        f"Holdout after:  "
        f"{result['holdout_accuracy_after']:.1%}"
    )

    click.echo(
        "Candidate: "
        + ("PROMOTED" if result["promoted"] else "REJECTED")
    )

    click.echo(
        f"Improvement: "
        f"{result['improvement']:+.1%}"
    )

    click.echo(
        "State: "
        + ("SAVED" if result["promoted"] else "UNCHANGED")
    )


if __name__ == "__main__":
    main()

