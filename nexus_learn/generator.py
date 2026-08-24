from dataclasses import dataclass
import random, math

@dataclass
class Problem:
    text:str
    answer:float
    split:str="train"

def generate(n=100, seed=0, split="train"):
    r=random.Random(seed)
    out=[]
    for _ in range(n):
        k=r.choice([2,3,4,5,10])
        exp=r.randint(1,5)
        base=k**exp
        typ=r.randrange(4)
        if typ==0:
            text=f"log_{k}({base}) = ?"; ans=exp
        elif typ==1:
            text=f"log_{k}(x) = {exp}; x = ?"; ans=base
        elif typ==2:
            text=f"{k}^? = {base}"; ans=exp
        else:
            text=f"log_{k}({base}) = {exp}"; ans=exp
        out.append(Problem(text,float(ans),split))
    return out


def generate_rich(n=1000, seed=0, split="online"):
    """Generate deterministic, diverse, bounded logarithm experiences."""

    r = random.Random(seed)

    # Keep numerical answers bounded so online validation accepts them.
    bases = list(range(2, 101))
    exps = list(range(1, 6))

    # Multiple mathematically equivalent surface forms provide
    # structural/lexical diversity without changing the answer.
    styles = [
        lambda k,b,e: f"log_{k}({b}) = ?",
        lambda k,b,e: f"log_{k}(x) = {e}; x = ?",
        lambda k,b,e: f"{k}^? = {b}",
        lambda k,b,e: f"log_{k}({b}) = {e}",
        lambda k,b,e: f"Find y: log_{k}({b}) = y",
        lambda k,b,e: f"Solve: log base {k} of {b}",
        lambda k,b,e: f"logarithm base {k}: {b} -> ?",
        lambda k,b,e: f"{b} = {k}^x; x = ?",
    ]

    candidates = []

    for k in bases:
        for e in exps:
            b = k ** e

            if b > 100000:
                continue

            for style in styles:
                text = style(k, b, e)

                # All forms above ask for the exponent except the
                # explicit inverse-log form.
                if "log_" in text and "x) =" in text:
                    answer = float(b)
                elif f"{b} = {k}^x" in text:
                    answer = float(e)
                else:
                    answer = float(e)

                candidates.append(
                    Problem(text, answer, split)
                )

    if n > len(candidates):
        raise ValueError(
            f"Requested {n} unique experiences, but only "
            f"{len(candidates)} are available"
        )

    r.shuffle(candidates)
    return candidates[:n]
