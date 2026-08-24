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
