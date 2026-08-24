import json
from pathlib import Path
from .generator import Problem

class ExperienceStore:
    def __init__(self,path):
        self.path=Path(path)

    def load(self):
        if not self.path.exists():
            return []
        out=[]
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                x=json.loads(line)
                out.append(Problem(str(x["text"]),float(x["answer"]),"stored"))
            except Exception:
                continue
        return out

    def merge(self,items):
        old=self.load()
        seen={(x.text.lower(),round(x.answer,12)) for x in old}
        self.path.parent.mkdir(parents=True,exist_ok=True)
        added=0
        with self.path.open("a",encoding="utf-8") as f:
            for x in items:
                key=(x.text.lower(),round(x.answer,12))
                if key in seen: continue
                seen.add(key)
                f.write(json.dumps({"text":x.text,"answer":x.answer})+"\n")
                added+=1
        return added
