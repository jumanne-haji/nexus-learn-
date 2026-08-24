import json, math
from dataclasses import asdict
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from .generator import Problem

MAX_BYTES=512_000

def fetch(source, timeout=15):
    u=urlparse(source)
    if u.scheme not in ("http","https"):
        raise ValueError("Only HTTP/HTTPS sources are allowed")
    req=Request(source,headers={"User-Agent":"NEXUS-Learn/1.5"})
    with urlopen(req,timeout=timeout) as r:
        data=r.read(MAX_BYTES+1)
    if len(data)>MAX_BYTES:
        raise ValueError("Remote source exceeds size limit")
    return data.decode("utf-8","replace")

def parse_json(raw):
    data=json.loads(raw)
    rows=data.get("experiences",[]) if isinstance(data,dict) else data
    if not isinstance(rows,list):
        return []
    out=[]
    for row in rows:
        if not isinstance(row,dict):
            continue
        if "text" not in row or "answer" not in row:
            continue
        try:
            out.append(Problem(str(row["text"]),float(row["answer"]),"online"))
        except (TypeError,ValueError):
            pass
    return out

def validate(items):
    out=[]; seen=set()
    for p in items:
        text=" ".join(str(p.text).split())
        try: answer=float(p.answer)
        except (TypeError,ValueError): continue
        if not text or not math.isfinite(answer): continue
        if abs(answer)>100000: continue
        # Accept the complete bounded logarithm experience vocabulary.
        normalized = text.lower()
        valid_math_marker = (
            "log_" in normalized
            or "^?" in normalized
            or "log base " in normalized
            or "logarithm base " in normalized
            or " = " in normalized and "^x" in normalized
        )
        if not valid_math_marker:
            continue
        key=(text.lower(),round(answer,12))
        if key in seen: continue
        seen.add(key)
        out.append(Problem(text,answer,"online_verified"))
    return out

def load_source(source):
    raw=fetch(source) if source.startswith(("http://","https://")) \
        else open(source,encoding="utf-8").read()
    return validate(parse_json(raw))

def save(items,path):
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"experiences":[asdict(x) for x in items]},f,indent=2)
