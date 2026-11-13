# Run AFTER training. Scoring only: dev is never used for training or cleaning.
import json, re, subprocess, sys, glob, os
from pathlib import Path
L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
norm = lambda r: re.sub(r"\W+", " ", r["messages"][1]["content"].split("\n\n", 1)[-1]).strip().lower()
train_bodies = {norm(r) for r in L(DATA / "train.jsonl")}
dev = L(DATA / "dev.jsonl")
keep = {r["id"] for r in dev if norm(r) not in train_bodies}
print("dev rows without train text overlap:", len(keep), "of", len(dev))
with open("dev_noleak.jsonl", "w", encoding="utf-8") as f:
    for r in dev:
        if r["id"] in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
for p in sorted(glob.glob("outputs/*_dev.jsonl")):
    sub = [r for r in L(p) if r["id"] in keep]
    q = "tmp_" + os.path.basename(p)
    with open(q, "w", encoding="utf-8") as f:
        for r in sub: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = subprocess.run([sys.executable, "score.py", "--gold", "dev_noleak.jsonl", "--pred", q],
                         capture_output=True, text=True).stdout
    print("====", os.path.basename(p)); print(out)
