#!/usr/bin/env python3
import json
from pathlib import Path

base_dir = Path("models/base")
for f in sorted(base_dir.glob("*.weights.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    analyzer = d.get("analyzer", "MISSING")
    ngram = d.get("ngram_range", "MISSING")
    nw = len(d.get("weights", {}))
    samples = list(d.get("weights", {}).keys())[:3]
    print(f"{f.name}: analyzer={analyzer!r}, ngram_range={ngram}, weights={nw}, samples={samples}")
