"""Versioned lightweight registry; no unverified Elo calculations."""
from pathlib import Path
import json
import hashlib

def register(path, checkpoint_path, label, tags=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    src=Path(checkpoint_path)
    entry={'label':label,'path':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
           'tags':list(tags or []),'rating':None}
    pool=json.loads(p.read_text()) if p.exists() else []
    if any(x['sha256']==entry['sha256'] for x in pool): return pool
    pool.append(entry)
    p.write_text(json.dumps(pool,indent=2))
    return pool
