from __future__ import annotations
import hashlib,json,platform
from datetime import datetime,timezone
import numpy as np,pandas as pd,sklearn
from .schemas import RESULTS,PREDICTIONS,HYPERPARAMETERS,EXPERIMENTS

def empty_csv(path,columns): pd.DataFrame(columns=columns).to_csv(path,index=False,encoding="utf-8")

def write_predictions(path,frame):
    from .schemas import PREDICTIONS
    frame.reindex(columns=PREDICTIONS).to_csv(path,index=False,encoding="utf-8")

def environment_metadata():
    return {"python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__}

def pipeline_fingerprint(config:dict)->str:
    raw=json.dumps(config,sort_keys=True,default=str).encode(); return hashlib.sha256(raw).hexdigest()

def utc_now(): return datetime.now(timezone.utc).isoformat()
