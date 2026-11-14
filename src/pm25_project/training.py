"""Training orchestration primitives; final test evaluation is deliberately separate."""
from __future__ import annotations
import time
import numpy as np
from .metrics import regression_metrics

def fit_estimator_fold(estimator,x_train,y_train,x_valid,y_valid):
    started=time.perf_counter(); estimator.fit(x_train,y_train); fit_time=time.perf_counter()-started
    started=time.perf_counter(); prediction=np.asarray(estimator.predict(x_valid)).reshape(-1); inference_time=time.perf_counter()-started
    return {"metrics":regression_metrics(y_valid,prediction),"prediction":prediction,"fit_time_sec":fit_time,"inference_time_sec":inference_time,"estimator":estimator}
