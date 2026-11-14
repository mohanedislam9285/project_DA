from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def numeric_preprocessor():
    """Fit only on a training fold; median imputation and scaling for neural models."""
    return Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("scaler",StandardScaler())])

class SequencePreprocessor:
    """Fold-local channel imputation/scaling for (samples, time, features) tensors."""
    def __init__(self,add_indicator=True): self.add_indicator=bool(add_indicator); self.imputer=SimpleImputer(strategy="median",add_indicator=self.add_indicator,keep_empty_features=True); self.scaler=StandardScaler(); self.shape_=None
    def fit(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=3 or not len(x): raise ValueError("Expected non-empty 3D training sequences")
        flat=x.reshape(-1,x.shape[-1]); imputed=self.imputer.fit_transform(flat); self.scaler.fit(imputed); self.shape_=(x.shape[1],imputed.shape[1]); return self
    def transform(self,x):
        if self.shape_ is None: raise RuntimeError("SequencePreprocessor must be fitted on training sequences first")
        x=np.asarray(x,dtype=float); flat=x.reshape(-1,x.shape[-1]); result=self.scaler.transform(self.imputer.transform(flat)); return result.reshape(x.shape[0],self.shape_[0],self.shape_[1]).astype(np.float32)
    def fit_transform(self,x): return self.fit(x).transform(x)
    def feature_names_out(self,feature_names):
        indicator=getattr(self.imputer,"indicator_",None)
        indices=list(getattr(indicator,"features_",[]))
        return list(feature_names)+[f"{feature_names[i]}__missing" for i in indices]
