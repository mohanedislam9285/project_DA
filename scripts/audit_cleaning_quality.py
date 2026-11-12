"""Development-only data quality audit: flags unusual values, never drops them."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'16'/'cleaning_quality_audit'; OUT.mkdir(parents=True,exist_ok=True)
lock=json.loads((ROOT/'results'/'16'/'future_test.json').read_text(encoding='utf-8'))
df=pd.read_csv(ROOT/'data'/'processed'/'cleaned_daily.csv',parse_dates=['timestamp'])
dev=df.iloc[np.asarray(lock['development_indices'],dtype=int)].copy()
dev['timestamp']=pd.to_datetime(dev.timestamp,utc=True)
if dev.timestamp.max()>pd.to_datetime(lock['development_end'],utc=True):
    raise RuntimeError('Development audit rows cross the locked cutoff')
# Development-only target validity, key integrity, coordinate range/stability.
duplicates=int(dev.duplicated(['station_id','timestamp']).sum())
invalid=int((~np.isfinite(dev.pm25)|(dev.pm25<=0)|(dev.pm25>=500)).sum())
coord_invalid=int((~np.isfinite(dev.latitude)|~np.isfinite(dev.longitude)|~dev.latitude.between(-90,90)|~dev.longitude.between(-180,180)).sum())
summary={'scope':'cleaned development partition only','development_rows':int(len(dev)),'stations':int(dev.station_id.nunique()),'date_min':dev.timestamp.min().isoformat(),'date_max':dev.timestamp.max().isoformat(),'duplicate_station_day_keys':duplicates,'nonfinite_or_out_of_rule_pm25':invalid,'invalid_coordinates':coord_invalid,'future_test_values_used':False,'statistical_flags_are_removed':False}
summary['target_distribution']={k:float(v) for k,v in dev.pm25.quantile([.5,.9,.95,.99,1]).items()}
summary['target_counts']={f'gt_{v}':int((dev.pm25>v).sum()) for v in (25,50,100,250)}
feature_cols=['temperature_celsius','pressure_mb','wind_u','wind_v','no2','co','o3','aod']
summary['predictor_missingness']={c:{'missing_n':int(dev[c].isna().sum()),'missing_fraction':float(dev[c].isna().mean()),'finite_n':int(np.isfinite(dev[c]).sum())} for c in feature_cols}
# Station-specific Tukey upper fences (Q3 + 3*IQR) are screening flags only.
rows=[]
for station,g in dev.groupby('station_id',sort=True):
    x=g.pm25.dropna();
    if len(x)<30: continue
    q1,q3=x.quantile([.25,.75]); iqr=q3-q1; fence=float(q3+3*iqr)
    flagged=g[g.pm25>fence]
    for _,r in flagged.iterrows():
        rows.append({'station_id':station,'timestamp':r.timestamp.isoformat(),'pm25':float(r.pm25),'q3_plus_3iqr_fence':fence,'station_n':int(len(x)),'flag':'station-relative statistical high value; review only; retained'})
summary['station_tukey_screen']={'stations_screened_min_n30':int(dev.groupby('station_id').pm25.count().ge(30).sum()),'flagged_rows':len(rows),'policy':'IQR Q3+3*IQR; diagnostic screening only; no automatic deletion'}
summary['cleaning_policy_assessment']={'configured_pm25_rule':'retain 0 < PM2.5 < 500 µg/m³','source_measurements_removed_for_statistical_outlier_only':0,'high_values_are_retained_unless_source_QA_invalidates_them':True,'high_value_validation_note':'Rows above 100 µg/m³ are retained; statistical extremeness alone cannot establish an instrument fault.'}
(OUT/'cleaning_quality_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
pd.DataFrame(rows).to_csv(OUT/'station_relative_high_value_flags.csv',index=False)
pd.DataFrame([{'variable':c,**summary['predictor_missingness'][c]} for c in feature_cols]).to_csv(OUT/'development_predictor_missingness.csv',index=False)
print(json.dumps(summary,indent=2))
