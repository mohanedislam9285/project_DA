from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .schemas import EXPLORATORY_DATA

def exploratory_table(df: pd.DataFrame, cfg: dict, audit: dict | None = None) -> pd.DataFrame:
    rows=[]; numeric_cols=df.select_dtypes(include="number").columns
    for col in df.columns:
        s=df[col]; vals=pd.to_numeric(s,errors="coerce").dropna(); row={k:np.nan for k in EXPLORATORY_DATA}
        row.update(model_id="dataset_audit",model_group="data",variable=col,family="Target" if col=="pm25" else next((v.get("family") for n,v in cfg["data"].get("predictors",{}).items() if n==col),"metadata"),type="numeric" if col in numeric_cols else "categorical",unit=cfg["data"].get("target_unit") if col=="pm25" else next((v.get("unit") for n,v in cfg["data"].get("predictors",{}).items() if n==col),None),n_total=len(s),n_missing=int(s.isna().sum()),pct_missing=float(s.isna().mean()*100),n_unique=int(s.nunique(dropna=True)),modalities=";".join(map(str,s.dropna().unique()[:30])) if col not in numeric_cols else np.nan,is_leakage_suspect=col in cfg["cleaning"].get("leakage_columns",[]))
        if len(vals):
            q1,med,q3=vals.quantile([.25,.5,.75]); iqr=q3-q1; sd=vals.std(ddof=1)
            row.update(min=float(vals.min()),q1=float(q1),median=float(med),q3=float(q3),max=float(vals.max()),mean=float(vals.mean()),std=float(sd) if np.isfinite(sd) else np.nan,skewness=float(vals.skew()),kurtosis=float(vals.kurt()),iqr=float(iqr),n_out_iqr=int(((vals<q1-1.5*iqr)|(vals>q3+1.5*iqr)).sum()),n_out_z=int(((vals-vals.mean()).abs()>3*sd).sum()) if sd and np.isfinite(sd) else 0)
            row["pct_out_iqr"]=100*row["n_out_iqr"]/len(vals); row["pct_out_z"]=100*row["n_out_z"]/len(vals)
        rows.append(row)
    result=pd.DataFrame(rows,columns=EXPLORATORY_DATA)
    for col in ("corr_partner","notes"):
        result[col]=result[col].astype("object")
    if audit and "cleaning" in audit and "pm25" in df:
        c=audit["cleaning"]["removed_by_reason_mutually_exclusive"]; ix=result.index[result.variable.eq("pm25")]
        if len(ix):
            i=ix[0]; result.loc[i,"n_total"]=audit["cleaning"]["input_rows"]; result.loc[i,"n_sentinel_low"]=c.get("target_le_min",0); result.loc[i,"n_sentinel_high"]=c.get("target_ge_max",0); result.loc[i,"n_removed_total"]=audit["cleaning"]["removed_total"]
            result.loc[i,"notes"]="Raw-row sentinel counts and removals use mutually exclusive cleaning reasons; valid extremes retained."
    predictors=[n for n in cfg["data"].get("predictors",{}) if n in df and n in numeric_cols]
    corr_cols=["pm25"]+predictors
    if len(corr_cols)>1:
        assoc=df[corr_cols].corr(method="spearman")["pm25"].drop("pm25",errors="ignore")
        for i,col in enumerate(result.variable):
            if col in assoc.index: result.loc[i,"target_assoc"]=float(assoc[col]); result.loc[i,"max_abs_corr"]=abs(float(assoc[col])); result.loc[i,"corr_partner"]="pm25"
    return result

def seasonal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Seasonal (DJF/MAM/JJA/SON) PM2.5 levels and exceedance frequencies above 25 and 50 µg/m³."""
    month=df.timestamp.dt.month
    season=pd.Series(np.select([month.isin([12,1,2]),month.isin([3,4,5]),month.isin([6,7,8]),month.isin([9,10,11])],["DJF","MAM","JJA","SON"],default=""),index=df.index)
    tagged=df.assign(season=season)
    levels=tagged.groupby("season").pm25.agg(n_days="count",mean="mean",median="median",std="std",q95=lambda s:float(s.quantile(.95)),max="max").reset_index()
    rates=tagged.assign(above25=tagged.pm25>25,above50=tagged.pm25>50).groupby("season")[["above25","above50"]].mean().mul(100).reset_index().rename(columns={"above25":"pct_days_above_25","above50":"pct_days_above_50"})
    return levels.merge(rates,on="season").sort_values("season").reset_index(drop=True)

def run_eda(df: pd.DataFrame, cfg: dict, audit: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(cfg["_config_dir"]); out=root/cfg["paths"]["results_root"]/cfg["project"]["group_id"]; out.mkdir(parents=True,exist_ok=True)
    audit_path=root/"data/processed/audit.json"
    if audit is None and audit_path.exists(): audit=json.loads(audit_path.read_text(encoding="utf-8"))
    figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    exploratory_table(df,cfg,audit).to_csv(out/"Exploratory_Data.csv",index=False,encoding="utf-8")
    fig,ax=plt.subplots(figsize=(8,4)); df.pm25.hist(ax=ax,bins=40); ax.set(xlabel="PM2.5 (µg/m³)",ylabel="Days",title="Daily PM2.5 distribution"); fig.tight_layout(); fig.savefig(figdir/"pm25_distribution.png",dpi=160); plt.close(fig)
    stations=df.station_id.dropna().unique(); fig,ax=plt.subplots(figsize=(10,4))
    for station,g in df.groupby("station_id"):
        if len(stations)<=12: ax.plot(g.timestamp,g.pm25,alpha=.45,lw=.7,label=str(station))
    ax.set(xlabel="Date (UTC)",ylabel="PM2.5 (µg/m³)",title="Daily PM2.5 by station");
    if len(stations)<=12: ax.legend(fontsize=7,ncol=3)
    fig.tight_layout(); fig.savefig(figdir/"pm25_over_time.png",dpi=160); plt.close(fig)
    miss=df.isna().mean().sort_values(ascending=False); fig,ax=plt.subplots(figsize=(7,5)); miss.plot.bar(ax=ax); ax.set(ylabel="Fraction missing",title="Missingness after target cleaning"); fig.tight_layout(); fig.savefig(figdir/"missingness.png",dpi=160); plt.close(fig)
    value_cols=["pm25"]+[n for n in cfg["data"].get("predictors",{}) if n in df and n in df.select_dtypes(include="number")]
    for col in value_cols:
        fig,axes=plt.subplots(1,2,figsize=(9,3.5)); df[col].dropna().hist(ax=axes[0],bins=35); axes[0].set(xlabel=col,ylabel="Rows",title=f"{col} histogram"); axes[1].boxplot(df[col].dropna()); axes[1].set(ylabel=col,title=f"{col} boxplot"); fig.tight_layout(); fig.savefig(figdir/f"distribution_{col}.png",dpi=150); plt.close(fig)
    station_summary=df.groupby("station_id").pm25.agg(n_days="count",median="median",mean="mean",std="std").reset_index(); rate=df.assign(above25=df.pm25>25,above50=df.pm25>50).groupby("station_id")[["above25","above50"]].mean().mul(100).reset_index(); station_summary=station_summary.merge(rate,on="station_id").rename(columns={"above25":"pct_above_25","above50":"pct_above_50"}); station_summary.to_csv(out/"station_summary.csv",index=False)
    mdf=df.assign(month=df.timestamp.dt.month); monthly=mdf.groupby("month").pm25.agg(n_days="count",median="median",mean="mean",std="std").reset_index(); rates=mdf.assign(above25=mdf.pm25>25,above50=mdf.pm25>50).groupby("month")[["above25","above50"]].mean().mul(100).reset_index(); monthly=monthly.merge(rates,on="month").rename(columns={"above25":"pct_above_25","above50":"pct_above_50"}); monthly.to_csv(out/"monthly_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,4)); monthly.set_index("month")[["median","mean"]].plot(marker="o",ax=ax); ax.set(xlabel="Month",ylabel="PM2.5 (µg/m³)",title="Monthly PM2.5 pattern"); fig.tight_layout(); fig.savefig(figdir/"pm25_by_month.png",dpi=150); plt.close(fig)
    seasonal=seasonal_table(df); seasonal.to_csv(out/"seasonal_summary.csv",index=False)
    fig,ax=plt.subplots(1,2,figsize=(10,3.7)); ax[0].bar(seasonal.season,seasonal["mean"],color="#4c72b0"); ax[0].set(xlabel="Season",ylabel="PM2.5 mean (µg/m³)",title="Seasonal PM2.5 level"); ax[1].bar(seasonal.season,seasonal.pct_days_above_25,color="#dd8452",label=">25 µg/m³"); ax[1].bar(seasonal.season,seasonal.pct_days_above_50,color="#c44e52",label=">50 µg/m³"); ax[1].set(xlabel="Season",ylabel="% of days",title="Seasonal exceedance frequency"); ax[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(figdir/"pm25_by_season.png",dpi=150); plt.close(fig)
    nums=df[value_cols]; pearson=nums.corr(method="pearson"); spearman=nums.corr(method="spearman"); pairs=[]
    for i,a in enumerate(value_cols):
        for b in value_cols[i+1:]: pairs.append({"variable_a":a,"variable_b":b,"pearson_r":pearson.loc[a,b],"spearman_rho":spearman.loc[a,b],"high_correlation_abs_gt_0_9":abs(spearman.loc[a,b])>.9})
    pd.DataFrame(pairs,columns=["variable_a","variable_b","pearson_r","spearman_rho","high_correlation_abs_gt_0_9"]).to_csv(out/"correlations.csv",index=False)
    if len(value_cols)>1:
        fig,ax=plt.subplots(figsize=(max(7,len(value_cols)*.7),6)); im=ax.imshow(spearman,vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(len(value_cols)),value_cols,rotation=90); ax.set_yticks(range(len(value_cols)),value_cols); fig.colorbar(im,ax=ax,label="Spearman ρ"); fig.tight_layout(); fig.savefig(figdir/"spearman_correlations.png",dpi=160); plt.close(fig)
        for col in value_cols[1:]:
            fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(df[col],df.pm25,s=7,alpha=.25); ax.set(xlabel=col,ylabel="PM2.5 (µg/m³)",title=f"PM2.5 vs {col}"); fig.tight_layout(); fig.savefig(figdir/f"association_pm25_{col}.png",dpi=150); plt.close(fig)
