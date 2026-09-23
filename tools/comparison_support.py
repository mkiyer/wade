"""Verified comparison tables and restrained scientific plots for the notebooks."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from scipy.stats import t, beta
from notebook_support import sha256

COLORS={"WADE":"#276da8","DESeq2":"#c8792d","DESeq2-no-Cooks":"#9a4b61","edgeR-QL":"#579082"}


def as_polars(frame):
    """Small display tables without an implicit PyArrow dependency."""
    import polars as pl
    return pl.DataFrame(frame.to_dict(orient="list"))


def load(root,kind,out,*,table="scores.tsv"):
    root=Path(root)/kind
    registry=json.loads((root/"checksums.json").read_text())
    for name,digest in registry["files"].items():
        if sha256(root/name)!=digest:
            raise ValueError(f"Changed comparison output: {root/name}")
    destination=Path(out)/kind
    shutil.copytree(root,destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns("package-source","*.py"))
    return pd.read_csv(root/table,sep="\t"),root


def summary(frame,keys,value):
    rows=[]
    for key,group in frame.groupby(keys,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        x=group[value].dropna().to_numpy(float); n=len(x)
        mean=float(x.mean()) if n else np.nan
        half=float(t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan
        # Replicate means, not gene-binomial confidence intervals. At a
        # boundary, a bounded-variable bound avoids a falsely exact interval.
        lo,hi=max(0,mean-half),min(1,mean+half)
        if n and np.all(x==0):lo,hi=0,1-.025**(1/n)
        if n and np.all(x==1):lo,hi=.025**(1/n),1
        rows.append(dict(zip(keys,key),n=n,mean=mean,lo=lo,hi=hi))
    return pd.DataFrame(rows)


def power_grid(scores,categories,*,geometry="balanced"):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    selected=scores[(scores.panel=="mixed") & (scores.geometry==geometry) & scores.category.isin(categories)]
    table=summary(selected,["method","category","n_case"],"power")
    fig,axes=plt.subplots(2,3,figsize=(11,6.6),layout="constrained",sharey=True)
    for ax,(category,title) in zip(axes.flat,categories.items()):
        for method,color in COLORS.items():
            part=table[(table.method==method)&(table.category==category)].sort_values("n_case")
            if len(part):
                ax.plot(part.n_case,part["mean"],"o-",ms=3,lw=1.3,color=color,label=method)
                ax.fill_between(part.n_case,part.lo,part.hi,color=color,alpha=.08)
        ax.set(title=title,xlabel="Case samples",ylim=(-.02,1.02))
        ax.set_xscale("log")
        ticks=[5,20,100,500] if geometry=="balanced" else [100,250,500]
        ax.set_xticks(ticks,labels=list(map(str,ticks)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.18)
    for ax in axes[:,0]:ax.set_ylabel("Altered genes recovered")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=4,frameon=False)
    return fig,table


def calibration_plot(scores):
    import matplotlib.pyplot as plt
    selected=scores[(scores.category=="all") & (scores.geometry=="balanced")]
    tables=[]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained",sharey=True)
    for ax,panel,title in zip(axes,["complete_null","mixed"],["Complete null: any false call","Mixed: distribution-null FDP"]):
        rows=selected[selected.panel==panel].copy()
        value="fdp_distribution"
        tab=summary(rows,["method","n_case"],value);tab["panel"]=panel;tables.append(tab)
        if panel=="complete_null":
            for i,row in tab.iterrows():
                group=rows[(rows.method==row.method)&(rows.n_case==row.n_case)]
                n=len(group);k=int((group.calls>0).sum())
                tab.loc[i,"lo"]=0 if k==0 else beta.ppf(.025,k,n-k+1)
                tab.loc[i,"hi"]=1 if k==n else beta.ppf(.975,k+1,n-k)
        for method,color in COLORS.items():
            part=tab[tab.method==method].sort_values("n_case")
            if len(part):
                ax.errorbar(part.n_case,part["mean"],yerr=[part["mean"]-part.lo,part.hi-part["mean"]],
                            fmt="o-",ms=4,lw=1.2,color=color,label=method,capsize=2)
        ax.axhline(.05,color="#444444",linestyle="--",lw=1)
        ax.set(title=title,xlabel="Samples per group",ylim=(0,1))
        ax.set_xscale("log");ax.set_xticks([5,20,100,500],labels=["5","20","100","500"])
    axes[0].set_ylabel("Replicate mean and 95% interval")
    axes[0].legend(fontsize=8)
    return fig,pd.concat(tables,ignore_index=True)
