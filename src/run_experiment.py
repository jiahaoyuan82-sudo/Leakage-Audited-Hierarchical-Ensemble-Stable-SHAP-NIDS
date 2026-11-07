import os, sys, json, time, hashlib, platform, warnings, itertools, traceback
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT
for p in [OUT / "results", OUT / "figures", OUT / "models", OUT / "logs", OUT / "data" / "processed"]: p.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
import numpy as np
from scoring import named_log_loss
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, log_loss, brier_score_loss, precision_recall_curve, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
SEED=42
N_JOBS=min(8, os.cpu_count() or 4)
RAW_DATA_DIR=Path(os.environ.get('UNSW_RAW_DATA_DIR', OUT/'data'/'raw')).expanduser().resolve()
TRAIN_PATH=RAW_DATA_DIR/'UNSW_NB15_training-set.csv'
TEST_PATH=RAW_DATA_DIR/'UNSW_NB15_testing-set.csv'
RUNLOG=OUT/"logs/run.log"

def log(msg):
    s=f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {msg}"
    print(s, flush=True)
    with RUNLOG.open("a",encoding="utf-8") as f: f.write(s+"\n")

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def json_dump(obj,path):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def set_plot():
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.titlesize':13,'axes.labelsize':10,'figure.dpi':120})

def target_from(df):
    s=df['attack_cat'].fillna('Normal').astype(str).str.strip()
    return s.where(~s.isin(['','-','nan','None']), 'Normal').replace({'Backdoor':'Backdoors','Dos':'DoS','Reconnaissance':'Reconnaissance'})

def audit_data(train,test):
    log('Starting data quality and split-overlap audit')
    report={'train_shape':list(train.shape),'test_shape':list(test.shape),'train_columns':train.columns.tolist(),'test_columns':test.columns.tolist(),
      'train_dtypes':train.dtypes.astype(str).to_dict(),'test_dtypes':test.dtypes.astype(str).to_dict(),
      'train_missing':train.isna().sum().to_dict(),'test_missing':test.isna().sum().to_dict(),
      'train_infinite_numeric':{},'test_infinite_numeric':{},'train_constant_columns':[], 'test_constant_columns':[],
      'train_exact_duplicate_rows':int(train.duplicated().sum()),'test_exact_duplicate_rows':int(test.duplicated().sum())}
    for name,df in [('train',train),('test',test)]:
        num=df.select_dtypes(include=np.number)
        report[f'{name}_infinite_numeric']={c:int(np.isinf(num[c].to_numpy(dtype=float)).sum()) for c in num if np.isinf(num[c].to_numpy(dtype=float)).any()}
        report[f'{name}_constant_columns']=[c for c in df.columns if df[c].nunique(dropna=False)<=1]
    yt=target_from(train); yv=target_from(test)
    report['train_class_counts']=yt.value_counts().to_dict(); report['test_class_counts']=yv.value_counts().to_dict()
    report['train_class_proportions']=(yt.value_counts(normalize=True)*100).round(6).to_dict()
    report['test_class_proportions']=(yv.value_counts(normalize=True)*100).round(6).to_dict()
    report['binary_train_counts']=yt.ne('Normal').map({True:'Attack',False:'Normal'}).value_counts().to_dict()
    report['binary_test_counts']=yv.ne('Normal').map({True:'Attack',False:'Normal'}).value_counts().to_dict()
    catcols=[c for c in train.select_dtypes(include=['object','category']).columns if c not in ['attack_cat']]
    report['categorical_train_test_values']={c:{'train_unique':int(train[c].nunique(dropna=False)),'test_unique':int(test[c].nunique(dropna=False)),
      'test_only_values':sorted(set(test[c].fillna('__NA__').astype(str))-set(train[c].fillna('__NA__').astype(str))),
      'train_only_values':sorted(set(train[c].fillna('__NA__').astype(str))-set(test[c].fillna('__NA__').astype(str)))} for c in catcols}
    # Full-row canonical hashes include labels; no target-derived value is used as a predictor.
    train_hash=pd.util.hash_pandas_object(train.astype(str),index=False).astype('uint64')
    test_hash=pd.util.hash_pandas_object(test.astype(str),index=False).astype('uint64')
    trset=set(train_hash.tolist()); overlap=test_hash.isin(trset)
    report['cross_split_exact_row_hash_overlap_count']=int(overlap.sum())
    report['cross_split_exact_row_hash_overlap_test_fraction']=float(overlap.mean())
    report['cross_split_exact_row_hash_method']='pandas hash_pandas_object over all CSV columns cast to string; uint64 row hashes; labels included for identity audit only'
    pd.DataFrame({'split':['train']*len(train_hash)+['test']*len(test_hash),'row_hash':pd.concat([train_hash,test_hash],ignore_index=True).map(lambda z:f'{int(z):016x}')}).to_csv(OUT/'data/processed/row_hashes.csv.gz',index=False,compression='gzip')
    report['feature_policy']={'dropped':['id','label','attack_cat'],'reason':'id is record identifier; label and attack_cat encode target; test labels retained only for final evaluation'}
    json_dump(report,OUT/'results/data_quality_audit.json')
    # plot class distribution
    set_plot(); counts=pd.concat([yt.value_counts().rename('Train'),yv.value_counts().rename('Test')],axis=1).fillna(0).reindex(['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoors','Shellcode','Worms'])
    ax=counts.plot(kind='bar',figsize=(10,5),color=['#2463a6','#ef8a26']); ax.set_title('UNSW-NB15 Class Distribution'); ax.set_ylabel('Records'); ax.set_xlabel('Class'); plt.xticks(rotation=35,ha='right'); plt.tight_layout(); plt.savefig(OUT/'figures/class_distribution.png',dpi=300); plt.savefig(OUT/'figures/class_distribution.pdf'); plt.close()
    return report,overlap.to_numpy()

def build_preprocessor(X):
    numeric=X.select_dtypes(include=np.number).columns.tolist()
    categorical=[c for c in X.columns if c not in numeric]
    numpipe=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler(with_mean=False))])
    catpipe=Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('ohe',OneHotEncoder(handle_unknown='ignore',min_frequency=5,sparse_output=True,dtype=np.float32))])
    return ColumnTransformer([('num',numpipe,numeric),('cat',catpipe,categorical)],sparse_threshold=1.0,verbose_feature_names_out=True)

def class_names(y): return sorted(pd.Series(y).unique().tolist())

def fit_predict(model,Xtr,ytr,Xval,weight=True,eval_metric=None):
    sw=compute_sample_weight('balanced',ytr) if weight else None
    if isinstance(model,xgb.XGBClassifier):
        kw={'sample_weight':sw}
        model.fit(Xtr,ytr,eval_set=[(Xval,_CURRENT_YVAL)],verbose=False,early_stopping_rounds=40,**kw)
    elif isinstance(model,lgb.LGBMClassifier):
        model.fit(Xtr,ytr,sample_weight=sw,eval_set=[(Xval,_CURRENT_YVAL)],callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
    elif isinstance(model,LogisticRegression): model.fit(Xtr,ytr,sample_weight=sw)
    else: model.fit(Xtr,ytr)
    return model

_CURRENT_YVAL=None

def models_binary(seed=42):
    return {
      'ExtraTrees':ExtraTreesClassifier(n_estimators=350,max_features='sqrt',min_samples_leaf=1,class_weight='balanced',n_jobs=N_JOBS,random_state=seed),
      'XGBoost':xgb.XGBClassifier(n_estimators=500,max_depth=7,learning_rate=.07,subsample=.85,colsample_bytree=.85,reg_lambda=2,
        objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=N_JOBS,random_state=seed,early_stopping_rounds=40),
      'LightGBM':lgb.LGBMClassifier(n_estimators=650,learning_rate=.06,num_leaves=31,max_depth=-1,subsample=.85,colsample_bytree=.85,
        reg_lambda=2,objective='binary',verbosity=-1,n_jobs=N_JOBS,random_state=seed)}

def models_family(seed=42):
    return {
      'ExtraTrees':ExtraTreesClassifier(n_estimators=350,max_features='sqrt',min_samples_leaf=1,class_weight='balanced',n_jobs=N_JOBS,random_state=seed),
      'XGBoost':xgb.XGBClassifier(n_estimators=500,max_depth=8,learning_rate=.06,subsample=.85,colsample_bytree=.85,reg_lambda=2,
        objective='multi:softprob',eval_metric='mlogloss',tree_method='hist',n_jobs=N_JOBS,random_state=seed,early_stopping_rounds=40),
      'LightGBM':lgb.LGBMClassifier(n_estimators=650,learning_rate=.06,num_leaves=31,max_depth=-1,subsample=.85,colsample_bytree=.85,
        reg_lambda=2,objective='multiclass',verbosity=-1,n_jobs=N_JOBS,random_state=seed)}

def fit_family_models(Xtr,ytr,Xval,yval,seed=42,weighted=True,selected=None):
    global _CURRENT_YVAL
    _CURRENT_YVAL=yval
    fitted={}
    for name,m in models_family(seed).items():
        t=time.perf_counter(); log(f'Fit family model {name} weighted={weighted} seed={seed}')
        if not weighted:
            if name=='ExtraTrees': m.set_params(class_weight=None)
        sw=compute_sample_weight('balanced',ytr) if weighted else None
        if isinstance(m,xgb.XGBClassifier):
            m.fit(Xtr,ytr,sample_weight=sw,eval_set=[(Xval,yval)],verbose=False)
        elif isinstance(m,lgb.LGBMClassifier):
            m.fit(Xtr,ytr,sample_weight=sw,eval_set=[(Xval,yval)],callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
        else: m.fit(Xtr,ytr)
        m._experiment_fit_seconds=time.perf_counter()-t
        log(f'Finished {name}; elapsed={m._experiment_fit_seconds:.1f}s; trees={getattr(m,"best_iteration_",None) or getattr(m,"best_iteration",None)}')
        fitted[name]=m
    return fitted

def fit_binary_models(Xtr,ytr,Xval,yval,seed=42,weighted=True):
    fitted={}
    for name,m in models_binary(seed).items():
        t=time.perf_counter(); log(f'Fit binary model {name} weighted={weighted} seed={seed}')
        sw=compute_sample_weight('balanced',ytr) if weighted else None
        if name=='ExtraTrees':
            if not weighted: m.set_params(class_weight=None)
            m.fit(Xtr,ytr)
        elif name=='XGBoost': m.fit(Xtr,ytr,sample_weight=sw,eval_set=[(Xval,yval)],verbose=False)
        else: m.fit(Xtr,ytr,sample_weight=sw,eval_set=[(Xval,yval)],callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
        m._experiment_fit_seconds=time.perf_counter()-t
        log(f'Finished {name}; elapsed={m._experiment_fit_seconds:.1f}s; trees={getattr(m,"best_iteration_",None) or getattr(m,"best_iteration",None)}')
        fitted[name]=m
    return fitted

def proba_aligned(model,X,labels):
    p=model.predict_proba(X); seen=list(model.classes_)
    out=np.zeros((len(p),len(labels)),dtype=np.float64)
    for j,c in enumerate(seen): out[:,labels.index(c)]=p[:,j]
    return out

def binary_metrics(y,p,threshold=.5):
    pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'Precision':precision_score(y,pred,zero_division=0),'Recall':recall_score(y,pred,zero_division=0),'Specificity':tn/(tn+fp) if tn+fp else 0,
      'F1':f1_score(y,pred,zero_division=0),'ROC_AUC':roc_auc_score(y,p),'PR_AUC':average_precision_score(y,p),'MCC':matthews_corrcoef(y,pred),
      'False_Positive_Rate':fp/(fp+tn) if fp+tn else 0,'False_Negative_Rate':fn/(fn+tp) if fn+tp else 0,'Brier_Score':brier_score_loss(y,p),'threshold':float(threshold)}

def multiclass_metrics(y,proba,labels):
    pred=np.asarray(labels)[np.argmax(proba,axis=1)]
    return {'Macro_F1':f1_score(y,pred,labels=labels,average='macro',zero_division=0),'Weighted_F1':f1_score(y,pred,labels=labels,average='weighted',zero_division=0),
      'Balanced_Accuracy':balanced_accuracy_score(y,pred),'MCC':matthews_corrcoef(y,pred),'Log_Loss':named_log_loss(y,proba,labels),'Accuracy':accuracy_score(y,pred)}

def optimize_binary_mix(probs,y):
    names=list(probs); P=np.stack([probs[n] for n in names],axis=1)
    best=None
    # simplex grid uses nonnegative weights summing to one; threshold selected only on validation.
    for a in np.arange(0,1.0001,.1):
      for b in np.arange(0,1-a+.0001,.1):
        c=max(0,1-a-b)
        if abs(a+b+c-1)>1e-6: continue
        w=np.array([a,b,c]); p=P@w
        for th in np.arange(.1,.901,.02):
            m=binary_metrics(y,p,float(th))
            # balanced blend favors attack recall and MCC, uses PR-AUC and false alarms as guardrails.
            score=.30*m['MCC']+.25*m['Recall']+.20*m['PR_AUC']+.15*m['Specificity']+.10*(1-m['False_Negative_Rate'])
            if best is None or score>best['score']: best={'weights':w,'threshold':float(th),'score':float(score),'metrics':m}
    return names,best

def optimize_family_mix(probs,y,labels):
    names=list(probs); P=np.stack([probs[n] for n in names],axis=2)
    best=None
    for a in np.arange(0,1.0001,.1):
      for b in np.arange(0,1-a+.0001,.1):
        c=max(0,1-a-b)
        if abs(a+b+c-1)>1e-6: continue
        w=np.array([a,b,c]); p=np.tensordot(P,w,axes=([2],[0])); p=p/np.maximum(p.sum(axis=1,keepdims=True),1e-15)
        m=multiclass_metrics(y,p,labels)
        if best is None or m['Macro_F1']>best['score']: best={'weights':w,'score':float(m['Macro_F1']),'metrics':m}
    return names,best

def hierarchical_probs(pb,pf,binary_labels,family_labels,attack_threshold=.5):
    pattack=pb
    out=np.zeros((len(pb),len(family_labels)+1),dtype=float)
    out[:,0]=1-pattack
    out[:,1:]=pattack[:,None]*pf
    # threshold decision governs whether any attack is emitted; conditional attack-family posterior retained.
    return out

def class_report(y,p,labels,prefix):
    pred=np.asarray(labels)[np.argmax(p,axis=1)]
    pr,rc,f1,s=__import__('sklearn.metrics').metrics.precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    pd.DataFrame({'class':labels,'precision':pr,'recall':rc,'f1':f1,'support':s}).to_csv(OUT/f'results/{prefix}_per_class.csv',index=False)
    return pred

def save_cm(y,p,labels,title,fn):
    pred=np.asarray(labels)[np.argmax(p,axis=1)]; cm=confusion_matrix(y,pred,labels=labels)
    pd.DataFrame(cm,index=labels,columns=labels).to_csv(OUT/f'results/{fn.replace(".png","_matrix.csv")}')
    set_plot(); fig,ax=plt.subplots(figsize=(10,8)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels,ax=ax,cbar=True)
    ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); plt.xticks(rotation=40,ha='right'); plt.tight_layout(); plt.savefig(OUT/f'figures/{fn}',dpi=300); plt.savefig(OUT/f'figures/{fn.replace(".png",".pdf")}'); plt.close()

def bootstrap_compare(y,p_best,p_ens,labels,n=300):
    rng=np.random.default_rng(SEED); rows=[]; N=len(y); y=np.asarray(y)
    for b in range(n):
        ix=rng.integers(0,N,N); yy=y[ix]
        yb=np.asarray(labels)[np.argmax(p_best[ix],axis=1)]; ye=np.asarray(labels)[np.argmax(p_ens[ix],axis=1)]
        rows.append({'replicate':b+1,'best_single_macro_f1':f1_score(yy,yb,labels=labels,average='macro',zero_division=0),
          'ensemble_macro_f1':f1_score(yy,ye,labels=labels,average='macro',zero_division=0),'delta_macro_f1':f1_score(yy,ye,labels=labels,average='macro',zero_division=0)-f1_score(yy,yb,labels=labels,average='macro',zero_division=0),
          'best_single_mcc':matthews_corrcoef(yy,yb),'ensemble_mcc':matthews_corrcoef(yy,ye),'delta_mcc':matthews_corrcoef(yy,ye)-matthews_corrcoef(yy,yb)})
    d=pd.DataFrame(rows); d.to_csv(OUT/'results/paired_bootstrap_300.csv',index=False)
    summary={}
    for m in ['best_single_macro_f1','ensemble_macro_f1','delta_macro_f1','best_single_mcc','ensemble_mcc','delta_mcc']:
        summary[m]={'mean':float(d[m].mean()),'ci95':[float(d[m].quantile(.025)),float(d[m].quantile(.975))]}
    json_dump(summary,OUT/'results/paired_bootstrap_ci.json'); return summary

def get_feature_groups(pre,cols):
    names=pre.get_feature_names_out().tolist(); groups=[]
    for fn in names:
        raw=fn.split('__',1)[-1]
        # For OHE columns, remove the encoded category suffix by longest raw name prefix.
        matches=[c for c in cols if raw==c or raw.startswith(c+'_')]
        groups.append(max(matches,key=len) if matches else raw)
    return names,groups

def shap_study(models_b,models_f,X_train, yb_train,yf_train,pre,feature_cols,X_family=None):
    import shap
    rng=np.random.default_rng(SEED)
    names,groups=get_feature_groups(pre,feature_cols)
    data_by_stage={'binary':(models_b,X_train,yb_train),'family':(models_f,X_family if X_family is not None else X_train,yf_train)}
    stable={}; consensus={}; per_model_shap={}
    for stage,(models,Xstage,y) in data_by_stage.items():
      stage_ranks={}; stage_imp={};
      yarr=np.asarray(y)
      for model_name in ['XGBoost','LightGBM']:
        model=models[model_name]
        log(f'SHAP bootstrap {stage}/{model_name}')
        by=[]; boot_ranks=[]; tops=[]; expl=shap.TreeExplainer(model)
        for boot in range(3):
            # At most 100 stratified bootstrap rows; only those rows are converted to dense.
            alloc={c:max(1,int(np.floor(100*np.sum(yarr==c)/len(yarr)))) for c in np.unique(yarr)}
            while sum(alloc.values())>100:
                c=max((c for c in alloc if alloc[c]>1),key=lambda z:alloc[z]); alloc[c]-=1
            while sum(alloc.values())<100:
                c=max(alloc,key=lambda z:np.sum(yarr==z)/alloc[z]); alloc[c]+=1
            pieces=[]
            for cls,k in alloc.items():
                pool=np.flatnonzero(yarr==cls); pieces.extend(rng.choice(pool,size=k,replace=True).tolist())
            sample=np.asarray(pieces,dtype=int)
            xs=Xstage[sample]
            if sparse.issparse(xs): xs=xs.toarray()
            sv=expl.shap_values(xs,check_additivity=False)
            if isinstance(sv,list): arr=np.mean([np.abs(z) for z in sv],axis=0)
            else:
                arr=np.asarray(sv)
                if arr.ndim==3: arr=np.mean(np.abs(arr),axis=-1)
                else: arr=np.abs(arr)
            encoded=np.mean(arr,axis=0)
            gd={}
            for g,v in zip(groups,encoded): gd[g]=gd.get(g,0.0)+float(v)
            by.append(gd)
        imp=pd.DataFrame(by).fillna(0); stage_imp[model_name]=imp.mean(axis=0).sort_values(ascending=False)
        ranks=imp.rank(axis=1,ascending=False,method='average'); stage_ranks[model_name]=ranks
        pair=[]; jac=[]
        tops=[set(imp.iloc[i].nlargest(20).index) for i in range(len(imp))]
        for i,j in itertools.combinations(range(3),2):
            jac.append(len(tops[i]&tops[j])/max(1,len(tops[i]|tops[j])))
            pair.append(float(spearmanr(ranks.iloc[i],ranks.iloc[j]).statistic))
        stable[f'{stage}_{model_name}']={'top20_jaccard_mean':float(np.mean(jac)),'top20_jaccard_std':float(np.std(jac,ddof=1)),
          'rank_spearman_mean':float(np.nanmean(pair)),'rank_spearman_std':float(np.nanstd(pair,ddof=1)),'top20_by_bootstrap':[list(t) for t in tops]}
        per_model_shap[(stage,model_name)]=(imp,None,model)
      weights = {'binary':binary_weights,'family':family_weights}[stage]
      explained_weight = sum(float(weights.get(n, 0)) for n in stage_imp) or 1.0
      acc=pd.Series(dtype=float)
      for n,s in stage_imp.items():
        z=s/s.sum() if s.sum() else s
        part=z*(float(weights.get(n,0))/explained_weight)
        acc=acc.add(part,fill_value=0)
      acc=acc.sort_values(ascending=False); consensus[stage]=acc
    # across both hierarchy stages: normalized mean consensus, explicitly termed weighted SHAP consensus importance.
    full=pd.concat([consensus['binary'].rename('binary'),consensus['family'].rename('family')],axis=1).fillna(0)
    full['weighted_shap_consensus_importance']=full.mean(axis=1); full=full.sort_values('weighted_shap_consensus_importance',ascending=False)
    full.to_csv(OUT/'results/shap_weighted_consensus_importance.csv')
    json_dump(stable,OUT/'results/shap_stability.json')
    # Bootstrap table and stability figure.
    stabrows=[{'model_stage':k,**{m:v[m] for m in ['top20_jaccard_mean','rank_spearman_mean']}} for k,v in stable.items()]
    sdf=pd.DataFrame(stabrows); sdf.to_csv(OUT/'results/shap_stability.csv',index=False)
    set_plot(); ax=sdf.set_index('model_stage')[['top20_jaccard_mean','rank_spearman_mean']].plot(kind='bar',figsize=(10,5),ylim=(0,1),color=['#2c7fb8','#7fcdbb']); ax.set_title('SHAP Bootstrap Stability'); ax.set_ylabel('Mean stability (0–1)'); ax.set_xlabel('Model'); plt.xticks(rotation=30,ha='right'); plt.tight_layout(); plt.savefig(OUT/'figures/shap_stability.png',dpi=300); plt.savefig(OUT/'figures/shap_stability.pdf'); plt.close()
    # Global consensus plot
    top=full.head(20).sort_values('weighted_shap_consensus_importance'); ax=top['weighted_shap_consensus_importance'].plot(kind='barh',figsize=(9,7),color='#3178a8'); ax.set_title('Weighted SHAP Consensus Global Importance'); ax.set_xlabel('Normalized mean absolute SHAP importance'); plt.tight_layout(); plt.savefig(OUT/'figures/shap_global_importance.png',dpi=300); plt.savefig(OUT/'figures/shap_global_importance.pdf'); plt.close()
    # Beeswarm for selected core binary LightGBM from training distribution.
    model=models_b['LightGBM']; yarr=np.asarray(yb_train); sample=[]
    for cls in np.unique(yarr):
        pool=np.flatnonzero(yarr==cls); sample.extend(rng.choice(pool,size=max(1,int(round(100*len(pool)/len(yarr)))),replace=True).tolist())
    sample=np.asarray(sample[:100],dtype=int); xs=X_train[sample]; xs=xs.toarray() if sparse.issparse(xs) else np.asarray(xs)
    shap_values=shap.TreeExplainer(model).shap_values(xs,check_additivity=False)
    if isinstance(shap_values,list): shap_values=shap_values[-1]
    elif np.asarray(shap_values).ndim==3: shap_values=np.asarray(shap_values)[:,:, -1]
    set_plot(); shap.summary_plot(shap_values,xs,feature_names=names,max_display=20,show=False); plt.gcf().set_size_inches(10,7); plt.tight_layout(); plt.savefig(OUT/'figures/shap_beeswarm.png',dpi=300,bbox_inches='tight'); plt.close()
    selected=full.head(20).index.tolist()
    return selected,full,stable

def waterfall_plots(model,X_test,raw_test,y_test,pred_test,feat_names,labels=None):
    import shap
    cases=[]
    yb=(np.asarray(y_test)!='Normal').astype(int); pred_attack=(np.asarray(pred_test)!='Normal').astype(int)
    masks={'correct_prediction':np.asarray(y_test)==np.asarray(pred_test),'false_positive':(yb==0)&(pred_attack==1),'false_negative':(yb==1)&(pred_attack==0),'rare_attack':np.asarray(y_test)=='Worms'}
    for key,mask in masks.items():
        ix=np.flatnonzero(mask)
        if len(ix): cases.append((key,int(ix[0])))
    ex=shap.TreeExplainer(model)
    for kind,ix in cases:
        x=X_test[ix:ix+1]
        if sparse.issparse(x): x=x.toarray()
        sv=ex(x,check_additivity=False)
        vals=np.asarray(sv.values)
        if vals.ndim==3: vals=vals[0,:,-1]
        elif vals.ndim==2: vals=vals[0]
        base=np.asarray(sv.base_values); base=float(base.flat[-1])
        row=np.asarray(x)[0]
        exp=shap.Explanation(values=vals,base_values=base,data=row,feature_names=feat_names)
        set_plot(); shap.plots.waterfall(exp,max_display=15,show=False); plt.title(f'{kind.replace("_"," ").title()} — true {y_test[ix]}, predicted {pred_test[ix]}'); plt.tight_layout(); plt.savefig(OUT/f'figures/shap_waterfall_{kind}.png',dpi=300,bbox_inches='tight'); plt.savefig(OUT/f'figures/shap_waterfall_{kind}.pdf',bbox_inches='tight'); plt.close()
        pd.DataFrame({'feature':feat_names,'value':row,'shap_value':vals}).assign(abs_shap=lambda d:d.shap_value.abs()).sort_values('abs_shap',ascending=False).head(20).to_csv(OUT/f'results/waterfall_{kind}_top_contributions.csv',index=False)

def perf_plot(comp):
    d=pd.DataFrame(comp); d.to_csv(OUT/'results/model_comparison.csv',index=False)
    set_plot(); fig,ax=plt.subplots(figsize=(11,6)); d.set_index('model')[['Macro_F1','Weighted_F1','Balanced_Accuracy']].plot(kind='bar',ax=ax,color=['#136c9e','#41ab5d','#feb24c']); ax.set_title('Model Performance Comparison'); ax.set_ylabel('Score'); ax.set_ylim(0,1); plt.xticks(rotation=35,ha='right'); plt.tight_layout(); plt.savefig(OUT/'figures/model_comparison.png',dpi=300); plt.savefig(OUT/'figures/model_comparison.pdf'); plt.close()

def main():
    global binary_weights,family_weights
    start=time.perf_counter(); set_plot()
    log(f'Environment {platform.platform()}; Python={platform.python_version()}; workers={N_JOBS}')
    train=pd.read_csv(TRAIN_PATH,low_memory=False); test=pd.read_csv(TEST_PATH,low_memory=False)
    audit,overlap=audit_data(train,test)
    # no predictors or transformations use test targets; only the audit records them and later evaluation reads them.
    yall=target_from(train); ytest=target_from(test)
    class_order=['Normal','Analysis','Backdoors','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms']
    for c in class_order:
        if c not in set(yall): raise ValueError(f'Missing expected train class: {c}')
    Xall=train.drop(columns=['id','label','attack_cat']); Xtest=test.drop(columns=['id','label','attack_cat'])
    Xtr_raw,Xval_raw,ytr,yval=train_test_split(Xall,yall,test_size=.20,random_state=SEED,stratify=yall)
    # Preprocessing fits only on the stratified training fold.
    pre=build_preprocessor(Xtr_raw); Xtr=pre.fit_transform(Xtr_raw); Xval=pre.transform(Xval_raw); Xt=pre.transform(Xtest)
    joblib.dump(pre,OUT/'models/preprocessor.joblib',compress=3)
    labels=['Normal','Analysis','Backdoors','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms']
    # Binary Normal/Attack models
    ytrb=(ytr!='Normal').astype(int).to_numpy(); yvb=(yval!='Normal').astype(int).to_numpy(); ytb=(ytest!='Normal').astype(int).to_numpy()
    bm=fit_binary_models(Xtr,ytrb,Xval,yvb,SEED,True)
    bval={n:m.predict_proba(Xval)[:,1] for n,m in bm.items()}; names_b,bopt=optimize_binary_mix(bval,yvb); binary_weights=dict(zip(names_b,bopt['weights'].tolist()))
    log(f'Binary validation optimized weights={binary_weights}, threshold={bopt["threshold"]:.3f}, score={bopt["score"]:.4f}')
    # Attack family models train only on attack flows in the training split.
    mask_tr=ytr.to_numpy()!='Normal'; mask_val=yval.to_numpy()!='Normal'
    Xtra=Xtr[mask_tr]; Xvala=Xval[mask_val]; yftr=ytr[mask_tr].to_numpy(); yfval=yval[mask_val].to_numpy()
    # Encode family labels to consecutive integers for XGBoost and LightGBM.
    le=LabelEncoder().fit(yftr); yftri=le.transform(yftr); yfvai=le.transform(yfval); fam_int=list(range(len(le.classes_)))
    fm=fit_family_models(Xtra,yftri,Xvala,yfvai,SEED,True)
    fval={n:proba_aligned(m,Xvala,fam_int) for n,m in fm.items()}; names_f,fopt=optimize_family_mix(fval,yfvai,fam_int); family_weights=dict(zip(names_f,fopt['weights'].tolist()))
    log(f'Family validation weights={family_weights}; macro_f1={fopt["score"]:.4f}')
    # Validation threshold/weight selected; hierarchical attack/no-attack decision uses threshold.
    fval_all={n:proba_aligned(m,Xval,fam_int) for n,m in fm.items()}
    pvalb=sum(binary_weights[n]*bval[n] for n in names_b); pvalf=sum(family_weights[n]*fval_all[n] for n in names_f)
    pvalfull=np.zeros((len(yval),len(labels))); pvalfull[:,0]=1-pvalb; pvalfull[:,1:]=pvalb[:,None]*pvalf
    pvalfull[pvalb < bopt['threshold'],1:]=0; pvalfull[pvalb < bopt['threshold'],0]=1
    yval10=yval.to_numpy(); hier_val=multiclass_metrics(yval10,pvalfull,labels)
    # Train required baselines on the same leakage-safe training split.
    log('Fit Logistic Regression and Random Forest baselines')
    lr=LogisticRegression(max_iter=600,C=1.0,class_weight='balanced',solver='saga',n_jobs=N_JOBS,random_state=SEED)
    t0=time.perf_counter(); lr.fit(Xtr,ytrb); time_lr=time.perf_counter()-t0
    rf=RandomForestClassifier(n_estimators=300,max_features='sqrt',class_weight='balanced_subsample',n_jobs=N_JOBS,random_state=SEED,min_samples_leaf=1)
    t0=time.perf_counter(); rf.fit(Xtr,ytrb); time_rf=time.perf_counter()-t0
    # Direct ten-class LightGBM, same train-only preprocessing and train-derived label mapping.
    le10=LabelEncoder().fit(ytr); ytri10=le10.transform(ytr); yvai10=le10.transform(yval)
    direct=lgb.LGBMClassifier(n_estimators=700,learning_rate=.06,num_leaves=31,objective='multiclass',verbosity=-1,n_jobs=N_JOBS,random_state=SEED)
    t0=time.perf_counter(); direct.fit(Xtr,ytri10,sample_weight=compute_sample_weight('balanced',ytri10),eval_set=[(Xval,yvai10)],callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)]); time_direct=time.perf_counter()-t0; direct._experiment_fit_seconds=time_direct
    # Final ensemble and individual tree model probabilities on validation/test data.
    btest={n:m.predict_proba(Xt)[:,1] for n,m in bm.items()}; pbt=sum(binary_weights[n]*btest[n] for n in names_b)
    pft={n:proba_aligned(m,Xt,fam_int) for n,m in fm.items()}; pfa=sum(family_weights[n]*pft[n] for n in names_f); pfa/=pfa.sum(axis=1,keepdims=True)
    ptest=np.zeros((len(test),len(labels))); ptest[:,0]=1-pbt; ptest[:,1:]=pbt[:,None]*pfa
    use_attack=pbt>=bopt['threshold']; ptest[~use_attack,0]=1; ptest[~use_attack,1:]=0
    # direct LightGBM output mapped to canonical label order
    pdirect_raw=proba_aligned(direct,Xt,list(range(len(le10.classes_)))); pdirect=np.zeros((len(test),len(labels)))
    for j,c in enumerate(le10.classes_): pdirect[:,labels.index(c)]=pdirect_raw[:,j]
    # binary single-model metrics and validation-chosen threshold for each model
    bin_rows=[]; bin_test_probs={}
    for name,m in bm.items():
        pv=bval[name]; cand=np.arange(.1,.901,.01); th=max(cand,key=lambda x:.30*binary_metrics(yvb,pv,x)['MCC']+.25*binary_metrics(yvb,pv,x)['Recall']+.20*binary_metrics(yvb,pv,x)['PR_AUC']+.15*binary_metrics(yvb,pv,x)['Specificity']+.10*(1-binary_metrics(yvb,pv,x)['False_Negative_Rate']))
        pt=btest[name]; bin_test_probs[name]=pt; row={'model':name,**binary_metrics(ytb,pt,float(th))}; bin_rows.append(row)
    ensemble_binary=binary_metrics(ytb,pbt,bopt['threshold']); bin_rows.append({'model':'Optimized hierarchical ensemble',**ensemble_binary})
    for name,model in [('Logistic Regression',lr),('Random Forest',rf)]:
        pv=model.predict_proba(Xval)[:,1]; pt=model.predict_proba(Xt)[:,1]
        cand=np.arange(.1,.901,.01)
        th=max(cand,key=lambda x:.30*binary_metrics(yvb,pv,x)['MCC']+.25*binary_metrics(yvb,pv,x)['Recall']+.20*binary_metrics(yvb,pv,x)['PR_AUC']+.15*binary_metrics(yvb,pv,x)['Specificity']+.10*(1-binary_metrics(yvb,pv,x)['False_Negative_Rate']))
        bin_test_probs[name]=pt; bin_rows.append({'model':name,**binary_metrics(ytb,pt,float(th))})
    pd.DataFrame(bin_rows).to_csv(OUT/'results/binary_metrics.csv',index=False)
    joblib.dump({'LogisticRegression':lr,'RandomForest':rf,'thresholds_validation_selected':{r['model']:r['threshold'] for r in bin_rows}},OUT/'models/baseline_binary_models.joblib',compress=3)
    # Core stage outputs saved for auditability.
    ypred=np.asarray(labels)[np.argmax(ptest,axis=1)]; ydirect=np.asarray(labels)[np.argmax(pdirect,axis=1)]
    comparison=[]
    for n,p in [('Direct LightGBM 10-class',pdirect),('Hierarchical optimized ensemble',ptest)]: comparison.append({'model':n,**multiclass_metrics(ytest.to_numpy(),p,labels)})
    # Equal-weight soft voting ablation
    peq_b=np.mean(np.column_stack([btest[n] for n in names_b]),axis=1); peq_f=np.mean(np.stack([pft[n] for n in names_f],axis=2),axis=2); peq_f/=peq_f.sum(axis=1,keepdims=True)
    peq=np.zeros_like(ptest); peq[:,0]=1-peq_b; peq[:,1:]=peq_b[:,None]*peq_f; peq[peq_b<bopt['threshold'],0]=1; peq[peq_b<bopt['threshold'],1:]=0
    equal_metrics=multiclass_metrics(ytest.to_numpy(),peq,labels); comparison.append({'model':'Equal-weight hierarchical ensemble',**equal_metrics})
    # strongest single hierarchical model is selected by validation macro-F1 over each base model paired with best family model
    single_probs={}
    for bn in names_b:
      for fn in names_f:
        ppb=btest[bn]; ppf=pft[fn]; q=np.zeros_like(ptest); q[:,0]=1-ppb; q[:,1:]=ppb[:,None]*ppf; q[ppb<bopt['threshold'],0]=1; q[ppb<bopt['threshold'],1:]=0
        single_probs[f'{bn}+{fn}']=q
    # Select the best single pair using validation only; test predictions are materialized afterward.
    val_single_scores={}
    for bn in names_b:
      for fn in names_f:
        q=np.zeros((len(yval),len(labels))); q[:,0]=1-bval[bn]; q[:,1:]=bval[bn][:,None]*fval_all[fn]
        q[bval[bn]<bopt['threshold'],0]=1; q[bval[bn]<bopt['threshold'],1:]=0
        val_single_scores[f'{bn}+{fn}']=multiclass_metrics(yval10,q,labels)['Macro_F1']
    best_name=max(val_single_scores,key=val_single_scores.get)
    best_bn,best_fn=best_name.split('+'); pbest=single_probs[best_name]
    comparison.append({'model':f'Best single pair ({best_name})',**multiclass_metrics(ytest.to_numpy(),pbest,labels)})
    # Individual stage-2 / direct reference metrics
    for n,p in pft.items():
        q=np.zeros_like(ptest); q[:,0]=1-pbt; q[:,1:]=pbt[:,None]*p; q[~use_attack,0]=1; q[~use_attack,1:]=0
        comparison.append({'model':f'Hierarchical {n}',**multiclass_metrics(ytest.to_numpy(),q,labels)})
    perf_plot(comparison)
    # Confusion / class-wise / prediction probabilities.
    class_report(ytest.to_numpy(),ptest,labels,'hierarchical_ensemble')
    save_cm(ytest.to_numpy(),ptest,labels,'Hierarchical Ensemble Confusion Matrix','confusion_matrix_10class.png')
    yf_true_binary=np.asarray(ytest); predb=(pbt>=bopt['threshold']).astype(int)
    # Save raw labels and all predicted class probabilities including direct and equal soft vote.
    pred_df=pd.DataFrame({'row_index':np.arange(len(test)),'true_label':ytest.to_numpy(),'predicted_label':ypred,'predicted_binary_attack_probability':pbt,'binary_threshold':bopt['threshold'],'direct_lightgbm_label':ydirect,'equal_weight_label':np.asarray(labels)[np.argmax(peq,axis=1)]})
    for j,c in enumerate(labels): pred_df[f'prob_{c}']=ptest[:,j]; pred_df[f'direct_prob_{c}']=pdirect[:,j]
    pred_df.to_csv(OUT/'results/test_predictions_and_probabilities.csv.gz',index=False,compression='gzip')
    # ROC and PR curves: binary and macro one-v-rest multiclass curves.
    set_plot(); fig,axs=plt.subplots(1,2,figsize=(12,5))
    for n,p in bin_test_probs.items():
        fpr,tpr,_=roc_curve(ytb,p); axs[0].plot(fpr,tpr,label=n)
        pr,rc,_=precision_recall_curve(ytb,p); axs[1].plot(rc,pr,label=f'{n} AP={average_precision_score(ytb,p):.3f}')
    fpr,tpr,_=roc_curve(ytb,pbt); axs[0].plot(fpr,tpr,label='Optimized ensemble',lw=2); pr,rc,_=precision_recall_curve(ytb,pbt); axs[1].plot(rc,pr,label='Optimized ensemble',lw=2)
    axs[0].plot([0,1],[0,1],'k--',alpha=.4); axs[0].set_title('Binary ROC Curves'); axs[1].set_title('Binary Precision–Recall Curves')
    for ax in axs: ax.set_xlabel('False Positive Rate' if ax is axs[0] else 'Recall'); ax.set_ylabel('True Positive Rate' if ax is axs[0] else 'Precision'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(OUT/'figures/roc_pr_curves.png',dpi=300); plt.savefig(OUT/'figures/roc_pr_curves.pdf'); plt.close()
    # Ablations: class weights, single-vs-ensemble, direct-vs-hierarchical, duplicates sensitivity.
    log('Fit class-weight ablation models (unweighted LightGBM at each hierarchy stage)')
    b_un= lgb.LGBMClassifier(n_estimators=650,learning_rate=.06,num_leaves=31,objective='binary',verbosity=-1,n_jobs=N_JOBS,random_state=SEED)
    t0=time.perf_counter(); b_un.fit(Xtr,ytrb,eval_set=[(Xval,yvb)],callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)]); time_bun=time.perf_counter()-t0
    f_un=lgb.LGBMClassifier(n_estimators=650,learning_rate=.06,num_leaves=31,objective='multiclass',verbosity=-1,n_jobs=N_JOBS,random_state=SEED)
    t0=time.perf_counter(); f_un.fit(Xtra,yftri,eval_set=[(Xvala,yfvai)],callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)]); time_fun=time.perf_counter()-t0
    pbu=b_un.predict_proba(Xt)[:,1]; pfu=proba_aligned(f_un,Xt,fam_int); pu=np.zeros_like(ptest); pu[:,0]=1-pbu; pu[:,1:]=pbu[:,None]*pfu; pu[pbu<bopt['threshold'],0]=1; pu[pbu<bopt['threshold'],1:]=0
    unweighted_metrics=multiclass_metrics(ytest.to_numpy(),pu,labels)
    ablations=[{'ablation':'Optimized class-weighted ensemble','Macro_F1':multiclass_metrics(ytest.to_numpy(),ptest,labels)['Macro_F1'],'MCC':multiclass_metrics(ytest.to_numpy(),ptest,labels)['MCC']},
      {'ablation':'Unweighted LightGBM hierarchy','Macro_F1':unweighted_metrics['Macro_F1'],'MCC':unweighted_metrics['MCC']},
      {'ablation':'Equal-weight soft vote','Macro_F1':equal_metrics['Macro_F1'],'MCC':equal_metrics['MCC']},
      {'ablation':'Direct 10-class LightGBM','Macro_F1':multiclass_metrics(ytest.to_numpy(),pdirect,labels)['Macro_F1'],'MCC':multiclass_metrics(ytest.to_numpy(),pdirect,labels)['MCC']},
      {'ablation':'Hierarchical optimized full features','Macro_F1':multiclass_metrics(ytest.to_numpy(),ptest,labels)['Macro_F1'],'MCC':multiclass_metrics(ytest.to_numpy(),ptest,labels)['MCC']}]
    pd.DataFrame(ablations).to_csv(OUT/'results/ablation_results.csv',index=False)
    set_plot(); fig,ax=plt.subplots(figsize=(10,5)); adf=pd.DataFrame(ablations).set_index('ablation')[['Macro_F1','MCC']]; adf.plot(kind='bar',ax=ax,color=['#2878a8','#f28e2b']); ax.set_title('Ablation Study'); ax.set_ylabel('Score'); ax.set_ylim(0,1); plt.xticks(rotation=30,ha='right'); plt.tight_layout(); plt.savefig(OUT/'figures/ablation_study.png',dpi=300); plt.savefig(OUT/'figures/ablation_study.pdf'); plt.close()
    # Duplicate-row sensitivity analysis uses exactly same fitted models and fixed threshold.
    keep=~overlap
    dup_metrics=multiclass_metrics(ytest.to_numpy()[keep],ptest[keep],labels) if keep.any() else {}
    duplicate_sensitivity={'official_test':multiclass_metrics(ytest.to_numpy(),ptest,labels),'test_after_removing_train_duplicate_rows':dup_metrics,'removed_rows':int(overlap.sum()),'remaining_rows':int(keep.sum())}
    json_dump(duplicate_sensitivity,OUT/'results/duplicate_sensitivity.json')
    # Bootstrap paired comparison, 300 fixed-seed paired resamples.
    boot=bootstrap_compare(ytest.to_numpy(),pbest,ptest,labels,300)
    # Model serialization and measure training/inference time and sizes.
    joblib.dump({'binary_models':bm,'family_models':fm,'direct_ten_class':direct,'label_encoder_family':le,'label_encoder_10class':le10,
      'binary_weights':binary_weights,'family_weights':family_weights,'threshold':float(bopt['threshold']),'class_order':labels,'feature_columns':Xall.columns.tolist()},OUT/'models/ensemble.joblib',compress=3)
    joblib.dump({'binary':b_un,'family':f_un},OUT/'models/unweighted_ablation.joblib',compress=3)
    modelsizes={p.name:p.stat().st_size for p in (OUT/'models').glob('*.joblib')}
    # Inference time, mean over repeated batches, per 1000 rows; include complete prediction pipeline.
    t0=time.perf_counter()
    for _ in range(5):
        xb=Xt[:min(10000,len(test))]
        pb0=sum(binary_weights[n]*m.predict_proba(xb)[:,1] for n,m in bm.items())
        pf0=sum(family_weights[n]*proba_aligned(m,xb,fam_int) for n,m in fm.items())
        _=np.column_stack([1-pb0,pb0[:,None]*pf0])
    infer_each1000=(time.perf_counter()-t0)/5/min(10000,len(test))*1000
    efficiency={'train_seconds':{'LogisticRegression':time_lr,'RandomForest':time_rf,'DirectLightGBM':time_direct,'binary_stage_ensemble':sum(getattr(m,'_experiment_fit_seconds',0) for m in bm.values()),'family_stage_ensemble':sum(getattr(m,'_experiment_fit_seconds',0) for m in fm.values()),'unweighted_ablation_fit_total':time_bun+time_fun},
      'inference_seconds_per_1000_rows_approx':infer_each1000,'saved_model_bytes':modelsizes,'compressed_models_total_bytes':int(sum(modelsizes.values()))}
    json_dump(efficiency,OUT/'results/efficiency.json')
    # Validation-only SHAP explains core tree models and selects compact features; test rows are excluded.
    selected,shapcons,stability=shap_study(bm,fm,Xtr,ytrb,yftri,pre,Xall.columns.tolist(),Xtra)
    json_dump({'selected_top20_stable_features':selected,'method':'top 20 from mean normalized weighted SHAP consensus across three training-only stratified bootstrap samples of at most 100 rows per XGBoost/LightGBM member and hierarchy stage','explained_models':['XGBoost','LightGBM'],'excluded_from_shap':'ExtraTrees (TreeSHAP cost too high)','test_data_used_for_selection':False},OUT/'results/stable_feature_selection.json')
    # Compact model uses only stable features, fitting a fresh encoder on train fold only.
    compact_start=time.perf_counter()
    raw_cols=[c for c in selected if c in Xtr_raw.columns]; Xtrc_raw=Xtr_raw[raw_cols]; Xvc_raw=Xval_raw[raw_cols]; Xtc_raw=Xtest[raw_cols]
    prec=build_preprocessor(Xtrc_raw); Xtrc=prec.fit_transform(Xtrc_raw); Xvc=prec.transform(Xvc_raw); Xtc=prec.transform(Xtc_raw); joblib.dump(prec,OUT/'models/compact_preprocessor.joblib',compress=3)
    bmc=fit_binary_models(Xtrc,ytrb,Xvc,yvb,SEED,True); bcmv={n:m.predict_proba(Xvc)[:,1] for n,m in bmc.items()}; _,bcopt=optimize_binary_mix(bcmv,yvb); bweights_c=dict(zip(bcmv.keys(),bcopt['weights'].tolist()))
    mt=ytr.to_numpy()!='Normal'; mv=yval.to_numpy()!='Normal'; lce=LabelEncoder().fit(ytr.to_numpy()[mt]); yctr=lce.transform(ytr.to_numpy()[mt]); ycv=lce.transform(yval.to_numpy()[mv])
    fmc=fit_family_models(Xtrc[mt],yctr,Xvc[mv],ycv,SEED,True); fcmv={n:proba_aligned(m,Xvc[mv],list(range(len(lce.classes_)))) for n,m in fmc.items()}; _,fcopt=optimize_family_mix(fcmv,ycv,list(range(len(lce.classes_)))); fweights_c=dict(zip(fcmv.keys(),fcopt['weights'].tolist()))
    pctb={n:m.predict_proba(Xtc)[:,1] for n,m in bmc.items()}; pc_b=sum(bweights_c[n]*pctb[n] for n in bweights_c); pctf={n:proba_aligned(m,Xtc,list(range(len(lce.classes_)))) for n,m in fmc.items()}; pc_f=sum(fweights_c[n]*pctf[n] for n in fweights_c); pc_f/=pc_f.sum(axis=1,keepdims=True)
    pc=np.zeros_like(ptest); pc[:,0]=1-pc_b; pc[:,1:]=pc_b[:,None]*pc_f; pc[pc_b<bcopt['threshold'],0]=1; pc[pc_b<bcopt['threshold'],1:]=0
    compact_metrics=multiclass_metrics(ytest.to_numpy(),pc,labels)
    joblib.dump({'binary_models':bmc,'family_models':fmc,'binary_weights':bweights_c,'family_weights':fweights_c,'threshold':float(bcopt['threshold']),'class_order':labels,'selected_features':raw_cols},OUT/'models/compact_ensemble.joblib',compress=3)
    # Save compact efficiency comparison.
    t0=time.perf_counter()
    for _ in range(5):
        xc=Xtc[:min(10000,len(test))]
        pcb=sum(bweights_c[n]*m.predict_proba(xc)[:,1] for n,m in bmc.items())
        pcf=sum(fweights_c[n]*proba_aligned(m,xc,list(range(len(lce.classes_)))) for n,m in fmc.items())
        _=np.column_stack([1-pcb,pcb[:,None]*pcf])
    compact_infer=(time.perf_counter()-t0)/5/min(10000,len(test))*1000
    compact_size=(OUT/'models/compact_ensemble.joblib').stat().st_size
    compact_train_seconds=time.perf_counter()-compact_start
    compact_eff={'full_macro_f1':multiclass_metrics(ytest.to_numpy(),ptest,labels)['Macro_F1'],'compact_macro_f1':compact_metrics['Macro_F1'],'full_mcc':multiclass_metrics(ytest.to_numpy(),ptest,labels)['MCC'],'compact_mcc':compact_metrics['MCC'],'compact_train_seconds':compact_train_seconds,
      'selected_features':raw_cols,'feature_count_full':len(Xall.columns),'feature_count_compact':len(raw_cols),'full_inference_seconds_per_1000_rows':infer_each1000,'compact_inference_seconds_per_1000_rows':compact_infer,
      'full_model_bundle_bytes':(OUT/'models/ensemble.joblib').stat().st_size,'compact_model_bundle_bytes':compact_size}
    json_dump(compact_eff,OUT/'results/full_vs_compact_efficiency.json')
    set_plot(); fig,ax=plt.subplots(figsize=(7,5)); ax.scatter([infer_each1000],[compact_eff['full_macro_f1']],s=110,label='Full features'); ax.scatter([compact_infer],[compact_metrics['Macro_F1']],s=110,label='Stable SHAP compact'); ax.set_xlabel('Inference seconds / 1,000 rows (lower is better)'); ax.set_ylabel('Macro-F1 (higher is better)'); ax.set_title('Performance–Efficiency: Full vs Compact'); ax.legend(); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(OUT/'figures/full_vs_compact_efficiency.png',dpi=300); plt.savefig(OUT/'figures/full_vs_compact_efficiency.pdf'); plt.close()
    pd.DataFrame([{'model':'Full hierarchical ensemble',**multiclass_metrics(ytest.to_numpy(),ptest,labels)}, {'model':'Compact SHAP-stable ensemble',**compact_metrics}]).to_csv(OUT/'results/full_vs_compact_metrics.csv',index=False)
    # Final test explanations use test cases only after feature selection is complete.
    enc_names=pre.get_feature_names_out().tolist(); waterfall_plots(bm['LightGBM'],Xt,test,ytest.to_numpy(),ypred,enc_names)
    # Seed stability check for final ensemble and best single pair, exactly two additional seeds.
    seed_rows=[]
    for seed in [17,2026]:
        log(f'Additional random seed robustness run {seed}')
        bseed=fit_binary_models(Xtr,ytrb,Xval,yvb,seed,True); fseed=fit_family_models(Xtra,yftri,Xvala,yfvai,seed,True)
        psb={n:m.predict_proba(Xt)[:,1] for n,m in bseed.items()}; ppb=sum(binary_weights[n]*psb[n] for n in names_b)
        psf={n:proba_aligned(m,Xt,fam_int) for n,m in fseed.items()}; ppf=sum(family_weights[n]*psf[n] for n in names_f); ppf/=ppf.sum(axis=1,keepdims=True)
        qs=np.zeros_like(ptest); qs[:,0]=1-ppb; qs[:,1:]=ppb[:,None]*ppf; qs[ppb<bopt['threshold'],0]=1; qs[ppb<bopt['threshold'],1:]=0
        qsingle=np.zeros_like(ptest); qsingle[:,0]=1-psb[best_bn]; qsingle[:,1:]=psb[best_bn][:,None]*psf[best_fn]; qsingle[psb[best_bn]<bopt['threshold'],0]=1; qsingle[psb[best_bn]<bopt['threshold'],1:]=0
        seed_rows.append({'seed':seed,'model':'Final optimized ensemble',**multiclass_metrics(ytest.to_numpy(),qs,labels)})
        seed_rows.append({'seed':seed,'model':f'Best single pair {best_name}',**multiclass_metrics(ytest.to_numpy(),qsingle,labels)})
    pd.DataFrame(seed_rows).to_csv(OUT/'results/two_seed_robustness.csv',index=False)
    # Record best parameters and fitted characteristics.
    bestparams={'random_seed':SEED,'split':{'validation_fraction':.20,'stratified':True},'features':{'drop':['id','label','attack_cat'],'full_feature_count':len(Xall.columns),'compact_features':raw_cols},
      'preprocessing':{'numeric':'median imputation + StandardScaler(with_mean=False)','categorical':'mode imputation + OneHotEncoder(handle_unknown=ignore,min_frequency=5)','fit_scope':'training fold only'},
      'binary_models':{n:m.get_params(deep=False) for n,m in bm.items()},'family_models':{n:m.get_params(deep=False) for n,m in fm.items()},'direct_lightgbm':direct.get_params(deep=False),
      'optimized_binary_weights':binary_weights,'optimized_family_weights':family_weights,'validation_threshold':float(bopt['threshold']),'family_label_encoder_classes':le.classes_.tolist(),'direct_label_encoder_classes':le10.classes_.tolist(),'selected_features':raw_cols}
    json_dump(bestparams,OUT/'best_params.json')
    # Aggregate validation/test tables.
    allmetrics={'validation_binary_weight_optimization':bopt,'validation_family_weight_optimization':fopt,'validation_hierarchical_ten_class':hier_val,
      'test_binary':ensemble_binary,'test_multiclass_hierarchical':multiclass_metrics(ytest.to_numpy(),ptest,labels),'test_multiclass_direct':multiclass_metrics(ytest.to_numpy(),pdirect,labels),
      'test_best_single_pair':{'name':best_name,**multiclass_metrics(ytest.to_numpy(),pbest,labels)},'test_equal_weight':equal_metrics,'test_compact':compact_metrics,'paired_bootstrap_ci':boot,
      'duplicate_sensitivity':duplicate_sensitivity,'elapsed_seconds':time.perf_counter()-start,'class_order':labels}
    json_dump(allmetrics,OUT/'results/summary_metrics.json')
    elapsed=time.perf_counter()-start; log(f'EXPERIMENT COMPLETE elapsed={elapsed/60:.1f} min; official test macro-F1={allmetrics["test_multiclass_hierarchical"]["Macro_F1"]:.5f}; MCC={allmetrics["test_multiclass_hierarchical"]["MCC"]:.5f}')

if __name__=='__main__':
    try: main()
    except Exception as e:
        log('FATAL: '+repr(e)); traceback.print_exc()
        raise
