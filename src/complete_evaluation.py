import json,time,os
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scoring import named_log_loss
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,matthews_corrcoef,log_loss,precision_recall_fscore_support,confusion_matrix
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
RAW_DATA_DIR=Path(os.environ.get('UNSW_RAW_DATA_DIR', ROOT/'data'/'raw')).expanduser().resolve()
LABELS=['Normal','Analysis','Backdoors','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms']
def target(df):
 y=df.attack_cat.fillna('Normal').astype(str).str.strip(); return y.where(~y.isin(['','-','nan','None']),'Normal').replace({'Backdoor':'Backdoors','Dos':'DoS'})
def aligned(m,x,n):
 p=m.predict_proba(x); out=np.zeros((x.shape[0],len(n))); out[:,np.asarray(m.classes_,dtype=int)]=p; return out
def metrics(y,pred,p):
 return {'Macro_F1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0)),'Weighted_F1':float(f1_score(y,pred,labels=LABELS,average='weighted',zero_division=0)),'Balanced_Accuracy':float(balanced_accuracy_score(y,pred)),'MCC':float(matthews_corrcoef(y,pred)),'Log_Loss':named_log_loss(y,p,LABELS),'Accuracy':float(accuracy_score(y,pred))}
def main():
 te=pd.read_csv(RAW_DATA_DIR/'UNSW_NB15_testing-set.csv',low_memory=False); yt=target(te).to_numpy(); xraw=te.drop(columns=['id','label','attack_cat'])
 predfile=pd.read_csv(ROOT/'results/test_predictions_and_probabilities.csv.gz'); full=joblib.load(ROOT/'models/ensemble.joblib'); pre=joblib.load(ROOT/'models/preprocessor.joblib'); comp=joblib.load(ROOT/'models/compact_ensemble.joblib'); pc=joblib.load(ROOT/'models/compact_preprocessor.joblib')
 x=pre.transform(xraw); xc=pc.transform(xraw[comp['selected_features']]); nclass=list(range(9)); t0=time.perf_counter()
 def infer(b,xx):
  pb=sum(float(b['binary_weights'][n])*m.predict_proba(xx)[:,1] for n,m in b['binary_models'].items()); pf=sum(float(b['family_weights'][n])*aligned(m,xx,nclass) for n,m in b['family_models'].items()); pf/=np.maximum(pf.sum(axis=1,keepdims=True),1e-15); return pb,pf
 pb,pf=infer(comp,xc); threshold=float(comp['threshold']); fam=np.asarray(LABELS[1:],dtype=object)[np.argmax(pf,axis=1)]; yp=np.where(pb>=threshold,fam,'Normal'); prob=np.column_stack([1-pb,pb[:,None]*pf]); cm=metrics(yt,yp,prob)
 cols={'true_label':yt,'predicted_label':yp,'predicted_binary_attack_probability':pb,'binary_threshold':threshold}
 for i,c in enumerate(LABELS): cols['prob_'+c]=prob[:,i]
 pd.DataFrame(cols).to_csv(ROOT/'results/compact_test_predictions_and_probabilities.csv.gz',index=False,compression='gzip')
 pd.DataFrame([{'model':'Full hierarchical ensemble',**metrics(yt,predfile.predicted_label.to_numpy(),predfile[[f'prob_{c}' for c in LABELS]].to_numpy())},{'model':'Compact SHAP-stable ensemble',**cm}]).to_csv(ROOT/'results/full_vs_compact_metrics.csv',index=False)
 pcl=precision_recall_fscore_support(yt,yp,labels=LABELS,zero_division=0); pd.DataFrame({'class':LABELS,'precision':pcl[0],'recall':pcl[1],'f1':pcl[2],'support':pcl[3]}).to_csv(ROOT/'results/compact_per_class_metrics.csv',index=False)
 pd.DataFrame(confusion_matrix(yt,yp,labels=LABELS),index=LABELS,columns=LABELS).to_csv(ROOT/'results/compact_confusion_matrix.csv')
 checks={'sample_rows':128,'full_bundle_reloaded':True,'compact_bundle_reloaded':True,'full_binary_probability_shape':list(full['binary_models']['LightGBM'].predict_proba(x[:128]).shape),'compact_binary_probability_shape':list(comp['binary_models']['LightGBM'].predict_proba(xc[:128]).shape),'full_family_probability_shape':list(full['family_models']['LightGBM'].predict_proba(x[:128]).shape),'compact_family_probability_shape':list(comp['family_models']['LightGBM'].predict_proba(xc[:128]).shape),'all_probabilities_finite':bool(np.isfinite(prob[:128]).all()),'probability_rows_sum_to_one':bool(np.allclose(prob[:128].sum(axis=1),1,atol=1e-5))}
 pb128,pf128=infer(full,x[:128]); full_prob128=np.column_stack([1-pb128,pb128[:,None]*pf128])
 checks.update({'class_order_correct':full['class_order']==LABELS and comp['class_order']==LABELS,'full_probabilities_finite':bool(np.isfinite(full_prob128).all()),'full_probability_rows_sum_to_one':bool(np.allclose(full_prob128.sum(axis=1),1,atol=1e-5))})
 if not all(checks[k] for k in ('all_probabilities_finite','probability_rows_sum_to_one','class_order_correct','full_probabilities_finite','full_probability_rows_sum_to_one')): raise RuntimeError('Reloaded model probability or class-order validation failed')
 (ROOT/'results/model_reload_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
 # Average latency across three runs on the first 10,000 test rows; sparse matrices remain sparse.
 def bench(bundle,xx):
  n=min(10000,xx.shape[0]); z=xx[:n]; times=[]
  for _ in range(3):
   a=time.perf_counter(); infer(bundle,z); times.append(time.perf_counter()-a)
  return float(np.mean(times)/n*1000)
 tf,tc=bench(full,x),bench(comp,xc)
 eff=json.loads((ROOT/'results/full_vs_compact_efficiency.json').read_text()); eff.update({'full_inference_seconds_per_1000_rows':tf,'compact_inference_seconds_per_1000_rows':tc,'full_model_bundle_bytes':(ROOT/'models/ensemble.joblib').stat().st_size,'compact_model_bundle_bytes':(ROOT/'models/compact_ensemble.joblib').stat().st_size,'full_macro_f1':metrics(yt,predfile.predicted_label.to_numpy(),predfile[[f'prob_{c}' for c in LABELS]].to_numpy())['Macro_F1'],'compact_macro_f1':cm['Macro_F1'],'full_mcc':metrics(yt,predfile.predicted_label.to_numpy(),predfile[[f'prob_{c}' for c in LABELS]].to_numpy())['MCC'],'compact_mcc':cm['MCC']}); (ROOT/'results/full_vs_compact_efficiency.json').write_text(json.dumps(eff,indent=2),encoding='utf-8')
 ab=pd.read_csv(ROOT/'results/ablation_results.csv'); ab=ab[ab.ablation!='Compact SHAP-stable ensemble']; ab=pd.concat([ab,pd.DataFrame([{'ablation':'Compact SHAP-stable ensemble','Macro_F1':cm['Macro_F1'],'MCC':cm['MCC']}])],ignore_index=True); ab.to_csv(ROOT/'results/ablation_results.csv',index=False)
 compare=pd.read_csv(ROOT/'results/model_comparison.csv'); compare=compare[compare.model!='Compact SHAP-stable ensemble']; compare=pd.concat([compare,pd.DataFrame([{'model':'Compact SHAP-stable ensemble',**cm}])],ignore_index=True); compare.to_csv(ROOT/'results/model_comparison.csv',index=False)
 dup=json.loads((ROOT/'results/duplicate_sensitivity.json').read_text()); summ=json.loads((ROOT/'results/summary_metrics.json').read_text()); summ.update({'test_multiclass_hierarchical':metrics(yt,predfile.predicted_label.to_numpy(),predfile[[f'prob_{c}' for c in LABELS]].to_numpy()),'test_multiclass_direct':next((r for r in compare.to_dict('records') if r['model']=='Direct LightGBM 10-class'),{}),'test_compact':cm,'test_equal_weight':next((r for r in compare.to_dict('records') if r['model']=='Equal-weight hierarchical ensemble'),{}),'test_best_single_pair':next((r for r in compare.to_dict('records') if r['model'].startswith('Best single pair')),{}),'paired_bootstrap_ci':json.loads((ROOT/'results/paired_bootstrap_ci.json').read_text()),'duplicate_sensitivity':dup})
 (ROOT/'results/summary_metrics.json').write_text(json.dumps(summ,indent=2),encoding='utf-8')
 print(json.dumps({'compact_metrics':cm,'full_latency_s_per_1000':tf,'compact_latency_s_per_1000':tc,'reload':checks},indent=2))
if __name__=='__main__': main()
