from pathlib import Path
import os
INPUT_ROOT=Path(os.environ.get('STATEFARM_INPUT_ROOT','/kaggle/input'))
_default_output='/kaggle/working/vgg_final' if Path('/kaggle/working').is_dir() else './vgg_final_outputs'
WORK_ROOT=Path(os.environ.get('STATEFARM_OUTPUT_ROOT',_default_output))
WORK_ROOT.mkdir(parents=True,exist_ok=True)
"""Select and calibrate fixed VGG CV candidates using labelled OOF data only.

No competition submission. Base OOF excludes each driver's images from its model.
Calibration crossfit is development evaluation, not full nested base-model CV.
"""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar
from scipy.special import softmax,expit
from sklearn.metrics import log_loss,accuracy_score

OUT=WORK_ROOT/'final';OUT.mkdir(parents=True,exist_ok=True)
ARTIFACT_DIRS={'full':WORK_ROOT/'full','crop':WORK_ROOT/'crop','face':WORK_ROOT/'face'}
SOURCES={key:str(value) for key,value in ARTIFACT_DIRS.items()}
def data_file(name):
    matches=set()
    for prefix in ['', '*/', '*/*/']:matches.update(INPUT_ROOT.glob(prefix+name))
    if len(matches)!=1:raise FileNotFoundError(str((name,sorted(matches))))
    return matches.pop()
CLASSES=[f'c{i}' for i in range(10)]
POLICY={'crop_weights':[0,.5,1],'face_weights':[0,.5,1],
        'global_temperature_bounds':[.5,3],'face_temperature_bounds':[.5,3],
        'class_bias_l2':.01,'bias_adoption_min_gain':.003,'minimum_folds_improved':3,
        'maximum_fold_loss_increase':.03,'candidate_min_gain':.003,
        'driver_bootstrap_lower_bound_positive':True,'seed':20261118,
        'note':'Latest CV-only final recipe. OOF chooses the mixture and calibration; no private notebook dependency or old-submission fallback.'}
(OUT/'calibration_policy.json').write_text(json.dumps(POLICY,indent=2))

def artifact(key,filename):
    path=ARTIFACT_DIRS[key]/filename
    if not path.is_file():raise FileNotFoundError(f'Missing {path}; run the corresponding training stage first.')
    return path

def loss(y,p):return float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean())

def transform(raw,q,params,use_face):
    p=softmax(np.log(np.clip(raw,1e-15,1))/params['temperature']+np.asarray(params['bias']),axis=1)
    if use_face:
        q=np.clip(q,1e-12,1-1e-12)
        q=expit((np.log(q)-np.log1p(-q))/params['face_temperature'])
        mass=p[:,0]+p[:,9]
        q=(1-float(use_face))*p[:,9]/np.clip(mass,1e-15,None)+float(use_face)*q
        p[:,0]=mass*(1-q);p[:,9]=mass*q
    return p

def fit(raw,q,y,mask,use_bias,use_face,fit_face_temperature=True):
    lp=np.log(np.clip(raw[mask],1e-15,1));labels=y[mask];face=q[mask]
    face_t=1.
    if use_face and fit_face_temperature:
        cond=np.isin(labels,[0,9]);qy=np.clip(face[cond],1e-12,1-1e-12)
        zq=np.log(qy)-np.log1p(-qy);by=(labels[cond]==9).astype(float)
        r=minimize_scalar(lambda t:float((np.logaddexp(0,zq/np.exp(t))-by*zq/np.exp(t)).mean()),
                          bounds=(np.log(.5),np.log(3)),method='bounded')
        assert r.success;face_t=float(np.exp(r.x))
    clipped=np.clip(face,1e-12,1-1e-12)
    calibrated_face=expit((np.log(clipped)-np.log1p(-clipped))/face_t)
    def objective(z):
        t=np.exp(z[0]);b=z[1:] if use_bias else np.zeros(10)
        p=softmax(lp/t+b,axis=1);target=np.eye(10)[labels]
        if use_face:
            cond=np.isin(labels,[0,9]);alpha=float(use_face);target[cond]=0
            qy=np.where(labels==9,calibrated_face,1-calibrated_face)
            coef0=alpha*qy+(1-alpha)*(labels==0)
            coef9=alpha*qy+(1-alpha)*(labels==9)
            denominator=np.clip(p[:,0]*coef0+p[:,9]*coef9,1e-15,None)
            target[cond,0]=p[cond,0]*coef0[cond]/denominator[cond]
            target[cond,9]=p[cond,9]*coef9[cond]/denominator[cond]
        g=(p-target)/len(p)
        params={'temperature':float(t),'bias':b.tolist(),'face_temperature':face_t}
        transformed=transform(raw[mask],face,params,use_face)
        value=loss(labels,transformed)
        # Match the clipped log-loss derivative for extremely confident errors.
        active=transformed[np.arange(len(labels)),labels]>1e-15
        g[~active]=0
        gradient=np.array([(g*(-lp/t)).sum()])
        if use_bias:
            value+=.005*np.square(b).sum();gradient=np.r_[gradient,g.sum(0)+.01*b]
        return value,gradient
    init=np.r_[np.log(.6),np.zeros(10)] if use_bias else np.array([np.log(.6)])
    result=minimize(objective,init,jac=True,method='L-BFGS-B',
        bounds=[(np.log(.5),np.log(3))]+([(-3,3)]*10 if use_bias else []),
        options={'ftol':1e-11,'gtol':1e-7,'maxiter':500})
    assert result.success,result.message
    return {'temperature':float(np.exp(result.x[0])),
            'bias':result.x[1:].tolist() if use_bias else [0.]*10,'face_temperature':face_t}

def crossfit(raw,q,y,groups,use_bias,use_face,fit_face_temperature=True):
    pred=np.empty_like(raw);folds=[]
    for held in sorted(set(groups)):
        mask=groups==held
        params=fit(raw,q,y,~mask,use_bias,use_face,fit_face_temperature)
        pred[mask]=transform(raw[mask],q[mask],params,use_face)
        folds.append({'fold':int(held),'n':int(mask.sum()),'log_loss':loss(y[mask],pred[mask]),'parameters':params})
    return pred,{'log_loss':loss(y,pred),'accuracy':float(accuracy_score(y,pred.argmax(1))),'folds':folds}

def driver_interval(reference,pred,y,subjects):
    old=-np.log(np.clip(reference[np.arange(len(y)),y],1e-15,1))
    new=-np.log(np.clip(pred[np.arange(len(y)),y],1e-15,1))
    names=sorted(set(subjects));n=np.array([(subjects==d).sum() for d in names])
    gain=np.array([(old-new)[subjects==d].sum() for d in names])
    ix=np.random.default_rng(20261118).integers(0,len(names),size=(10000,len(names)))
    samples=gain[ix].sum(1)/n[ix].sum(1)
    return {'lower':float(np.quantile(samples,.025)),'upper':float(np.quantile(samples,.975)),
            'drivers_improved':int((gain>0).sum()),'driver_mean_gain':dict(zip(names,(gain/n).tolist()))}

full=pd.read_csv(artifact('full','oof_predictions.csv'))
crop=pd.read_csv(artifact('crop','oof_predictions.csv'))
face=pd.read_csv(artifact('full','face_oof_predictions.csv'))
for c in ['subject','classname','img','label','fold']:
    assert full[c].tolist()==crop[c].tolist()==face[c].tolist()
assert len(full)==22424 and full.subject.nunique()==26 and not full.img.duplicated().any()
assert set(full.fold)==set(range(5)) and full.groupby('subject').fold.nunique().max()==1
official=pd.read_csv(data_file('driver_imgs_list.csv'))
for col in ['subject','classname','img']:assert official[col].tolist()==full[col].tolist()
np.testing.assert_array_equal(full.label.to_numpy(),full.classname.str[1:].astype(int).to_numpy())
for key in ['full','crop']:
    r=json.loads(artifact(key,'cv_report.json').read_text())
    assert r['every_oof_driver_excluded_from_its_training_model']
    assert r['test_labels_used'] is False and r['automatically_submitted'] is False
    for f in r['folds']:
        assert not set(f['fit_drivers'])&set(f['validation_drivers'])
        assert f['epochs_fixed_before_training']==5 and not f['validation_used_for_checkpoint_selection']
        assert set(full.loc[full.fold==f['fold'],'subject'])==set(f['validation_drivers'])
Y=full.label.to_numpy();GROUPS=full.fold.to_numpy();SUBJECTS=full.subject.to_numpy()
q=face.face_c9.to_numpy();P=full[CLASSES].to_numpy();C=crop[CLASSES].to_numpy()
assert np.isfinite(q).all() and (q>=0).all() and (q<=1).all()
for p in [P,C]:
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    np.testing.assert_allclose(p.sum(1),1,atol=1e-8)

candidates=[];selected_families=[]
for weight in POLICY['crop_weights']:
    raw=(1-weight)*P+weight*C
    for use_face in POLICY['face_weights']:
        options=[]
        for use_bias in [False,True]:
            pred,report=crossfit(raw,q,Y,GROUPS,use_bias,use_face)
            row={'crop_weight':weight,'face':use_face,'class_bias':use_bias,**report,'pred':pred}
            options.append(row);candidates.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ['pred','folds']}),flush=True)
        scalar,vector=options
        gains=[a['log_loss']-b['log_loss'] for a,b in zip(scalar['folds'],vector['folds'])]
        adopt=(vector['log_loss']<scalar['log_loss']-.003 and sum(g>0 for g in gains)>=3 and min(gains)>=-.03)
        selected_families.append(vector if adopt else scalar)
baseline=next(r for r in selected_families if r['crop_weight']==0 and r['face']==1)
eligible=[baseline]
for r in selected_families:
    interval=driver_interval(baseline['pred'],r['pred'],Y,SUBJECTS)
    r['driver_interval_vs_full_face']=interval
    gains=[a['log_loss']-b['log_loss'] for a,b in zip(baseline['folds'],r['folds'])]
    if r['log_loss']<baseline['log_loss']-.003 and sum(g>0 for g in gains)>=3 and min(gains)>=-.03 and interval['lower']>0:
        eligible.append(r)
choice=min(eligible,key=lambda r:(r['log_loss'],int(r['crop_weight']==.5),r['face']))
chosen_raw=(1-choice['crop_weight'])*P+choice['crop_weight']*C
chosen_params=fit(chosen_raw,q,Y,np.ones(len(Y),dtype=bool),choice['class_bias'],choice['face'])
chosen_recipe={k:v for k,v in choice.items() if k!='pred'}
chosen_recipe['full_fit_parameters']=chosen_params
validation=full[['subject','classname','img','label','fold']].copy();validation[CLASSES]=choice['pred']
validation.to_csv(OUT/'selected_oof_predictions.csv',index=False)
pd.DataFrame([{k:r[k] for k in ['crop_weight','face','class_bias','log_loss','accuracy']} for r in candidates]).to_csv(OUT/'candidate_scores.csv',index=False)


(OUT/'selected_recipe.json').write_text(json.dumps(chosen_recipe,indent=2))
full_fit={'recipe':chosen_recipe,'sources':SOURCES,'parameters_frozen_before_test_read':True}
(OUT/'decision_frozen_before_test_read.json').write_text(json.dumps(full_fit,indent=2))
ta=pd.read_csv(artifact('full','test_mean_raw.csv'));tc=pd.read_csv(artifact('crop','test_mean_raw.csv'))
tf=pd.read_csv(artifact('face','test_face_predictions.csv'))
official_sample=pd.read_csv(data_file('sample_submission.csv'))
assert ta.img.tolist()==tc.img.tolist()==tf.img.tolist()==official_sample.img.tolist()
assert len(ta)==79726
raw=(1-choice['crop_weight'])*ta[CLASSES].to_numpy()+choice['crop_weight']*tc[CLASSES].to_numpy()
prob=transform(raw,tf.face_c9.to_numpy(),chosen_params,choice['face'])
result=ta[['img']].copy();result[CLASSES]=prob
result.to_csv(OUT/'submission.csv',index=False,float_format='%.12g')
saved=pd.read_csv(OUT/'submission.csv');p=saved[CLASSES].to_numpy()
assert saved.columns.tolist()==['img']+CLASSES and len(saved)==79726 and not saved.img.duplicated().any()
assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
np.testing.assert_allclose(p.sum(1),1,atol=1e-9)
report={'rows':len(saved),'sha256':hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest(),
        'oof_log_loss':choice['log_loss'],'oof_accuracy':choice['accuracy'],
        'recipe':chosen_recipe,'test_labels_used':False,'automatically_submitted':False,
        'kaggle_public_score':None,'kaggle_private_score':None}
(OUT/'submission_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
