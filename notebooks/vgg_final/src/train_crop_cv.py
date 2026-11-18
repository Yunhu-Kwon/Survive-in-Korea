from pathlib import Path
import os
INPUT_ROOT=Path(os.environ.get('STATEFARM_INPUT_ROOT','/kaggle/input'))
_default_output='/kaggle/working/vgg_final' if Path('/kaggle/working').is_dir() else './vgg_final_outputs'
WORK_ROOT=Path(os.environ.get('STATEFARM_OUTPUT_ROOT',_default_output))
WORK_ROOT.mkdir(parents=True,exist_ok=True)
CV_CONFIG = {'driver_folds': {'p002': 0, 'p026': 0, 'p035': 0, 'p041': 0, 'p050': 0, 'p075': 0, 'p081': 1, 'p051': 1, 'p015': 1, 'p024': 1, 'p012': 1, 'p014': 2, 'p056': 2, 'p052': 2, 'p016': 2, 'p061': 2, 'p064': 3, 'p047': 3, 'p072': 3, 'p022': 3, 'p045': 3, 'p066': 4, 'p039': 4, 'p049': 4, 'p021': 4, 'p042': 4}, 'metadata_sha256': '891ca8e2bfd7a3522ac2098a1bc93825c581d07104d211c329de0b16c83ca5ee', 'fixed_epochs': 5, 'schedule_epochs': 9, 'fixed_variant': 'EMA .995; BN refreshed on fit drivers only', 'face_fixed_epochs': 2, 'family_comparison': 'full image vs fixed driver crop; same VGG16-BN GAP and optimizer', 'crop_fractions': [0.15, 0, 0.85, 0.9], 'fixed_combination_choices': {'global_crop_weights': [0, 0.5, 1], 'face_weights': [0, 1], 'calibration': ['scalar_temperature', 'regularized_class_bias'], 'class_bias_l2': 0.01}, 'notes': ['Each OOF image is predicted only by a model excluding its driver.', 'Fixed epoch configuration was informed by prior experiments on the original fold0; results remain development estimates.', 'Calibrator crossfit is not a full nested re-training of base models.', 'Test predictions are transformed only after validation-based choices; no pseudo labels or test fitting.'], 'new_competition_submissions_authorized': 0, 'family': 'crop', 'face_oof': False}
DRIVER_CROP = [0.15, 0, 0.85, 0.9]
# %% Configuration and data
import copy
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from scipy.optimize import minimize_scalar

ARCH = 'vgg16_bn_gap_driver_cv'
SEED = 42
MAX_EPOCHS = 9
PATIENCE = 3
BATCH_SIZE = 64
WORKERS = 4
LABEL_SMOOTHING = 0.03
EMA_DECAY = 0.995
INPUT = INPUT_ROOT
OUT = WORK_ROOT/'crop';OUT.mkdir(parents=True,exist_ok=True)
CLASSES = [f'c{i}' for i in range(10)]
STARTED = time.perf_counter()
if not torch.cuda.is_available():
    raise RuntimeError('GPU is required.')
DEVICE = torch.device('cuda:0')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def emit(event, **fields):
    print(json.dumps({'event':event, 'arch':ARCH, **fields}), flush=True)

def find_input(name):
    paths=set()
    for prefix in ('','*/','*/*/'):
        paths.update(INPUT.glob(prefix+name))
    if len(paths)!=1:
        raise FileNotFoundError(f'Expected one {name}: {sorted(paths)}')
    return next(iter(paths))

metadata_path=find_input('driver_imgs_list.csv')
assert hashlib.sha256(metadata_path.read_bytes()).hexdigest()==CV_CONFIG['metadata_sha256']
metadata=pd.read_csv(metadata_path)
metadata['label']=metadata['classname'].str[1:].astype(int)
assert set(metadata['label'])==set(range(10)) and not metadata['img'].duplicated().any()
DATA_ROOT=metadata_path.parent
TRAIN_ROOT=DATA_ROOT/'imgs/train'
TEST_ROOT=DATA_ROOT/'imgs/test'
assert TRAIN_ROOT.is_dir() and TEST_ROOT.is_dir()
normalize=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
train_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomAffine(degrees=8,translate=(.05,.05),scale=(.9,1.1),fill=(123,116,103)),
    transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.1),
    transforms.ToTensor(),normalize])
eval_transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])

class DriverDataset(Dataset):
    def __init__(self,frame,training=False,test=False):
        self.frame=frame.reset_index(drop=True)
        self.transform=train_transform if training else eval_transform
        self.test=test
    def __len__(self):
        return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        path=TEST_ROOT/row['img'] if self.test else TRAIN_ROOT/row['classname']/row['img']
        with Image.open(path) as image:
            image=image.convert('RGB')
            if DRIVER_CROP is not None:
                w,h=image.size
                a,b,c,d=DRIVER_CROP
                image=image.crop((int(a*w),int(b*h),int(c*w),int(d*h)))
            image=self.transform(image)
        return image,index if self.test else int(row['label'])

def seed_worker(worker_id):
    seed=torch.initial_seed()%(2**32)
    random.seed(seed); np.random.seed(seed)

def loader(frame,training=False,test=False):
    return DataLoader(DriverDataset(frame,training,test),batch_size=BATCH_SIZE,shuffle=training,
        num_workers=WORKERS,pin_memory=True,persistent_workers=False,worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(SEED))

def prediction_frame(frame,prob):
    result=frame[['subject','classname','img','label']].copy()
    result[CLASSES]=prob
    return result

# %% Model, optimization and inference
class VGG16BNGAP(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.features=features
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(nn.Identity(),nn.Identity(),nn.Identity(),nn.Identity(),
                                      nn.Identity(),nn.Dropout(.3),nn.Linear(512,10))
    def forward(self,x):
        return self.classifier(self.pool(self.features(x)).flatten(1))

def make_model():
    seed_all(SEED)
    pretrained=models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
    core=VGG16BNGAP(pretrained.features).to(DEVICE)
    del pretrained
    gc.collect()
    wrapped=nn.DataParallel(core) if torch.cuda.device_count()>1 else core
    return core,wrapped,24

def make_optimizer(core,freeze_until):
    return torch.optim.AdamW([
        {'params':list(core.features[freeze_until:].parameters()),'lr':0},
        {'params':list(core.classifier[:6].parameters()),'lr':3e-5},
        {'params':list(core.classifier[6].parameters()),'lr':3e-4}],weight_decay=1e-4)

def configure_epoch(core,model,optimizer,freeze_until,epoch):
    model.train()
    core.features.requires_grad_(False)
    if epoch==1:
        core.features.eval()
        rates=[0,0,1e-3]
    else:
        core.features[freeze_until:].requires_grad_(True)
        core.features[:freeze_until].eval()
        phase=(epoch-2)/max(1,MAX_EPOCHS-2)
        factor=.1+.9*.5*(1+math.cos(math.pi*phase))
        rates=[3e-5*factor,0,3e-4*factor]
    for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate
    return rates

@torch.no_grad()
def update_ema(ema,core):
    for average,current in zip(ema.parameters(),core.parameters()):
        average.lerp_(current.detach(),1-EMA_DECAY)
    for average,current in zip(ema.buffers(),core.buffers()):
        average.copy_(current.detach())

def train_epoch(core,model,optimizer,scaler,train_loader,ema,stage,epoch):
    loss_fn=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    total=correct=0
    loss_sum=0.
    for batch,(images,targets) in enumerate(train_loader):
        images=images.to(DEVICE,non_blocking=True); targets=targets.to(DEVICE,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):logits=model(images)
        loss=loss_fn(logits.float(),targets)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss.')
        scaler.scale(loss).backward()
        if epoch==2 and batch==0:
            convs=[m for m in core.features if isinstance(m,nn.Conv2d)]
            assert convs[-1].weight.grad is not None and convs[0].weight.grad is None
            emit('gradient_check_passed',stage=stage)
        scaler.step(optimizer);scaler.update()
        if ema is not None:update_ema(ema,core)
        total+=len(targets); correct+=(logits.argmax(1)==targets).sum().item()
        loss_sum+=loss.item()*len(targets)
        if (batch+1)%100==0:
            emit('training_progress',stage=stage,epoch=epoch,batch=batch+1,batches=len(train_loader),train_accuracy=correct/total)
    return loss_sum/total,correct/total

def predict(model,data_loader,views=1,test=False):
    model.eval()
    results=[]; targets=[]
    offsets=[0] if views==1 else [0,-4,4]
    with torch.inference_mode():
        for batch,(images,label) in enumerate(data_loader):
            images=images.to(DEVICE,non_blocking=True)
            if views>1: padded=nn.functional.pad(images,(4,4,0,0),value=0)
            summed=None
            for offset in offsets:
                current=images if views==1 else padded[:,:,:,4+offset:4+offset+224].contiguous()
                with torch.autocast('cuda',dtype=torch.float16):logits=model(current)
                prob=logits.double().softmax(1)
                summed=prob if summed is None else summed+prob
            results.append((summed/len(offsets)).cpu().numpy())
            targets.append(label.numpy())
            if test and (batch+1)%200==0:
                emit('test_progress',completed_images=sum(len(x) for x in results),total_images=len(data_loader.dataset),views=views)
    results=np.concatenate(results);targets=np.concatenate(targets)
    assert np.isfinite(results).all()
    np.testing.assert_allclose(results.sum(axis=1),1,atol=1e-8)
    return targets,results

def evaluate(model,data_loader,views=1):
    targets,prob=predict(model,data_loader,views)
    np.testing.assert_array_equal(targets,val['label'].to_numpy())
    return float(log_loss(targets,prob,labels=list(range(10)))),float(accuracy_score(targets,prob.argmax(1))),prob

def save_model(core,path,report):
    state={k:v.detach().cpu() for k,v in core.state_dict().items()}
    torch.save({'model_state_dict':state,'arch':ARCH,'metadata':report},path)

def temperature_transform(prob,temperature):
    z=np.log(np.clip(prob,1e-15,1))/temperature
    z-=z.max(axis=1,keepdims=True)
    z=np.exp(z)
    return z/z.sum(axis=1,keepdims=True)

def calibrate(prob):
    labels=val['label'].to_numpy()
    def fit(mask):
        result=minimize_scalar(lambda z:log_loss(labels[mask],temperature_transform(prob[mask],np.exp(z)),labels=list(range(10))),
            bounds=(np.log(.5),np.log(3)),method='bounded')
        return float(np.exp(result.x))
    drivers=sorted(val['subject'].unique())
    crossfit=np.zeros_like(prob);folds=[]
    for held_out in (drivers[::2],drivers[1::2]):
        test_mask=val['subject'].isin(held_out).to_numpy();train_mask=~test_mask
        temperature=fit(train_mask)
        crossfit[test_mask]=temperature_transform(prob[test_mask],temperature)
        folds.append({'evaluation_drivers':held_out,'temperature':temperature,
            'raw_log_loss':float(log_loss(labels[test_mask],prob[test_mask],labels=list(range(10)))),
            'calibrated_log_loss':float(log_loss(labels[test_mask],crossfit[test_mask],labels=list(range(10))))})
    raw_score=float(log_loss(labels,prob,labels=list(range(10))))
    crossfit_score=float(log_loss(labels,crossfit,labels=list(range(10))))
    enabled=crossfit_score<raw_score-.003 and all(f['calibrated_log_loss']<=f['raw_log_loss']+.02 for f in folds)
    temperature=fit(np.ones(len(val),dtype=bool)) if enabled else 1.
    if not enabled:crossfit=prob.copy();crossfit_score=raw_score
    return temperature,crossfit,{'enabled':enabled,'temperature':temperature,'raw_log_loss':raw_score,
        'crossfit_log_loss':crossfit_score,'folds':folds}


"""Fixed-epoch, driver-disjoint out-of-fold training and test averaging.

There is no epoch/weight selection using the current fold's labels. The fixed
5-epoch EMA configuration was informed by earlier development trials, including
the original six-driver fold, so these are still development estimates.
"""
FOLD_IDS=metadata.subject.map(CV_CONFIG['driver_folds']).to_numpy()
assert set(FOLD_IDS)==set(range(5)) and len(metadata)==22424
assert metadata.subject.nunique()==26
EPOCHS=5
CLASSES=[f'c{i}' for i in range(10)]
oof=np.full((len(metadata),10),np.nan)
seen=np.zeros(len(metadata),dtype=np.int8)
face_oof=np.full(len(metadata),np.nan)
sample=pd.read_csv(find_input('sample_submission.csv'))
assert sample.columns.tolist()==['img']+CLASSES and len(sample)==79726
test_sum=np.zeros((len(sample),10),dtype=np.float64)
manifest=metadata.copy();manifest['fold']=FOLD_IDS
manifest.to_csv(OUT/'fold_manifest.csv',index=False)
(OUT/'cv_protocol.json').write_text(json.dumps(CV_CONFIG,indent=2))

class FaceData(Dataset):
    def __init__(self,frame,training=False):
        self.frame=frame.reset_index(drop=True);self.training=training
        self.transform=transforms.Compose([transforms.Resize((224,224)),
            transforms.RandomAffine(degrees=5,translate=(.04,.04),scale=(.95,1.05),fill=(123,116,103)),
            transforms.ColorJitter(brightness=.15,contrast=.15,saturation=.1),
            transforms.ToTensor(),normalize]) if training else eval_transform
    def __len__(self):return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        with Image.open(TRAIN_ROOT/row.classname/row.img) as image:
            image=image.convert('RGB');w,h=image.size
            image=image.crop((int(.2*w),0,int(.7*w),int(.65*h)))
            image=self.transform(image)
        return image,int(row.label==9)

def face_loader(frame,training=False):
    return DataLoader(FaceData(frame,training),batch_size=32 if training else 64,shuffle=training,
        num_workers=4,pin_memory=True,worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(SEED))

class FaceVGG(nn.Module):
    def __init__(self,features):
        super().__init__();self.features=features
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Sequential(nn.Dropout(.3),nn.Linear(512,2))
    def forward(self,x):return self.classifier(self.pool(self.features(x)).flatten(1))

def fit_face(train_frame,val_frame,fold_dir,fold):
    seed_all(SEED)
    pre=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    core=FaceVGG(pre.features).to(DEVICE);del pre;gc.collect()
    model=nn.DataParallel(core) if torch.cuda.device_count()>1 else core
    opt=torch.optim.AdamW([{'params':list(core.features[17:].parameters()),'lr':0},
                          {'params':core.classifier.parameters(),'lr':1e-3}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    binary=train_frame[train_frame.label.isin([0,9])].reset_index(drop=True)
    assert not set(binary.subject)&set(val_frame.subject)
    dl=face_loader(binary,True)
    for epoch in [1,2]:
        model.train();core.features.requires_grad_(False)
        if epoch==2:core.features[17:].requires_grad_(True)
        rates=[0,1e-3] if epoch==1 else [3e-5,2e-4]
        for group,lr in zip(opt.param_groups,rates):group['lr']=lr
        n=0;total=0.
        for x,y in dl:
            x=x.to(DEVICE,non_blocking=True);y=y.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):z=model(x)
            loss=nn.functional.cross_entropy(z.float(),y,label_smoothing=.02)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            n+=len(y);total+=float(loss)*len(y)
        emit('face_epoch',fold=fold,epoch=epoch,images=n,loss=total/n)
    model.eval();qs=[]
    with torch.inference_mode():
        for x,y in face_loader(val_frame):
            with torch.autocast('cuda',dtype=torch.float16):z=model(x.to(DEVICE,non_blocking=True))
            qs.append(z.double().softmax(1)[:,1].cpu().numpy())
    q=np.concatenate(qs)
    assert len(q)==len(val_frame) and np.isfinite(q).all()
    frame=val_frame[['subject','classname','img','label']].copy();frame['face_c9']=q
    frame.to_csv(fold_dir/'face_oof_predictions.csv',index=False)
    torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in core.state_dict().items()},
                'fold':fold,'epochs':2,'train_drivers':sorted(binary.subject.unique().tolist())},fold_dir/'face_model.pt')
    del core,model,opt,scaler,dl;gc.collect();torch.cuda.empty_cache()
    return q

reports=[]
emit('cv_started',family=CV_CONFIG['family'],folds=5,epochs=EPOCHS,
     training_schedule_epochs=MAX_EPOCHS,crop=DRIVER_CROP,face_oof=CV_CONFIG['face_oof'])
for fold in range(5):
    start=time.perf_counter();SEED=42+1000*fold
    fit_indices=np.flatnonzero(FOLD_IDS!=fold);val_indices=np.flatnonzero(FOLD_IDS==fold)
    train=metadata.iloc[fit_indices].reset_index(drop=True)
    val=metadata.iloc[val_indices].reset_index(drop=True)
    assert not set(train.subject)&set(val.subject)
    dest=OUT/f'fold_{fold}';dest.mkdir(exist_ok=True)
    train.to_csv(dest/'train_manifest.csv',index=False);val.to_csv(dest/'validation_manifest.csv',index=False)
    core,model,freeze_until=make_model();opt=make_optimizer(core,freeze_until)
    scaler=torch.amp.GradScaler('cuda');dl=loader(train,training=True);ema=None;history=[]
    emit('fold_started',family=CV_CONFIG['family'],fold=fold,fit_images=len(train),val_images=len(val),
         fit_drivers=sorted(train.subject.unique().tolist()),val_drivers=sorted(val.subject.unique().tolist()))
    for epoch in range(1,EPOCHS+1):
        epoch_start=time.perf_counter()
        configure_epoch(core,model,opt,freeze_until,epoch)
        loss,accuracy=train_epoch(core,model,opt,scaler,dl,ema,f'fold_{fold}',epoch)
        if epoch==1:ema=copy.deepcopy(core).requires_grad_(False).eval()
        row={'fold':fold,'epoch':epoch,'train_loss':loss,'train_accuracy':accuracy,'seconds':time.perf_counter()-epoch_start}
        history.append(row);pd.DataFrame(history).to_csv(dest/'training_history.csv',index=False)
        emit('cv_epoch',family=CV_CONFIG['family'],**row)
    core.load_state_dict(ema.state_dict())
    emit('bn_refresh_started',family=CV_CONFIG['family'],fold=fold,data='fit drivers only')
    from torch.optim.swa_utils import update_bn
    update_bn(loader(train),core,device=DEVICE)
    save_model(core,dest/'global_model.pt',{'fold':fold,'config':CV_CONFIG,'epochs':EPOCHS,'variant':'ema',
                                         'train_drivers':sorted(train.subject.unique().tolist())})
    core.requires_grad_(False);core.zero_grad(set_to_none=True)
    del opt,scaler,dl,ema;gc.collect();torch.cuda.empty_cache()
    labels,prob=predict(model,loader(val))
    np.testing.assert_array_equal(labels,val.label.to_numpy())
    assert not seen[val_indices].any()
    oof[val_indices]=prob;seen[val_indices]+=1
    frame=prediction_frame(val,prob);frame['fold']=fold;frame.to_csv(dest/'oof_predictions.csv',index=False)
    row={'fold':fold,'fit_drivers':sorted(train.subject.unique().tolist()),'validation_drivers':sorted(val.subject.unique().tolist()),
         'validation_log_loss':float(log_loss(val.label,prob,labels=list(range(10)))),
         'validation_accuracy':float(accuracy_score(val.label,prob.argmax(1))),
         'epochs_fixed_before_training':EPOCHS,'validation_used_for_checkpoint_selection':False}
    emit('fold_validation',family=CV_CONFIG['family'],**row)
    indices,test_prob=predict(model,loader(sample,test=True),test=True)
    np.testing.assert_array_equal(indices,np.arange(len(sample)))
    test_sum+=test_prob
    test_frame=sample[['img']].copy();test_frame[CLASSES]=test_prob
    test_frame.to_csv(dest/'test_raw.csv',index=False,float_format='%.12g')
    del core,model,test_prob;gc.collect();torch.cuda.empty_cache()
    if CV_CONFIG['face_oof']:face_oof[val_indices]=fit_face(train,val,dest,fold)
    row['seconds']=time.perf_counter()-start;reports.append(row)
    (OUT/'fold_reports.json').write_text(json.dumps(reports,indent=2))
    emit('fold_complete',family=CV_CONFIG['family'],**row)
assert (seen==1).all() and np.isfinite(oof).all()
np.testing.assert_allclose(oof.sum(1),1,atol=1e-8)
result=metadata[['subject','classname','img','label']].copy();result['fold']=FOLD_IDS;result[CLASSES]=oof
result.to_csv(OUT/'oof_predictions.csv',index=False)
test_frame=sample[['img']].copy();test_frame[CLASSES]=test_sum/5
test_frame.to_csv(OUT/'test_mean_raw.csv',index=False,float_format='%.12g')
if CV_CONFIG['face_oof']:
    assert np.isfinite(face_oof).all() and (face_oof>=0).all() and (face_oof<=1).all()
    face_frame=metadata[['subject','classname','img','label']].copy();face_frame['fold']=FOLD_IDS;face_frame['face_c9']=face_oof
    face_frame.to_csv(OUT/'face_oof_predictions.csv',index=False)
report={'family':CV_CONFIG['family'],'oof_images':len(metadata),'drivers':int(metadata.subject.nunique()),
        'oof_log_loss':float(log_loss(metadata.label,oof,labels=list(range(10)))),
        'oof_accuracy':float(accuracy_score(metadata.label,oof.argmax(1))),
        'test_images':len(sample),'test_models_averaged':5,'folds':reports,
        'every_oof_driver_excluded_from_its_training_model':True,
        'test_labels_used':False,'automatically_submitted':False,'config':CV_CONFIG,
        'elapsed_seconds':time.perf_counter()-STARTED}
(OUT/'cv_report.json').write_text(json.dumps(report,indent=2))
emit('cv_complete',**report)
