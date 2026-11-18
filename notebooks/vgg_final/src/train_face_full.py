from pathlib import Path
import os
INPUT_ROOT=Path(os.environ.get('STATEFARM_INPUT_ROOT','/kaggle/input'))
_default_output='/kaggle/working/vgg_final' if Path('/kaggle/working').is_dir() else './vgg_final_outputs'
WORK_ROOT=Path(os.environ.get('STATEFARM_OUTPUT_ROOT',_default_output))
WORK_ROOT.mkdir(parents=True,exist_ok=True)
import gc,hashlib,json,math,random,time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
START=time.perf_counter();INPUT=INPUT_ROOT;OUT=WORK_ROOT/'face';OUT.mkdir(parents=True,exist_ok=True)
SEED=42;MAX_EPOCHS=6
assert torch.cuda.is_available(), 'A CUDA GPU is required for training.'
DEVICE=torch.device('cuda:0')
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
normalize=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
augment=transforms.Compose([transforms.Resize((224,224)),
    transforms.RandomAffine(degrees=5,translate=(.04,.04),scale=(.95,1.05),fill=(123,116,103)),
    transforms.ColorJitter(brightness=.15,contrast=.15,saturation=.1),transforms.ToTensor(),normalize])
plain=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])

def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def emit(event,**fields):print(json.dumps({'event':event,'arch':'vgg16_face',**fields}),flush=True)

def find_one(pattern):
    found=set()
    for prefix in ('','*/','*/*/'):found.update(INPUT.glob(prefix+pattern))
    if len(found)!=1:raise FileNotFoundError(str(found))
    return next(iter(found))

class Faces(Dataset):
    def __init__(self,frame,training=False,test=False):self.frame=frame.reset_index(drop=True);self.training=training;self.test=test
    def __len__(self):return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        path=DATA/'imgs/test'/row['img'] if self.test else DATA/'imgs/train'/row['classname']/row['img']
        with Image.open(path) as original:
            image=original.convert('RGB');w,h=image.size
            image=image.crop((int(.2*w),0,int(.7*w),int(.65*h)))
            image=(augment if self.training else plain)(image)
        return image,index if self.test else int(row['label']==9)

def seed_worker(worker):
    s=torch.initial_seed()%(2**32);random.seed(s);np.random.seed(s)

def loader(frame,training=False,test=False):
    return DataLoader(Faces(frame,training,test),batch_size=64 if test else 32,shuffle=training,
        num_workers=4,pin_memory=True,persistent_workers=False,worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(SEED))

class FaceVGG(nn.Module):
    def __init__(self,features):
        super().__init__();self.features=features;self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(nn.Dropout(.3),nn.Linear(512,2))
    def forward(self,x):return self.classifier(self.pool(self.features(x)).flatten(1))

def make_model():
    seed_all();pre=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    core=FaceVGG(pre.features).to(DEVICE);del pre;gc.collect()
    return core,nn.DataParallel(core) if torch.cuda.device_count()>1 else core

def configure(core,model,optimizer,epoch):
    model.train();core.features.requires_grad_(False)
    if epoch==1:rates=[0,1e-3]
    else:
        core.features[17:].requires_grad_(True)
        factor=.2+.8*.5*(1+math.cos(math.pi*(epoch-2)/max(1,MAX_EPOCHS-2)))
        rates=[3e-5*factor,2e-4*factor]
    for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate

def optimizer_for(core):
    return torch.optim.AdamW([{'params':list(core.features[17:].parameters()),'lr':0},
        {'params':core.classifier.parameters(),'lr':1e-3}],weight_decay=1e-4)

def train_epoch(model,data,optimizer,scaler):
    loss_fn=nn.CrossEntropyLoss(label_smoothing=.02);n=correct=0;loss_sum=0.
    for images,labels in data:
        images=images.to(DEVICE,non_blocking=True);labels=labels.to(DEVICE,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):logits=model(images)
        loss=loss_fn(logits.float(),labels)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite face loss')
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        n+=len(labels);correct+=(logits.argmax(1)==labels).sum().item();loss_sum+=loss.item()*len(labels)
    return loss_sum/n,correct/n

def predict(model,data,test=False):
    model.eval();q=[];indices=[]
    with torch.inference_mode():
        for batch,(images,index) in enumerate(data):
            images=images.to(DEVICE,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):logits=model(images)
            q.append(logits.double().softmax(1)[:,1].cpu().numpy());indices.append(index.numpy())
            if test and (batch+1)%200==0:emit('test_progress',completed_images=sum(len(x) for x in q))
    return np.concatenate(q),np.concatenate(indices)
metadata_path=find_one('driver_imgs_list.csv');DATA=metadata_path.parent
metadata=pd.read_csv(metadata_path);metadata['label']=metadata.classname.str[1:].astype(int)
assert len(metadata)==22424 and metadata.subject.nunique()==26
binary_full=metadata[metadata.label.isin([0,9])].reset_index(drop=True)
core,model=make_model();optimizer=optimizer_for(core);scaler=torch.amp.GradScaler('cuda')
training=loader(binary_full,True)
for epoch in [1,2]:
    configure(core,model,optimizer,epoch)
    loss,accuracy=train_epoch(model,training,optimizer,scaler)
    emit('full_refit_epoch',epoch=epoch,train_loss=loss,train_accuracy=accuracy)
report={'training_images':len(binary_full),'training_drivers':26,'epochs':2,'temperature_applied':1.,'output_probabilities':'raw'}
torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in core.state_dict().items()},'metadata':report},OUT/'full_face_model.pt')
core.requires_grad_(False);core.zero_grad(set_to_none=True)
del optimizer,scaler,training;gc.collect();torch.cuda.empty_cache()
sample=pd.read_csv(find_one('sample_submission.csv'))
q,indices=predict(model,loader(sample,test=True),True)
np.testing.assert_array_equal(indices,np.arange(len(sample)))
assert np.isfinite(q).all() and (q>=0).all() and (q<=1).all()
pd.DataFrame({'img':sample.img,'face_c9':q}).to_csv(OUT/'test_face_predictions.csv',index=False,float_format='%.12g')
(OUT/'full_training_report.json').write_text(json.dumps(report,indent=2))
