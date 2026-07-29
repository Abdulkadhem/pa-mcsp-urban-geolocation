"""Evaluate fixed-alpha StreetCLIP patch fusion on the official test split.

The alpha is fixed to 0.75 from the nested development evaluation; no official
test labels are used for tuning. Existing gallery patch descriptors are reused
and only missing gallery rows plus official query images are encoded.
"""
from __future__ import annotations
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
from urban_s2.cells import build_adaptive_s2_partition
from urban_s2.geodesy import unit_to_latlon
from urban_s2.pa_mcsp import topk_cosine

def geo_km(a, b):
    a=np.asarray(a,float); b=np.asarray(b,float); lat1,lon1=np.radians(a[...,0]),np.radians(a[...,1]); lat2,lon2=np.radians(b[...,0]),np.radians(b[...,1]); dlat,dlon=lat2-lat1,lon2-lon1; h=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2; return 6371.0088*2*np.arcsin(np.sqrt(np.clip(h,0,1)))

def summary(e):
    e=np.asarray(e,float); return {"queries":int(len(e)),"accuracy_1km":float(np.mean(e<=1)),"accuracy_25km":float(np.mean(e<=25)),"accuracy_200km":float(np.mean(e<=200)),"median_error_km":float(np.median(e)),"mean_error_km":float(np.mean(e)),"p90_error_km":float(np.quantile(e,.9)),"geoscore":float(np.mean(np.exp(-e/1492.7)))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-features',type=Path,required=True); ap.add_argument('--train-index',type=Path,required=True); ap.add_argument('--test-features',type=Path,required=True); ap.add_argument('--test-index',type=Path,required=True); ap.add_argument('--gallery-desc',type=Path,required=True); ap.add_argument('--gallery-desc-rows',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--k',type=int,default=16); ap.add_argument('--batch',type=int,default=256); ap.add_argument('--extract-batch',type=int,default=32); ap.add_argument('--alpha',type=float,default=.75); ap.add_argument('--workers',type=int,default=12); args=ap.parse_args()
    t0=time.perf_counter(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    train_idx=pd.read_parquet(args.train_index).sort_values('feature_row').reset_index(drop=True); test_idx=pd.read_parquet(args.test_index).reset_index(drop=True)
    train_feat=np.load(args.train_features,mmap_mode='r'); test_feat=np.asarray(np.load(args.test_features,mmap_mode='r'),dtype=np.float32)
    train_mask=train_idx['split'].astype(str).eq('train').to_numpy(); tr=train_idx.loc[train_mask].reset_index(drop=True); tr_feat=np.asarray(train_feat[train_mask],dtype=np.float32)
    part=build_adaptive_s2_partition(tr.latitude.to_numpy(),tr.longitude.to_numpy(),min_level=3,max_level=12,max_samples_per_leaf=64)
    sums=np.zeros((len(part.leaf_counts),tr_feat.shape[1]),np.float64); np.add.at(sums,part.labels,tr_feat.astype(np.float64)); proto=(sums/np.maximum(np.linalg.norm(sums,axis=1,keepdims=True),1e-12)).astype(np.float32)
    q=torch.from_numpy(test_feat).to(dev); pp=topk_cosine(q,torch.from_numpy(proto).to(dev),top_k=args.k,batch_size=args.batch); rp=topk_cosine(q,torch.from_numpy(tr_feat).to(dev),top_k=args.k,batch_size=args.batch)
    pidx=np.asarray(pp.indices); ridx=np.asarray(rp.indices); ps=np.asarray(pp.scores); rs=np.asarray(rp.scores); centers=unit_to_latlon(part.leaf_centers_unit); train_coords=tr[['latitude','longitude']].to_numpy(); truth=test_idx[['latitude','longitude']].to_numpy(); pcoords=centers[pidx]; rcoords=train_coords[ridx]; raw=np.concatenate((ps,rs),1); coords=np.concatenate((pcoords,rcoords),1)
    grows=tr.feature_row.to_numpy(np.int64); glabel=part.labels.astype(np.int32); reps=np.full(len(part.leaf_counts),-1,np.int64); seen=np.zeros(len(reps),bool)
    for i,g in enumerate(glabel):
        if not seen[g]: reps[g]=grows[i]; seen[g]=True
    cand=np.concatenate((reps[pidx],grows[ridx]),1)
    desc=np.lib.format.open_memmap(args.gallery_desc,mode='r'); drows=np.load(args.gallery_desc_rows,mmap_mode='r').astype(np.int64,copy=False); pos=np.searchsorted(drows,cand); covered=(pos<len(drows)) & (drows[np.minimum(pos,len(drows)-1)]==cand)
    missing=np.unique(cand[~covered]); print(json.dumps({'candidate_rows':int(cand.size),'missing_gallery_rows':int(len(missing))}),flush=True)
    # Encode official queries and any gallery rows absent from the existing cache.
    processor=AutoProcessor.from_pretrained('geolocal/StreetCLIP',use_fast=True); model=CLIPModel.from_pretrained('geolocal/StreetCLIP',torch_dtype='auto').eval().cuda(); model.requires_grad_(False)
    qdesc=np.empty((len(test_idx),4,model.config.vision_config.hidden_size),np.float16); extra={}
    train_paths=train_idx.set_index('feature_row')['image_path'].astype(str).to_dict(); items=[('q',i,str(p)) for i,p in enumerate(test_idx.image_path.astype(str))] + [('g',int(r),train_paths[int(r)]) for r in missing]
    def load(item):
        with Image.open(item[2]) as im: return item,im.convert('RGB')
    pool=ThreadPoolExecutor(max_workers=args.workers)
    with torch.inference_mode():
        for s in range(0,len(items),args.extract_batch):
            batch_items=items[s:s+args.extract_batch]; loaded=list(pool.map(load,batch_items)); pix=processor(images=[x[1] for x in loaded],return_tensors='pt')['pixel_values'].cuda(non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16): tok=model.vision_model(pixel_values=pix).last_hidden_state[:,1:]
            n,tokn,d=tok.shape; side=int(round(tokn**.5)); tok=tok.reshape(n,side,side,d); h=side//2; quads=torch.stack((tok[:,:h,:h].mean((1,2)),tok[:,:h,h:].mean((1,2)),tok[:,h:,:h].mean((1,2)),tok[:,h:,h:].mean((1,2))),1); arr=torch.nn.functional.normalize(quads.float(),dim=-1).half().cpu().numpy()
            for j,(kind,key,_) in enumerate(batch_items):
                if kind=='q': qdesc[key]=arr[j]
                else: extra[int(key)]=arr[j]
            if s % (args.extract_batch*20)==0: print(f'extracted={min(s+len(items),len(items))}/{len(items)}',flush=True)
    pool.shutdown(wait=True)
    # Score candidates in GPU batches; existing descriptors are mmap'd, extras are in memory.
    patch=np.empty((len(test_idx),2*args.k),np.float32); qpos=np.searchsorted(drows,np.arange(len(test_idx)))
    # qdesc is separate; candidate lookup uses exact feature-row IDs.
    def get_desc(row):
        j=np.searchsorted(drows,row)
        if j<len(drows) and drows[j]==row: return np.asarray(desc[j],np.float32)
        return np.asarray(extra[int(row)],np.float32)
    with torch.inference_mode():
        for s in range(0,len(test_idx),args.batch):
            e=min(len(test_idx),s+args.batch); qv=torch.from_numpy(qdesc[s:e].astype(np.float32)).to(dev); cv=np.stack([[get_desc(int(x)) for x in row] for row in cand[s:e]],0); cv=torch.from_numpy(cv).to(dev); patch[s:e]=(qv[:,None]*cv).sum(-1).mean(-1).cpu().numpy()
            if s % (args.batch*20)==0: print(f'scored={e}/{len(test_idx)}',flush=True)
    def z(a): return (a-a.mean(1,keepdims=True))/np.maximum(a.std(1,keepdims=True),1e-6)
    rz,pz=z(raw),z(patch); pred=np.argmax(rz+args.alpha*pz,1); err=geo_km(truth,coords); fused=err[np.arange(len(err)),pred]
    payload={'method':'official_patch_token_late_fusion','official_test_evaluated':True,'queries':int(len(test_idx)),'k':int(args.k),'alpha_fixed_from_development':float(args.alpha),'missing_gallery_rows':int(len(missing)),'elapsed_seconds':float(time.perf_counter()-t0),'prototype':summary(err[:,:args.k].min(1)),'retrieval':summary(err[:,args.k:].min(1)),'union_anchor':summary(err[np.arange(len(err)),np.argmax(rz,1)]),'patch_fusion':summary(fused),'predictions_path':str(args.output.with_suffix('.predictions.parquet'))}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2),encoding='utf-8'); pd.DataFrame({'id':test_idx.id.to_numpy(),'distance_km':fused,'pred_latitude':coords[np.arange(len(coords)),pred,0],'pred_longitude':coords[np.arange(len(coords)),pred,1]}).to_parquet(args.output.with_suffix('.predictions.parquet'),index=False); print(json.dumps(payload,indent=2),flush=True)
if __name__=='__main__': main()
