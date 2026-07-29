"""Fast official patch-fusion evaluator with reusable descriptor caches."""
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

def geo(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.ndim == b.ndim - 1: a = np.expand_dims(a, -2)
    la1,lo1=np.radians(a[...,0]),np.radians(a[...,1]); la2,lo2=np.radians(b[...,0]),np.radians(b[...,1]); h=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2; return 6371.0088*2*np.arcsin(np.sqrt(np.clip(h,0,1)))
def met(e):
    e=np.asarray(e,float); return {'queries':int(len(e)),'accuracy_1km':float(np.mean(e<=1)),'accuracy_25km':float(np.mean(e<=25)),'accuracy_200km':float(np.mean(e<=200)),'median_error_km':float(np.median(e)),'mean_error_km':float(np.mean(e)),'geoscore':float(np.mean(np.exp(-e/1492.7)))}
def main():
    ap=argparse.ArgumentParser();
    for n in ['train-features','train-index','test-features','test-index','gallery-desc','gallery-desc-rows','output']: ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--k',type=int,default=16); ap.add_argument('--batch',type=int,default=256); ap.add_argument('--extract-batch',type=int,default=32); ap.add_argument('--alpha',type=float,default=.75); ap.add_argument('--workers',type=int,default=12); a=ap.parse_args()
    t0=time.perf_counter(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ti=pd.read_parquet(a.train_index).sort_values('feature_row').reset_index(drop=True); qi=pd.read_parquet(a.test_index).reset_index(drop=True); tf=np.load(a.train_features,mmap_mode='r'); qf=np.asarray(np.load(a.test_features,mmap_mode='r'),np.float32); mask=ti.split.astype(str).eq('train').to_numpy(); tr=ti.loc[mask].reset_index(drop=True); x=np.asarray(tf[mask],np.float32)
    part=build_adaptive_s2_partition(tr.latitude.to_numpy(),tr.longitude.to_numpy(),min_level=3,max_level=12,max_samples_per_leaf=64); sums=np.zeros((len(part.leaf_counts),x.shape[1]),np.float64); np.add.at(sums,part.labels,x.astype(np.float64)); proto=(sums/np.maximum(np.linalg.norm(sums,axis=1,keepdims=True),1e-12)).astype(np.float32); q=torch.from_numpy(qf).to(dev); pp=topk_cosine(q,torch.from_numpy(proto).to(dev),top_k=a.k,batch_size=a.batch); rp=topk_cosine(q,torch.from_numpy(x).to(dev),top_k=a.k,batch_size=a.batch); pidx=np.asarray(pp.indices); ridx=np.asarray(rp.indices); raw=np.concatenate((np.asarray(pp.scores),np.asarray(rp.scores)),1); centers=unit_to_latlon(part.leaf_centers_unit); truth=qi[['latitude','longitude']].to_numpy(); coords=np.concatenate((centers[pidx],tr[['latitude','longitude']].to_numpy()[ridx]),1); grows=tr.feature_row.to_numpy(np.int64); reps=np.full(len(part.leaf_counts),-1,np.int64); seen=np.zeros(len(reps),bool)
    for i,g in enumerate(part.labels):
        if not seen[g]: reps[g]=grows[i]; seen[g]=True
    cand=np.concatenate((reps[pidx],grows[ridx]),1); desc=np.lib.format.open_memmap(a.gallery_desc,mode='r'); drows=np.load(a.gallery_desc_rows,mmap_mode='r').astype(np.int64,copy=False); pos=np.searchsorted(drows,cand); covered=(pos<len(drows))&(drows[np.minimum(pos,len(drows)-1)]==cand); missing=np.unique(cand[~covered]); print(json.dumps({'candidate_rows':int(cand.size),'missing_gallery_rows':int(len(missing))}),flush=True)
    outdir=a.output.parent; outdir.mkdir(parents=True,exist_ok=True); qcache=outdir/'official_query_patch.npy'; gcache=outdir/'official_missing_gallery_patch.npz'
    if qcache.exists() and gcache.exists(): qdesc=np.load(qcache,mmap_mode='r'); z=np.load(gcache); mrows=z['rows']; mdesc=z['desc']; print('reusing_patch_cache',flush=True)
    else:
        processor=AutoProcessor.from_pretrained('geolocal/StreetCLIP',use_fast=True); model=CLIPModel.from_pretrained('geolocal/StreetCLIP',torch_dtype='auto').eval().cuda(); model.requires_grad_(False); qdesc=np.empty((len(qi),4,model.config.vision_config.hidden_size),np.float16); extra={}; paths=ti.set_index('feature_row').image_path.astype(str).to_dict(); items=[('q',i,str(p)) for i,p in enumerate(qi.image_path.astype(str))]+[('g',int(r),paths[int(r)]) for r in missing]
        def load(it):
            with Image.open(it[2]) as im: return it,im.convert('RGB')
        pool=ThreadPoolExecutor(max_workers=a.workers)
        with torch.inference_mode():
            for s in range(0,len(items),a.extract_batch):
                bi=items[s:s+a.extract_batch]; imgs=list(pool.map(load,bi)); pix=processor(images=[v[1] for v in imgs],return_tensors='pt')['pixel_values'].cuda(non_blocking=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16): tok=model.vision_model(pixel_values=pix).last_hidden_state[:,1:]
                n,tokn,d=tok.shape; side=int(round(tokn**.5)); tok=tok.reshape(n,side,side,d); h=side//2; quads=torch.stack((tok[:,:h,:h].mean((1,2)),tok[:,:h,h:].mean((1,2)),tok[:,h:,:h].mean((1,2)),tok[:,h:,h:].mean((1,2))),1); arr=torch.nn.functional.normalize(quads.float(),dim=-1).half().cpu().numpy()
                for j,(kind,key,_) in enumerate(bi): (qdesc.__setitem__((key,slice(None),slice(None)),arr[j]) if kind=='q' else extra.__setitem__(int(key),arr[j]))
                if s%(a.extract_batch*100)==0: print(f'extracted={min(s+len(bi),len(items))}/{len(items)}',flush=True)
        pool.shutdown(wait=True); np.save(qcache,qdesc); mrows=np.asarray(sorted(extra),np.int64); mdesc=np.stack([extra[int(r)] for r in mrows]); np.savez(gcache,rows=mrows,desc=mdesc)
    cvpath=outdir/'official_candidate_patch.npy'; cv=np.lib.format.open_memmap(cvpath,mode='w+',dtype=np.float16,shape=(len(cand),cand.shape[1],qdesc.shape[1],qdesc.shape[2])) if not cvpath.exists() else np.lib.format.open_memmap(cvpath,mode='r+',dtype=np.float16,shape=(len(cand),cand.shape[1],qdesc.shape[1],qdesc.shape[2])); pos=np.searchsorted(drows,cand); covered=(pos<len(drows))&(drows[np.minimum(pos,len(drows)-1)]==cand); cv[covered]=desc[pos[covered]]; mp=np.searchsorted(mrows,cand[~covered]); cv[~covered]=mdesc[mp]; cv.flush(); print('candidate_patch_materialized',flush=True)
    patch=np.empty((len(qi),2*a.k),np.float32)
    with torch.inference_mode():
        for s in range(0,len(qi),a.batch):
            e=min(len(qi),s+a.batch); qv=torch.from_numpy(np.asarray(qdesc[s:e],np.float32)).to(dev); c=torch.from_numpy(np.asarray(cv[s:e],np.float32)).to(dev); patch[s:e]=(qv[:,None]*c).sum(-1).mean(-1).cpu().numpy();
            if s%(a.batch*20)==0: print(f'scored={e}/{len(qi)}',flush=True)
    z=lambda v:(v-v.mean(1,keepdims=True))/np.maximum(v.std(1,keepdims=True),1e-6); rz,pz=z(raw),z(patch); err=geo(truth,coords); ii=np.arange(len(err)); ptop=err[ii,np.argmax(rz[:,:a.k],1)]; rtop=err[ii,a.k+np.argmax(rz[:,a.k:],1)]; union=err[ii,np.argmax(rz,1)]; pred=np.argmax(rz+a.alpha*pz,1); fused=err[ii,pred]; np.save(a.output.parent/'official_patch_fusion_dist.npy',fused); np.save(a.output.parent/'official_union_dist.npy',union); np.save(a.output.parent/'official_retrieval_dist.npy',rtop); payload={'method':'official_patch_token_late_fusion','official_test_evaluated':True,'queries':int(len(qi)),'k':a.k,'alpha_fixed_from_development':a.alpha,'missing_gallery_rows':int(len(missing)),'elapsed_seconds':time.perf_counter()-t0,'prototype':met(ptop),'retrieval':met(rtop),'union_anchor':met(union),'patch_fusion':met(fused)}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2),flush=True)
if __name__=='__main__': main()
