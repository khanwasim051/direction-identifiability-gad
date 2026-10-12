"""CondCal-GAD reference implementation of the supplied manuscript.
NumPy/SciPy implementation; explicit full-batch Adam and backpropagation.
No target labels enter representation construction, fitting, or selection.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from dataclasses import dataclass, asdict
import argparse, copy, csv, hashlib, json, pickle, platform, resource, time
import numpy as np
import scipy
from scipy import sparse
from scipy.special import expit
from scipy.stats import rankdata
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

@dataclass
class Config:
    q:int=32
    pair_limit:int=128
    degree_exact:int=10
    degree_tolerance:float=0.15
    rank_tolerance:float=0.10
    pool_max:int=64
    pool_min:int=16
    anomaly_fraction:float=0.05
    clique_size:int=5
    donor_candidates:int=50
    learning_rate:float=0.001
    regularization:float=0.0001
    epochs:int=200
    patience:int=20
    validation_fraction:float=0.20

STAT_NAMES=['node_neighbor_discrepancy','neighbor_pair_mean','neighbor_pair_variance','closure','neighbor_degree']
GRAPH_IDS={'cora':1,'citeseer':2,'pubmed':3}

def rng_for(seed,graph_id,stage):
    return np.random.default_rng(np.random.SeedSequence([seed,graph_id,stage]))

def load_planetoid(name,root):
    """Same allx/tx test-index assembly as Planetoid; retain CiteSeer isolates.
    Pickle loading is only for files from the recorded original dataset repository.
    """
    def read(part):
        with (Path(root)/f'ind.{name}.{part}').open('rb') as f:
            return pickle.load(f,encoding='latin1')
    allx,tx,graph=read('allx'),read('tx'),read('graph')
    test_order=np.loadtxt(Path(root)/f'ind.{name}.test.index',dtype=np.int64)
    test_sorted=np.sort(test_order)
    if name=='citeseer':
        full_range=np.arange(test_sorted.min(),test_sorted.max()+1)
        expanded=sparse.lil_matrix((len(full_range),allx.shape[1]),dtype=np.float64)
        expanded[test_sorted-full_range[0],:]=tx
        tx=expanded.tocsr()
    x=sparse.vstack([allx,tx]).tolil()
    x[test_order,:]=x[test_sorted,:]
    x=x.toarray().astype(np.float64)
    n=len(x)
    row=np.repeat(list(graph.keys()),[len(v) for v in graph.values()])
    col=np.concatenate([np.asarray(v,dtype=np.int64) for v in graph.values()])
    a=sparse.csr_matrix((np.ones(len(row),dtype=np.int8),(row,col)),shape=(n,n))
    a=a.maximum(a.T); a.setdiag(0); a.eliminate_zeros(); a.data[:]=1; a.sort_indices()
    expected={'cora':(2708,1433,5278),'citeseer':(3327,3703,4552),'pubmed':(19717,500,44324)}
    assert (n,x.shape[1],a.nnz//2)==expected[name], (name,n,x.shape,a.nnz//2)
    return x,a

def inject(x,a,cfg,rng):
    """Disjoint anomalies; contextual donors maximize raw-feature Euclidean
    distance over 50 uniformly sampled nonself candidates. Copy ORIGINAL rows.
    Total floor(.05*N); structural half rounded down to complete cliques.
    """
    n=len(x); total=int(np.floor(cfg.anomaly_fraction*n))
    structural=(total//2//cfg.clique_size)*cfg.clique_size
    ids=rng.choice(n,total,replace=False)
    groups=ids[:structural].reshape(-1,cfg.clique_size)
    context=ids[structural:]
    out=x.copy(); donors=[]
    for i in context:
        candidates=rng.choice(n-1,min(cfg.donor_candidates,n-1),replace=False)
        candidates=candidates+(candidates>=i)
        distances=np.sum((x[candidates]-x[i])**2,axis=1)
        donor=candidates[np.argmax(distances)]
        out[i]=x[donor]; donors.append(int(donor))
    aa=a.tolil(copy=True); added=[]
    for group in groups:
        for p,u in enumerate(group):
            for v in group[p+1:]:
                if aa[u,v]==0: added.append([int(u),int(v)])
                aa[u,v]=1;aa[v,u]=1
    aa=aa.tocsr(); aa.sort_indices()
    labels=np.zeros(n,dtype=np.int8); labels[ids]=1
    types=np.zeros(n,dtype=np.int8); types[ids[:structural]]=1;types[context]=2
    manifest={'total':total,'structural':structural,'contextual':len(context),
              'structural_groups':groups.tolist(),'contextual_nodes':context.tolist(),
              'contextual_donors':donors,'added_undirected_edges':added,
              'donor_distance':'squared Euclidean on original unstandardized features',
              'donor_candidates':cfg.donor_candidates,'original_edges':a.nnz//2,'injected_edges':aa.nnz//2}
    assert np.all(np.any(out[context]!=x[context],axis=1))
    return out,aa,labels,types,manifest

def preprocess(x):
    x=x.copy()
    for j in np.where(~np.isfinite(x).all(axis=0))[0]:
        good=np.isfinite(x[:,j]);x[~good,j]=np.median(x[good,j]) if good.any() else 0
    std=x.std(axis=0); retained=std>0
    x=x[:,retained]
    sparsity=(x==0).mean(axis=1) if x.shape[1] else np.zeros(len(x))
    if x.shape[1]:
        x=(x-x.mean(axis=0))/std[retained]
        norm=np.linalg.norm(x,axis=1)
        x=np.divide(x,norm[:,None],out=np.zeros_like(x),where=norm[:,None]>0)
    return np.ascontiguousarray(x,dtype=np.float32),sparsity,retained

def midrank(v):
    good=np.isfinite(v);out=np.full(len(v),np.nan,dtype=float)
    if good.any():
        if np.ptp(v[good])==0:out[good]=0.0
        else:out[good]=(rankdata(v[good],method='average')-0.5)/good.sum()
    return out

def extract(x,a,cfg,rng):
    n=len(x);d=np.diff(a.indptr);ld=np.log1p(d)
    avg=np.divide(a@ld,d,out=np.full(n,np.nan),where=d>0)
    t=np.full((n,5),np.nan,dtype=float);t[:,4]=avg
    for i in range(n):
        nb=a.indices[a.indptr[i]:a.indptr[i+1]];degree=len(nb)
        if degree and x.shape[1]:
            chosen=nb if degree<=cfg.q else rng.choice(nb,cfg.q,replace=False)
            t[i,0]=np.mean(1.0-x[chosen]@x[i])
        if degree<2:continue
        count=degree*(degree-1)//2
        if count<=cfg.pair_limit:
            p,q=np.triu_indices(degree,k=1)
        else:
            pair_ids=rng.choice(count,cfg.pair_limit,replace=False)
            starts=np.arange(degree,dtype=np.int64)
            starts=starts*(2*degree-starts-1)//2
            p=np.searchsorted(starts,pair_ids,side='right')-1
            q=p+1+(pair_ids-starts[p])
        u,v=nb[p],nb[q]
        if x.shape[1]:
            dist=1.0-np.einsum('ij,ij->i',x[u],x[v])
            t[i,1]=dist.mean();t[i,2]=dist.var(ddof=0)
        t[i,3]=np.asarray(a[u,v]).ravel().mean()
    return t,d,avg

def calibrate(t,a,d,sparsity,avg,cfg,rng):
    """Exact statistic-specific pools. No labels accepted by this interface."""
    n=len(t);ld=np.log1p(d)
    b=[midrank(sparsity),midrank(avg),midrank(avg),midrank(avg),np.zeros(n)]
    r=np.zeros((n,5));mask=np.zeros((n,5));support=np.zeros((n,5))
    refs=np.full((n,5,cfg.pool_max),-1,dtype=np.int32)
    counts=np.zeros((n,5),dtype=np.int32);eligible_counts=np.zeros_like(counts)
    maxdist=np.full((n,5),np.nan);meandist=np.full((n,5),np.nan)
    # Random priorities assigned exchangeably to candidate nodes; fixed within run.
    priorities=rng.random((n,5));excluded=np.zeros(n,dtype=bool)
    for k in range(5):
        valid=np.isfinite(t[:,k]) & np.isfinite(b[k])
        valid_ids=np.flatnonzero(valid)
        degree_order=valid_ids[np.argsort(ld[valid_ids],kind='stable')]
        log_sorted=ld[degree_order]
        buckets={}
        for degree in range(cfg.degree_exact+1):
            ids=np.flatnonzero(valid & (d==degree))
            ids=ids[np.argsort(b[k][ids],kind='stable')]
            buckets[degree]=(ids,b[k][ids])
        for i in valid_ids:
            if d[i]<=cfg.degree_exact:
                ids,vals=buckets[int(d[i])]
                lo=np.searchsorted(vals,b[k][i]-cfg.rank_tolerance,'left')
                hi=np.searchsorted(vals,b[k][i]+cfg.rank_tolerance,'right')
                candidates=ids[lo:hi]
            else:
                lo=np.searchsorted(log_sorted,ld[i]-cfg.degree_tolerance,'left')
                hi=np.searchsorted(log_sorted,ld[i]+cfg.degree_tolerance,'right')
                candidates=degree_order[lo:hi]
                candidates=candidates[np.abs(b[k][candidates]-b[k][i])<=cfg.rank_tolerance]
            nb=a.indices[a.indptr[i]:a.indptr[i+1]]
            excluded[i]=True;excluded[nb]=True
            candidates=candidates[~excluded[candidates]]
            excluded[i]=False;excluded[nb]=False
            eligible_counts[i,k]=len(candidates)
            if len(candidates)<cfg.pool_min:continue
            degree_dist=np.zeros(len(candidates)) if d[i]<=cfg.degree_exact else np.abs(ld[candidates]-ld[i])/cfg.degree_tolerance
            distances=np.maximum(degree_dist,np.abs(b[k][candidates]-b[k][i])/cfg.rank_tolerance)
            order=np.lexsort((priorities[candidates,k],distances))[:cfg.pool_max]
            selected=candidates[order];size=len(selected)
            refs[i,k,:size]=selected;counts[i,k]=size
            maxdist[i,k]=distances[order].max();meandist[i,k]=distances[order].mean()
            values=t[selected,k]
            r[i,k]=2.0*(np.sum(values<t[i,k])+0.5*np.sum(values==t[i,k]))/size-1.0
            mask[i,k]=1;support[i,k]=size/cfg.pool_max
    z=np.concatenate([r,mask,support],axis=1)
    assert np.isfinite(z).all() and np.all(np.abs(r)<=1)
    assert np.all((counts==0)|((counts>=cfg.pool_min)&(counts<=cfg.pool_max)))
    return z,refs,{'counts':counts,'eligible_counts':eligible_counts,'max_distance':maxdist,'mean_distance':meandist}

def source_split(y,types,groups,cfg,rng):
    train=[];val=[]
    for label in [0,2]:
        ids=rng.permutation(np.flatnonzero(types==label))
        nv=max(1,int(np.rint(cfg.validation_fraction*len(ids))))
        val.extend(ids[:nv]);train.extend(ids[nv:])
    groups=np.asarray(groups,dtype=int)
    idx=rng.permutation(len(groups));nv=max(1,int(np.rint(cfg.validation_fraction*len(groups))))
    val.extend(groups[idx[:nv]].ravel());train.extend(groups[idx[nv:]].ravel())
    train=np.asarray(train,dtype=int);val=np.asarray(val,dtype=int)
    assert len(np.intersect1d(train,val))==0
    assert len(train)+len(val)==len(y)
    assert len(np.unique(y[train]))==len(np.unique(y[val]))==2
    for g in groups:assert np.isin(g,train).all() or np.isin(g,val).all()
    return train,val

class MLP:
    def __init__(self,seed):
        rng=np.random.default_rng(seed);self.p={}
        for j,(din,dout) in enumerate([(15,32),(32,16),(16,1)]):
            # PyTorch Linear-style uniform initialization, documented for reproduction.
            bound=1/np.sqrt(din)
            self.p[f'w{j}']=rng.uniform(-bound,bound,(din,dout))
            self.p[f'b{j}']=rng.uniform(-bound,bound,(dout,))
    def forward(self,x):
        p=self.p;a=x@p['w0']+p['b0'];h=np.maximum(a,0)
        b=h@p['w1']+p['b1'];g=np.maximum(b,0)
        logits=(g@p['w2']+p['b2']).ravel()
        return logits,(x,a,h,b,g)
    def predict(self,x):return expit(self.forward(x)[0])
    def loss_gradient(self,x,y,weights,beta):
        logits,(x,a,h,b,g)=self.forward(x)
        loss=np.sum(weights*(np.logaddexp(0,logits)-y*logits))
        delta=(weights*(expit(logits)-y))[:,None]
        grad={'w2':g.T@delta,'b2':delta.sum(axis=0)}
        delta=(delta@self.p['w2'].T)*(b>0)
        grad.update(w1=h.T@delta,b1=delta.sum(axis=0))
        delta=(delta@self.p['w1'].T)*(a>0)
        grad.update(w0=x.T@delta,b0=delta.sum(axis=0))
        for k in self.p:
            loss+=beta*np.sum(self.p[k]**2);grad[k]+=2*beta*self.p[k]
        return float(loss),grad

def train_model(sources,cfg,seed,out):
    # This function receives ONLY source representations, labels and splits.
    xs=[];ys=[];ws=[]
    for s in sources:
        idx=s['train'];y=s['y'][idx];x=s['z'][idx]
        w=np.where(y==1,0.5/(y==1).sum(),0.5/(y==0).sum())/len(sources)
        xs.append(x);ys.append(y);ws.append(w)
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws)
    model=MLP(seed);m={k:np.zeros_like(v) for k,v in model.p.items()};v=copy.deepcopy(m)
    best=-np.inf;best_params=None;history=[];stale=0
    start=time.perf_counter()
    for epoch in range(1,cfg.epochs+1):
        loss,grad=model.loss_gradient(x,y,w,cfg.regularization)
        for k in model.p:
            m[k]=0.9*m[k]+0.1*grad[k];v[k]=0.999*v[k]+0.001*grad[k]**2
            mh=m[k]/(1-0.9**epoch);vh=v[k]/(1-0.999**epoch)
            model.p[k]-=cfg.learning_rate*mh/(np.sqrt(vh)+1e-8)
        aps=[average_precision_score(s['y'][s['val']],model.predict(s['z'][s['val']])) for s in sources]
        mean_ap=float(np.mean(aps));history.append({'epoch':epoch,'training_loss':loss,'mean_source_validation_ap':mean_ap,'citeseer_validation_ap':float(aps[0]),'pubmed_validation_ap':float(aps[1])})
        if mean_ap>best:
            best=mean_ap;best_epoch=epoch;best_params=copy.deepcopy(model.p);stale=0
        else:stale+=1
        if stale>=cfg.patience:break
    model.p=best_params
    np.savez_compressed(out/'model.npz',**model.p)
    write_csv(out/'training_history.csv',history)
    return model,{'best_epoch':best_epoch,'epochs_executed':epoch,'source_validation_ap':best,'training_seconds':time.perf_counter()-start}

def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def prepare_graph(name,base,cfg,seed,out,is_source):
    x,a=base;gid=GRAPH_IDS[name];start=time.perf_counter()
    x,a,y,types,injection=inject(x,a,cfg,rng_for(seed,gid,0))
    xx,sp,retained=preprocess(x);feature_seconds=time.perf_counter()-start
    t0=time.perf_counter();t,d,avg=extract(xx,a,cfg,rng_for(seed,gid,1));stat_seconds=time.perf_counter()-t0
    t0=time.perf_counter();z,refs,diag=calibrate(t,a,d,sp,avg,cfg,rng_for(seed,gid,2));cal_seconds=time.perf_counter()-t0
    s={'name':name,'z':z,'y':y,'types':types,'injection':injection}
    if is_source:s['train'],s['val']=source_split(y,types,injection['structural_groups'],cfg,rng_for(seed,gid,3))
    np.savez_compressed(out/f'{name}_representation.npz',z=z,statistics=t,degree=d,sparsity=sp,neighbor_degree=avg,labels=y,anomaly_type=types,retained_features=retained,**diag,**({k:s[k] for k in ['train','val']} if is_source else {'references':refs}))
    (out/f'{name}_injection.json').write_text(json.dumps(injection,indent=2))
    coverage=[]
    for k,label in enumerate(STAT_NAMES):
        supported=diag['counts'][:,k]>0;count=diag['counts'][supported,k]
        reference_y=y[np.maximum(refs[:,k],0)];present=refs[:,k]>=0
        contamination=np.divide((reference_y*present).sum(axis=1),present.sum(axis=1),out=np.full(len(y),np.nan),where=present.sum(axis=1)>0)
        coverage.append({'graph':name,'seed':seed,'statistic':label,'supported_fraction':float(supported.mean()),'normal_supported_fraction':float(supported[y==0].mean()),'anomaly_supported_fraction':float(supported[y==1].mean()),'mean_pool_size_supported':float(count.mean()) if len(count) else None,'median_pool_size_supported':float(np.median(count)) if len(count) else None,'mean_max_matching_distance':float(np.nanmean(diag['max_distance'][:,k])),'normal_reference_contamination':float(np.nanmean(contamination[y==0])),'anomaly_reference_contamination':float(np.nanmean(contamination[y==1]))})
    s['coverage']=coverage;s['runtime']={'preprocessing_and_injection_seconds':feature_seconds,'statistics_seconds':stat_seconds,'matching_seconds':cal_seconds,'dense_features_MiB':xx.nbytes/2**20,'nodes':len(y),'features_retained':int(retained.sum()),'edges':a.nnz//2,'isolated_nodes':int((d==0).sum()),'all_statistics_supported_fraction':float((z[:,5:10].sum(axis=1)==5).mean()),'any_statistic_supported_fraction':float((z[:,5:10].sum(axis=1)>0).mean())}
    print(f"seed={seed} {name}: N={len(y)}, anomalies={y.sum()}, support={[round(r['supported_fraction'],3) for r in coverage]}, elapsed={time.perf_counter()-start:.1f}s",flush=True)
    return s

def run(args):
    cfg=Config();root=Path(__file__).resolve().parent;data=root/'data';outroot=Path(args.output or root/'results');outroot.mkdir(parents=True,exist_ok=True)
    seeds=args.seeds
    (outroot/'config.json').write_text(json.dumps({'configuration':asdict(cfg),'seeds':seeds,'sources':['citeseer','pubmed'],'target':'cora','protocol':'citation diagnostic cross-graph transfer; injected labels only','dtype':'float32 standardized features; float64 statistics, ranks and MLP','optimizer':'full-batch Adam beta1=.9 beta2=.999 eps=1e-8; explicit beta*sum(theta^2), including biases','data_loading':'original Planetoid files; same reconstruction as Planetoid interface; no PyG dependency','numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'python':platform.python_version(),'platform':platform.platform(),'methodology_sha256':args.methodology_sha256},indent=2))
    from download_data import main as download
    if not (data/'provenance.json').exists():download(data)
    # Target graph is loaded and processed only after source-only checkpoint selection.
    bases={name:load_planetoid(name,data) for name in ['citeseer','pubmed']}
    rows=[];allcoverage=[];timings=[]
    for seed in seeds:
        out=outroot/f'seed_{seed}';out.mkdir(exist_ok=True);start=time.perf_counter()
        sources=[prepare_graph(name,bases[name],cfg,seed,out,True) for name in ['citeseer','pubmed']]
        model,fit=train_model(sources,cfg,seed,out)
        target=prepare_graph('cora',load_planetoid('cora',data),cfg,seed,out,False)
        t0=time.perf_counter();scores=model.predict(target['z']);score_seconds=time.perf_counter()-t0
        y=target['y'];typ=target['types']
        metrics={'seed':seed,'auroc':float(roc_auc_score(y,scores)),'average_precision':float(average_precision_score(y,scores)),**fit,'normal_count':int((y==0).sum()),'structural_count':int((typ==1).sum()),'contextual_count':int((typ==2).sum()),'prevalence':float(y.mean()),'cora_scoring_seconds':score_seconds,'total_seed_seconds':time.perf_counter()-start}
        for v,label in [(1,'structural'),(2,'contextual')]:
            keep=(typ==0)|(typ==v)
            metrics[f'{label}_vs_normal_auroc']=float(roc_auc_score(y[keep],scores[keep]));metrics[f'{label}_vs_normal_ap']=float(average_precision_score(y[keep],scores[keep]))
        rows.append(metrics)
        write_csv(out/'cora_predictions.csv',[{'node':i,'injected_anomaly':int(y[i]),'anomaly_type':int(typ[i]),'score':float(scores[i]),'supported_statistics':int(target['z'][i,5:10].sum())} for i in range(len(y))])
        for s in sources+[target]:
            allcoverage.extend(s['coverage']);timings.append({'seed':seed,'graph':s['name'],**s['runtime']})
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        write_csv(outroot/'seed_metrics.csv',rows);write_csv(outroot/'reference_coverage.csv',allcoverage);write_csv(outroot/'graph_runtime.csv',timings)
        print('RESULT '+json.dumps(metrics),flush=True)
    summary={key:{'mean':float(np.mean([r[key] for r in rows])),'sample_sd':float(np.std([r[key] for r in rows],ddof=1)) if len(rows)>1 else None} for key in ['auroc','average_precision','structural_vs_normal_auroc','structural_vs_normal_ap','contextual_vs_normal_auroc','contextual_vs_normal_ap','total_seed_seconds']}
    summary['peak_process_rss_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    summary['n_seeds']=len(rows);summary['note']='Variation includes injection, source splits, statistic sampling, reference tie priorities and model initialization; same target domain, not independent domains.'
    (outroot/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY '+json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,nargs='+',default=[11,23,37,53,71]);parser.add_argument('--output');parser.add_argument('--methodology-sha256',default=None)
    run(parser.parse_args())
