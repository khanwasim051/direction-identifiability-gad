"""Download original Planetoid files; no document-class labels are downloaded."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request, hashlib, json, time
BASE='https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/'
def download(task):
    name, part, root = task
    filename=f'ind.{name}.{part}'
    path=root/filename
    for attempt in range(3):
        try:
            if not path.exists():
                with urllib.request.urlopen(BASE+filename, timeout=90) as response:
                    content=response.read()
                path.write_bytes(content)
            content=path.read_bytes()
            return {'filename':filename,'url':BASE+filename,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
        except Exception:
            if attempt==2: raise
    
def main(root=None):
    root=Path(root or Path(__file__).parent/'data'); root.mkdir(parents=True,exist_ok=True)
    jobs=[(n,p,root) for n in ['cora','citeseer','pubmed'] for p in ['allx','tx','graph','test.index']]
    with ThreadPoolExecutor(max_workers=6) as pool:
        records=list(pool.map(download,jobs))
    (root/'provenance.json').write_text(json.dumps(records,indent=2))
    for r in records: print(r['filename'],r['bytes'],flush=True)
if __name__=='__main__':main()
