import sys,numpy as np
for p in sys.argv[1:]:
 d=np.load(p,allow_pickle=True); print(p); print({k:(d[k].shape,str(d[k].dtype)) for k in d.files})
