# The input image IDs are checked against vendor/source-lock.jepa-mppi.json.
FROM stonefish-slam:vggt-runtime-cached AS cuda_runtime
RUN python3 - <<'PY'
from pathlib import Path
import shutil
source=Path('/opt/research/lib/python3.12/site-packages')
target=Path('/torch-runtime');target.mkdir()
modules={'torch','torchgen','functorch','nvidia','triton','sympy','mpmath','networkx','filelock','fsspec','typing_extensions.py','jinja2','markupsafe'}
for path in source.iterdir():
    if path.name in modules or (path.name.endswith('.dist-info') and path.name.split('-')[0] in modules|{'typing_extensions'}):
        if path.is_dir():shutil.copytree(path,target/path.name)
        else:shutil.copy2(path,target/path.name)
PY
FROM stonefish-slam:actuators
USER root
COPY --from=cuda_runtime /torch-runtime/ /usr/local/lib/python3.12/dist-packages/
COPY --from=cuda_runtime /opt/runtime-cache-manifest.json /opt/uw/jepa-torch-runtime.json
COPY algorithm.tar source-lock.json /opt/uw/jepa-mppi/
RUN python3 - <<'PY'
import hashlib,json,tarfile
from pathlib import Path
folder=Path('/opt/uw/jepa-mppi')
lock=json.loads((folder/'source-lock.json').read_text())
assert hashlib.sha256((folder/'algorithm.tar').read_bytes()).hexdigest()==lock['algorithm_archive_sha256']
with tarfile.open(folder/'algorithm.tar') as archive:archive.extractall(folder/'source',filter='data')
import torch,numpy,cv2
assert torch.__version__=='2.9.1+cu128'
print('JEPA runtime:',torch.__version__,numpy.__version__,cv2.__version__)
PY
