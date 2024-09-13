conda create -n "ray" python=3.9.15
cd /opt/conda/envs/ray/bin
/opt/conda/envs/ray/bin/pip install ipykernel
/opt/conda/envs/ray/bin/python3 -m ipykernel install --user --name=ray
/opt/conda/envs/ray/bin/pip install -r /mnt/code/requirements-ray.txt
cd /mnt/code