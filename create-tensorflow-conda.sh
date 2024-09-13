conda create -n "tensorboard" python=3.9.15
cd /opt/conda/envs/tensorboard/bin
/opt/conda/envs/tensorboard/bin/pip install ipykernel
/opt/conda/envs/tensorboard/bin/python3 -m ipykernel install --user --name=tensorboard
/opt/conda/envs/tensorboard/bin/pip install -r /mnt/code/requirements-tensorboard.txt
cd /mnt/code