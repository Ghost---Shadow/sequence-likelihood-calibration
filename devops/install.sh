conda create -n slic python=3.9 -y
conda activate slic
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install -e .
