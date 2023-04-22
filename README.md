# VQ-VAE

Basic Pytorch implementation of [VQ-VAE](https://paperswithcode.com/method/vq-vae).


## Environment
Build docker image:
`docker build . -t vq-vae`

Run docker image:
`docker run -it --shm-size 8g -v $PWD/:/project/ --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all vq-vae`

## Training
`python src/models/train_model.py`
