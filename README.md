# Regional Homogeneity: Towards Learning Transferable Universal Adversarial Perturbations Against Defenses

## Introduction
This repository contains the code for paper [Regional Homogeneity: Towards Learning Transferable Universal Adversarial Perturbations Against Defenses](https://arxiv.org/abs/1904.0????). 
This paper shows that a simple universal perturbation can fool a series of state-of-the-art defenses.

## Usage

### Dependencies
+ [Anaconda](https://www.anaconda.com/distribution/) 
+ Python3.6
+ Tensorflow 1.10.0
+ Tensorpack 0.9.0.1
+ easydict
+ scipy
+ pillow

Here is a sample scrip to install Dependencies after you have Anaconda.
```bash
conda create -n python3 python=3.6
source activate python3
pip install --upgrade tensorflow-gpu
pip install --upgrade git+https://github.com/tensorpack/tensorpack.git
pip install easydict
conda install -c anaconda scipy
pip install pillow
```

### Dataset and model checkpoints
We use images from ImageNet LSVRC 2012 Validation Set and resized them to 299x299.
You can download the preprocessed images **[HERE](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/yli286_jh_edu/Ecdhl1ZmYLVDmjsEBCTxAEsBYQndaXNu4StPmrAuin2IrQ?e=wRVSUd)**
if you accept the [terms](http://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5).

We support generate adversarial examples with 3 clean trained models (Inception-{v3, v4}, Inception-Resnet-v2),
and evaluate them by 3 ensemble adversarial trained models (ens3_inception_v3, ens4_inception_v3, ens_inception_resnet_v2).
We will release more defense models that mentioned in the paper.
We original download them from [here](https://github.com/tensorflow/models/tree/master/research/slim) and [here](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models)
and then slightly modified the tensor name. You can download the modified checkpoints from **[HERE](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/yli286_jh_edu/Eb6l0vTS84pHpctEW8lT0I4BT6T8RhbDA-E1wvWuxd7Ccw?e=gyhmQt)**.

After download them, edit and use ```data/link_to_data.sh``` to build soft link ```data/checkpoints``` and ```data/val_data``` by
```bash
bash data/link_to_data.sh
```

We assign every network with an id, so that they can be shortly mentioned in one character. Here is a table to provide ids for each network.
You can see line 69 to 70 of config.py for more details.

ID | 0 | 1 | 2 | 
---|---|---|---|
Networks for Training|IncV3|IncV4|IncRes|
Networks for Evaluation|Ens3IncV3|Ens4IncV3|EnsIncRes|
### Train, Attack and Eval
```bash
python train.py                          # train based on IncV3
python attack.py --GPU_ID 0              # attack
python eval.py --GPU_ID 0                # evaluate
python attack.py --universal --GPU_ID 0  # universal attack  
python eval.py --universal --GPU_ID 0    # evaluate
```

If you find the code useful, please consider citing the following paper.

    @article{li2019regional,
      title={Regional Homogeneity: Towards Learning Transferable Universal Adversarial Perturbations Against Defenses},
      author={Li, Yingwei and Bai, Song and Xie, Cihang and Liao, Zhenyu and Shen, Xiaohui and Yuille, Alan},
      journal={arXiv preprint arXiv:1904.0????},
      year={2019}
    }

If you encounter any problems or have any inquiries, please contact us at yingwei.li@jhu.edu.
