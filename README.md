# CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature Leveraging

This repository hosts the source code for the paper titled "[*CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature Leveraging*](https://arxiv.org/abs/2503.00325)", which has been accepted for publication at the **Conference on Computer Vision and Pattern Recognition (CVPR) 2025**.

## Getting Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download datasets
We evaluate on both large-scale and small-scale OOD datasets.

- Large-scale OOD datasets: 
    - In-distribution dataset: **ImageNet-1k**
    - Out-of-distribution datasets: **iNaturalist, SUN, Places, Textures, OpenImages-O, ImageNet-O, SSB-hard, NINCO**
- Small-scale OOD datasets:
    - In-distribution dataset: **CIFAR-10, CIFAR-100**
    - Out-of-distribution datasets: **SVHN, LSUN_crop, LSUN_resize, iSUN, Places, Textures**

- Download the datasets and put them in the `data` folder.

### 3. Pre-trained Model Preparation
For the **CIFAR-10** and **CIFAR-100** benchmarks, we provide the pre-trained checkpoints in the `checkpoints` folder.
For the **ImageNet-1k** benchmark, we utilize the well-trained model provided by *torchvision*.

### 4. Run the code

```bash
python ood_eval.py --OOD_method CADRef --gpu 0 --bs 32  --ind_dataset ImageNet --model resnet --ood_dataset iNat SUN Places Textures openimage_o imagenet_o
```
```
usage: ood_eval.py [-h] [--ind_dataset IND_DATASET] [--ood_dataset OOD_DATASET [OOD_DATASET ...]] [--model {resnet,vit,convnext,densenet,regnet,efficientnet,swin}] [--gpu GPU] [--num_classes NUM_CLASSES]
                   [--random_seed RANDOM_SEED] [--bs BS] [--OOD_method {MSP,ODIN,Energy,GEN,ReAct,DICE,GradNorm,MaxLogit,ASH,OptFS,VIM,Residual,CARef,CADRef}] [--use_feature_cache USE_FEATURE_CACHE]
                   [--use_score_cache USE_SCORE_CACHE] [--cache_dir CACHE_DIR] [--result_dir RESULT_DIR] [--num_workers NUM_WORKERS] [--logit_method {Energy,MSP,MaxLogit,GEN}]

options:
  -h, --help            show this help message and exit
  --ind_dataset IND_DATASET
                        in-distribution dataset name
  --ood_dataset OOD_DATASET [OOD_DATASET ...]
                        OOD dataset list
  --model {resnet,vit,convnext,densenet,regnet,efficientnet,swin}
                        model name
  --gpu GPU             gpu id
  --num_classes NUM_CLASSES
                        number of classes
  --random_seed RANDOM_SEED
                        random seed
  --bs BS               batch size
  --OOD_method {MSP,ODIN,Energy,GEN,ReAct,DICE,GradNorm,MaxLogit,ASH,OptFS,VIM,Residual,CARef,CADRef}
                        OOD method name
  --use_feature_cache USE_FEATURE_CACHE
                        use feature cache
  --use_score_cache USE_SCORE_CACHE
                        use score cache
  --cache_dir CACHE_DIR
                        cache directory
  --result_dir RESULT_DIR
                        result directory
  --num_workers NUM_WORKERS
                        number of workers
  --logit_method {Energy,MSP,MaxLogit,GEN}
                        logit method for CADRef
```

## Citation
```
@inproceedings{ling2025cadref,
  title = {CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature Leveraging},
  author = {Zhiwei Ling and Yachen Chang and Hailiang Zhao and Xinkui Zhao and Kingsum Chow and Shuiguang Deng},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {4968--4977},
  year = {2025},
}
```