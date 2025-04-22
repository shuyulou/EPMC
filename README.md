## Ensuring Pre-Fusion Modality Consistency: A New Contrastive Learning Approach to Multimodal Sentiment Detection

This is the repository of Ensuring Pre-Fusion Modality Consistency: A New Contrastive Learning Approach to Multimodal Sentiment Detection

![image](EPMC.pdf)

### Requirements:
```shell
pytorch 1.8.0
transformers 4.8.1
timm 0.4.9
```

### Download:
MSCOCO: http://cocodataset.org/

MVSA-*: http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/

RU-senti: https://drive.google.com/file/d/1ED1SHlYRVhduDi14-f2Xp0Mk35PdjQJU/view

HFM: https://github.com/wrk226/pytorch-multimodal_sarcasm_detection

Our checkpoint: https://drive.google.com/file/d/1I2Ivlz3qqLp3puD6ZKT1lmOMmF2KlwMj/view?usp=drive_link

### Pre-training:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env Pretrain.py --config ./configs/Pretrain.yaml --output_dir output/Pretrain
```

### Multimodal Sentiment Detection:
train:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env EPMC.py --config ./configs/MSD.yaml --output_dir output/MSD 
```
eval:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env EPMC.py --config ./configs/MSD.yaml --output_dir output/MSD --checkpoint ./output/MSD/checkpoint_best.pth --eval 1
```  
