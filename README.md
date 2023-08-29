## Synchronizing Modalities before Fusion: A New Contrastive Learning Approach to Multimodal Sentiment Detection

This is the repository of Synchronizing Modalities before Fusion: A New Contrastive Learning Approach to Multimodal Sentiment Detection

![image](SMF.png)

### Requirements:
```shell
pytorch 1.8.0
transformers 4.8.1
timm 0.4.9
```

### Download:
MSCOCO: http://cocodataset.org/
MVSA-*: http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/
RU-senti: coming soon

### Pre-training:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env Pretrain.py --config ./configs/Pretrain.yaml --output_dir output/Pretrain
```

### Multimodal Sentiment Detection:
train:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env MSD.py --config ./configs/MSD.yaml --output_dir output/MSD 
```
eval:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env MSD.py --config ./configs/MSD.yaml --output_dir output/MSD --checkpoint ./output/MSD/checkpoint_best.pth --eval 1
```  
