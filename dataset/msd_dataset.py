import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption


class msd_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):    
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.max_words = max_words
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(ann['text'], self.max_words)

        return image, sentence, int(ann['target'])
    