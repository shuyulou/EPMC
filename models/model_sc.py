'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertModel
from models.transformer import TransformerEncoder
from models.tokenization_bert import BertTokenizer

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random
from torch.autograd import Variable

def compute_class_weights(histogram):
    classWeights = np.ones(3, dtype=np.float32)
    normHist = histogram / np.sum(histogram)
    for i in range(3):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
    return classWeights
def focal_loss_zhihu(input, target):
    '''
    :param input: 使用知乎上面大神给出的方案 https://zhuanlan.zhihu.com/p/28527749
    :param target:
    :return:
    '''
    n, c= input.size()

    target = target.long()
    inputs = input.contiguous().view(-1, c)
    target = target.contiguous().view(-1)

    N = inputs.size(0)
    C = inputs.size(1)

    number_0 = torch.sum(target == 0).item()
    number_1 = torch.sum(target == 1).item()
    number_2 = torch.sum(target == 2).item()

    frequency = torch.tensor((number_0, number_1, number_2), dtype=torch.float32)
    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency)

    weights = torch.from_numpy(classWeights).float()
    weights=weights[target.view(-1)]#这行代码非常重要

    gamma = 2

    P = F.softmax(inputs, dim=1)#shape [num_samples,num_classes]

    class_mask = inputs.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = target.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)#shape [num_samples,num_classes] one-hot encoding

    probs = (P * class_mask).sum(1).view(-1, 1)#shape [num_samples,]
    log_p = probs.log()

    print('in calculating batch_loss',weights.shape,probs.shape,log_p.shape)

    # batch_loss = -weights * (torch.pow((1 - probs), gamma)) * log_p
    batch_loss = -(torch.pow((1 - probs), gamma)) * log_p

    print(batch_loss.shape)

    loss = batch_loss.mean()
    return loss

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 fusion_encoder = None,
                 config = None,    
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        embed_dim = config['embed_dim']
        self.embed_dim= config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   #图像编码器
        
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        fusion_bert_config = BertConfig.from_json_file(config['fusion_bert_config'])
        
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)  #文本编码器  
        self.fusion_encoder = BertForMaskedLM.from_pretrained(fusion_encoder, config=fusion_bert_config)  #融合编码器

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim) #全连接
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.conv1d_t = nn.Conv1d(text_width, self.embed_dim, kernel_size=1, padding=0, bias=False)
        self.conv1d_i = nn.Conv1d(vision_width, self.embed_dim, kernel_size=1, padding=0, bias=False)
        
        num_heads = 16
        attn_dropout = 0.1
        relu_dropout = 0.1
        res_dropout = 0.1
        embed_dropout = 0.1
        attn_mask = 'store_false'
        layers = 4

        self.trans_iichange = TransformerEncoder(embed_dim=embed_dim,
                                                 num_heads=num_heads,
                                                 layers=layers,
                                                 attn_dropout=attn_dropout,
                                                 relu_dropout=relu_dropout,
                                                 res_dropout=res_dropout,
                                                 embed_dropout=embed_dropout,
                                                 attn_mask=attn_mask)
        self.trans_ttchange = TransformerEncoder(embed_dim=embed_dim,
                                                 num_heads=num_heads,
                                                 layers=layers,
                                                 attn_dropout=attn_dropout,
                                                 relu_dropout=relu_dropout,
                                                 res_dropout=res_dropout,
                                                 embed_dropout=embed_dropout,
                                                 attn_mask=attn_mask)
        self.trans_itchange = TransformerEncoder(embed_dim=embed_dim,
                                                 num_heads=num_heads,
                                                 layers=layers,
                                                 attn_dropout=attn_dropout,
                                                 relu_dropout=relu_dropout,
                                                 res_dropout=res_dropout,
                                                 embed_dropout=embed_dropout,
                                                 attn_mask=attn_mask)
        self.trans_tichange = TransformerEncoder(embed_dim=embed_dim,
                                                 num_heads=num_heads,
                                                 layers=layers,
                                                 attn_dropout=attn_dropout,
                                                 relu_dropout=relu_dropout,
                                                 res_dropout=res_dropout,
                                                 embed_dropout=embed_dropout,
                                                 attn_mask=attn_mask)

        self.cls_head = nn.Sequential(
            nn.Linear(self.fusion_encoder.config.hidden_size, self.fusion_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.fusion_encoder.config.hidden_size, 3)
        )

    def forward(self, image, text, target, train = True):
        
        image_embeds_old = self.visual_encoder(image) #vit提取图片向量
        image_atts = torch.ones(image_embeds_old.size()[:-1],dtype=torch.long).to(image.device) #一个长为矩阵行数的张量 转换数据类型

        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text')  #bert提取文本向量           
        text_embeds_old = text_output.last_hidden_state #bert最后一层作为向量

        m_text_feat_tran = text_embeds_old.clone().transpose(1, 2)
        m_image_feat_tran = image_embeds_old.clone().transpose(1, 2)

        m_text_feat = self.conv1d_t(m_text_feat_tran).permute(2, 0, 1)
        m_image_feat = self.conv1d_i(m_image_feat_tran).permute(2, 0, 1)

        tt_embeds = self.trans_ttchange(m_text_feat, m_text_feat, m_text_feat).permute(1, 0, 2)
        ii_embeds = self.trans_iichange(m_image_feat, m_image_feat, m_image_feat).permute(1, 0, 2)
        

        i2t_embeds = self.trans_itchange(m_text_feat, m_image_feat, m_image_feat).permute(1, 0, 2)
        t2i_embeds = self.trans_tichange(m_image_feat, m_text_feat, m_text_feat).permute(1, 0, 2)   
        
        # text_embeds = self.trans_ttchange(m_text_feat, m_text_feat, m_text_feat).permute(1, 0, 2)
        # image_embeds = self.trans_iichange(m_image_feat, m_image_feat, m_image_feat).permute(1, 0, 2)        

        text_embeds = torch.cat((tt_embeds,t2i_embeds), 1)
        image_embeds = torch.cat((ii_embeds,i2t_embeds), 1)

        # 对图像、文本向量做自注意力
        text_atts = torch.ones(text_embeds.size()[:-1],dtype=torch.long).to(image.device)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        output_pos = self.fusion_encoder.bert(encoder_embeds = text_embeds, 
                                       attention_mask = text_atts,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       mode = 'fusion',
                                      )  # bert作为融合模型      
        
        pre_feat = output_pos.last_hidden_state[:,0,:]
        prediction = self.cls_head(pre_feat)
        # prediction = F.softmax(prediction_s, dim=1)

        # loss = F.cross_entropy(prediction_s, target)  
        pre_weight = torch.tensor([1., 1., 1.]).to(image.device)
        # 6 6 1
        # 6 4 1
        loss = F.cross_entropy(prediction, target, weight = pre_weight)
        # loss = focal_loss_zhihu(prediction, target)

        if train:
            return loss
        else:
            return pre_feat, prediction
            # return prediction
        # return output_pos.last_hidden_state[:,0,:]       
        
