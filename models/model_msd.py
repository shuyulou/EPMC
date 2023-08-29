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

class SMF(nn.Module):
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
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        fusion_bert_config = BertConfig.from_json_file(config['fusion_bert_config'])
        
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False) 
        self.fusion_encoder = BertForMaskedLM.from_pretrained(fusion_encoder, config=fusion_bert_config) 

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
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
        
        image_embeds_old = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds_old.size()[:-1],dtype=torch.long).to(image.device)

        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text')       
        text_embeds_old = text_output.last_hidden_state 

        m_text_feat_tran = text_embeds_old.clone().transpose(1, 2)
        m_image_feat_tran = image_embeds_old.clone().transpose(1, 2)

        m_text_feat = self.conv1d_t(m_text_feat_tran).permute(2, 0, 1)
        m_image_feat = self.conv1d_i(m_image_feat_tran).permute(2, 0, 1)

        tt_embeds = self.trans_ttchange(m_text_feat, m_text_feat, m_text_feat).permute(1, 0, 2)
        ii_embeds = self.trans_iichange(m_image_feat, m_image_feat, m_image_feat).permute(1, 0, 2)
        

        i2t_embeds = self.trans_itchange(m_text_feat, m_image_feat, m_image_feat).permute(1, 0, 2)
        t2i_embeds = self.trans_tichange(m_image_feat, m_text_feat, m_text_feat).permute(1, 0, 2)             

        text_embeds = torch.cat((tt_embeds,t2i_embeds), 1)
        image_embeds = torch.cat((ii_embeds,i2t_embeds), 1)

        # self-attention
        text_atts = torch.ones(text_embeds.size()[:-1],dtype=torch.long).to(image.device)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        output_pos = self.fusion_encoder.bert(encoder_embeds = text_embeds, 
                                       attention_mask = text_atts,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       mode = 'fusion',
                                      )  
        
        pre_feat = output_pos.last_hidden_state[:,0,:]
        prediction = self.cls_head(pre_feat)

        pre_weight = torch.tensor([1., 1., 1.]).to(image.device)
        loss = F.cross_entropy(prediction, target, weight = pre_weight)

        if train:
            return loss
        else:
            return pre_feat, prediction
            # return prediction
        
