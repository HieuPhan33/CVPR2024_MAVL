# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
from sklearn.metrics import log_loss
import torch.nn as nn
import torch
import math
import numpy as np  
from torch.nn.utils.rnn import pad_sequence
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from .transformer import *
import torchvision.models as models
import torchvision
from einops import rearrange
from transformers import AutoModel
from . import clip
import timm
# from timm.loss import SoftTargetCrossEntropy


'''
args.N
args.d_model
args.res_base_model
args.H 
args.num_queries
args.dropout
args.attribute_set_size
'''


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]
    


class MAVL(nn.Module):

    def __init__(self, config, disease_book, concept_book):
        super(MAVL, self).__init__()
        self.d_model = config['d_model']
        device = disease_book['input_ids'].device
        # ''' book embedding'''
        with torch.no_grad():
            bert_model = self._get_bert_basemodel(config['text_encoder'],freeze_layers = None).to(device)
            ## Location / anatomical terms/ landmark
            self.disease_book = bert_model(input_ids = disease_book['input_ids'],attention_mask = disease_book['attention_mask'])#(**encoded_inputs)
            self.disease_book = self.disease_book.last_hidden_state[:,0,:] # Detailed description of dieases codebook - 75 x 768
            self.n_disease = self.disease_book.shape[0]
            self.n_concepts = concept_book['input_ids'].shape[0] // self.n_disease
            self.concept_book = bert_model(input_ids = concept_book['input_ids'],attention_mask = concept_book['attention_mask'])#(**encoded_inputs)
            self.concept_book = self.concept_book.last_hidden_state[:,0,:] # Detailed description of dieases codebook 
            self.concept_book = self.concept_book[:self.n_disease * self.n_concepts]
            self.concept_book = self.concept_book.view(self.n_disease, self.n_concepts, 768) # 75 x 7 x 768 Support 7 concepts for now
            self.concept_book = torch.cat((self.disease_book.unsqueeze(1), self.concept_book), dim = 1).view(-1, 768)
            self.n_concepts += 1
        self.global_concept_idx = config.get('global_concept_idx', list(range(self.n_concepts))) 
        self.disease_embedding_layer = nn.Linear(768, self.d_model)
        #self.cl_fc = nn.Linear(256,768) # Location embedding
        
        
        ## Index of important diseases
        self.model_name = config["base_model"]
        print("Image feature extractor:", config["base_model"])
        if 'CLIP' in config['base_model']:
            model_name = config['base_model'].replace('CLIP-', '')
            model, _ = clip.load(model_name, device=device, jit=False)
            self.backbone = model.visual
            num_ftrs = int(self.backbone.output_dim)
        else:
            self.backbone, num_ftrs = self._get_basemodel(config['base_model'], pretrained=config['pretrained'], 
                                                          layers=config.get('layers', None))
            if 'resnet' in config['base_model']:
                self.pool = AttentionPool2d(224//16, embed_dim=num_ftrs, num_heads=8, output_dim=self.d_model)
        if ('vit' in self.model_name.lower() or 'timm' in self.model_name.lower()) and self.d_model != 768:
            self.global_l = nn.Linear(768, out_features=self.d_model)
        else:
            self.global_l = None
        # self.global_l = nn.Linear(768, out_features=self.d_model)
        # self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        # self.res_l2 = nn.Linear(num_ftrs, self.d_model)
        self.res_l1 = nn.Sequential(
            nn.Conv2d(num_ftrs, out_channels=num_ftrs, kernel_size=1),
            nn.PReLU()
        )
        self.res_l2 = nn.Conv2d(num_ftrs, out_channels=self.d_model, kernel_size=1)
        self.location_emb = nn.Conv1d(in_channels=self.d_model, out_channels=768, kernel_size=self.n_concepts, stride=self.n_concepts)
        #self.location_emb = nn.ModuleList([nn.Linear(in_channels=256*self.n_concepts, out_channels=768) for _ in len(self.disease_name)])

        


        ###################################
        ''' Query Decoder'''
        ###################################
        self.decoder_name = config['decoder']
        if config['decoder'] == 'slot': 
            self.decoder = SlotAttention(dim=256, n_concepts=self.n_concepts, iters=config['N'], hidden_dim=1024)
        elif config['decoder'] == 'cross':
            self.H = config['H'] 
            decoder_layer = TransformerDecoderLayer(self.d_model, config['H'] , 1024,
                                            0.1, 'relu',normalize_before=True, self_attention=config['self_attention'])
            decoder_norm = nn.LayerNorm(self.d_model)
            self.decoder = TransformerDecoder(decoder_layer, config['N'] , decoder_norm,
                                    return_intermediate=False)

        # Learnable Queries
        #self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout2d(config['dropout'])
        self.bottneck = config.get('bottneck', False)
        d_model = self.d_model
        if self.bottneck:
            self.bottleneck = nn.Conv1d(self.d_model, self.d_model//4, kernel_size=1)
            d_model = self.d_model//4
        
        # Attribute classifier
        self.classifier = nn.Linear(d_model*self.n_concepts, config['attribute_set_size'])



        # # Class classifier
        # self.cls_classifier = nn.Linear(self.d_model,args.num_classes)

        self.apply(self._init_weights)

    def _get_basemodel(self, model_name, pretrained=False, layers=['blocks.9']):
        # try:
        ''' visual backbone'''
        net_dict = {"resnet18": models.resnet18(pretrained=pretrained),
                        "resnet50": models.resnet50(pretrained=pretrained),
                        "ViT-B/16": models.vit_b_16(torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1),
                        "ViT-L/16": models.vit_l_16(torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)}
        if "resnet" in model_name:
            model = net_dict[model_name]
            num_ftrs = int(model.fc.in_features/2)
            backbone = nn.Sequential(*list(model.children())[:-3])
        elif 'timm-' in model_name:
            model_name = model_name.replace('timm-', '')
            model = timm.create_model(model_name, pretrained=True)
            backbone = create_feature_extractor(model, return_nodes={layers[0]: 'layer'})
            num_ftrs = backbone.patch_embed.proj.out_channels
        elif "ViT" in model_name:
            model = net_dict[model_name]
            backbone = create_feature_extractor(model, return_nodes={'encoder.ln': 'layer'}) 
            num_ftrs = model.hidden_dim
        return backbone, num_ftrs
        # except:
        #     raise ("Invalid model name. Check the config file and pass one of: resnet18, resnet50, ViT-B/16, ViT-L/16, timm-model")

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model
    
    def image_encoder(self, xis):
        #patch features
        """
        16 torch.Size([16, 1024, 14, 14])
        torch.Size([16, 196, 1024])
        torch.Size([3136, 1024])
        torch.Size([16, 196, 256])
        """
        batch_size, _, H, W = xis.shape
        res_fea = self.backbone(xis) #batch_size,feature_size,patch_num,patch_num
        if 'vit' in self.model_name.lower() or 'timm' in self.model_name.lower():
            global_fea = res_fea['layer'][:, 0, :]
            if self.global_l is not None:
                global_fea = self.global_l(global_fea)
            res_fea = res_fea['layer'][:, 1:, :]
            h = int(np.sqrt(res_fea.shape[1]))
            res_fea = res_fea.permute(0, 2, 1).contiguous().view(batch_size, -1, h, h)
        else:
            global_fea = self.pool(res_fea)
        #global_fea = self.global_l(global_fea) EXPLORE
        #batch_size,num,feature_size
        # h = h.squeeze()
        x = self.res_l1(res_fea)
        x = self.res_l2(x)
        return x, global_fea

    def forward(self, images, return_immediate=False):
        '''
        images: visual images - [B, 1, H, W]
        labels: presence label of n_disease=75 diseases in the report [-1, 0, 1] - [B, 75]
        sample_index: queue of location idxs for contrastive learning: [B, 8]
        '''
        # labels batch,51,75 binary_label batch,75 sample_index batch,index
        B = images.shape[0]
        device = images.device
        ''' Visual Backbone '''
        x, x_global = self.image_encoder(images) #batch_size,patch_num,dim

        #query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
        
        ## Concept learning
        # self.concept_book = self.concept_learning(self.disease_book)
        # self.concept_book = self.concept_book.view(n_queries*self.n_concepts, -1) # N_disease*n_concepts, 768
        query_embed_ = self.disease_embedding_layer(self.concept_book) 
        logits = torch.matmul(F.normalize(x_global, dim=-1), F.normalize(query_embed_, dim=-1).T) # B x N_disease*n_concepts
        #logits = F.cosine_similarity(F.normalize(x_global, dim=-1)[:, None, :], F.normalize(query_embed_, dim=-1)[None, :, :], dim=-1)
        logits = logits.view(B, self.n_disease, self.n_concepts).contiguous().permute(0, 2, 1).mean(1)

        ## Extract concept embeddings of n_disease*n_concepts from the visual features
        if self.decoder_name == 'cross':
            b, d, h, w = x.shape
            x = x.permute((0, 2, 3, 1))
            x = x.contiguous().view(b, h*w, -1).permute(1, 0, 2)
            query_embed = query_embed_.unsqueeze(1).repeat(1, B, 1) # 75 x B x dim
            
        features, att = self.decoder(query_embed, x)
        # feature shape is B x (n_disease*n_concepts) x dim
        # att shape is B x (n_disease*n_concepts) x N_pixels
        out = self.dropout_feas(features)
        if self.decoder_name == 'cross':
            out = out.permute(1, 2, 0)
        else:
            out = out.permute(0, 2, 1)
        ## B x dim x (n_disease*n_concepts)

        # if mode == 'max':
        #     att = [a.view(B, self.n_disease, self.n_concepts, -1).max(2)[0] for a in att]
        # elif mode == 'avg':
        #     att = [a.view(B, self.n_disease, self.n_concepts, -1).mean(2) for a in att]
        # elif mode == 'global':
        #     att = [a.view(B, self.n_disease, self.n_concepts, -1)[:, :, 0,:] for a in att]

        
        ## Output 3: Predict disease presence
        if self.bottneck:
            out = self.bottleneck(out)
        out = out.permute(0, 2, 1) ## Debug
        out = out.contiguous().view(B, self.n_disease, -1)
        x = self.classifier(out) # B x N_disease x 2
        if return_immediate:
            return x, att, logits
        return x, att

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

if __name__ == '__main__':
    import ruamel_yaml as yaml
    import json
    from models.tokenization_bert import BertTokenizer
    def get_tokenizer(tokenizer,target_text):
        target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length=128,return_tensors="pt")
        return target_tokenizer
    config = yaml.load(open('configs/Pretrain_MedSLIP.yaml', 'r'), Loader=yaml.Loader)
    print("Creating book")
    json_book = json.load(open(config['disease_book'],'r'))
    disease_book = [json_book[i] for i in json_book]
    concepts = json.load(open(config['concept_book'], 'r'))
    concepts_book = sum(concepts.values(), [])

    ana_book = [ 'It is located at ' + i for i in ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
            'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
            'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
            'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
            'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
            'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
            'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
            'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other']]
    ana_book.append('It is not present')
    print("Number of anatomies:", len(ana_book))
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    ana_book_tokenizer = get_tokenizer(tokenizer,ana_book)
    disease_book_tokenizer = get_tokenizer(tokenizer,disease_book)
    concepts_book_tokenizer = get_tokenizer(tokenizer, concepts_book)

    model = MedSLIP(config,ana_book_tokenizer, disease_book_tokenizer, concepts_book_tokenizer, mode = 'train')
    images = torch.rand(2, 3, 224, 224)
    labels = torch.randint(0, 2, (2, 75))
    indices = torch.randint(0, 20, (2, 75, 8))
    _ = model(images, labels, indices)