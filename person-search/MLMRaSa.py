from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from vit import VisionTransformer
from xbert import BertConfig, BertForMaskedLM
from tokenization_bert import BertTokenizer
import numpy as np
#! -------- all variable -------------
text_encoder = 'bert-base-uncased'
bert_config = BertConfig.from_json_file('config.json')
text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder,config=bert_config)
tokenizer = BertTokenizer.from_pretrained(text_encoder)
text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)


#! --------------- my function ----------------------------
def mask(input_ids, vocab_size, targets=None, masked_indices = None, probability_matrix = None):
    if masked_indices is None:
        masked_indices = torch.bernoulli(probability_matrix).bool()
    masked_indices[input_ids==tokenizer.pad_token_id] = False
    masked_indices[input_ids==tokenizer.cls_token_id] = False
    if targets is not None:
        targets[~masked_indices] = -100
        
    #! replace 80% mask input token with [mask]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape,0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id
    
    #! 10% replace with random word:
    indices_random = torch.bernoulli(torch.full(input_ids.shape,0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    #! 10% còn lại keep the masked input token unchange:
    if targets is not None:
        return input_ids, targets
    else:
        return input_ids
    



#! -----------------------------------------------------------implement loss_mlm_output truoc:----------------------------------
from vit import VisionTransformer
# image1 = torch.tensor(np.random.random(size=([10,3,50,50])),dtype=torch.float64)
# image2 = torch.tensor(np.random.random(size=([10,3,50,50])),dtype=torch.float64)
# class Image1(torch.Tensor):
#     def __init__(self,size,dtype=torch.float):
#         super().__init__()
#         self.data = torch.randn(size=(size),dtype=dtype)  
# image1 = Image1(size=(10,3,50,50))
# class Image2(torch.Tensor):
#     def __init__(self,size,dtype=torch.float):
#         super().__init__()
#         self.data = torch.randn(size=(size),dtype=dtype)
# image2 = Image2(size=(10,3,50,50))
image1 = torch.randn(size=(10,3,50,50),dtype=torch.float)
image2 = torch.randn(size=(10,3,50,50),dtype=torch.float)

visual_encoder = VisionTransformer(
            img_size=50, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), )

visual_encoder_m = VisionTransformer(
            img_size=50, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), )
image_embeds = visual_encoder(image1)
image_embeds_m = visual_encoder_m(image2)

image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image1.device)
#! find labels and input_ids: 
#input_ids = text1.input_ids.clone()
input_ids = torch.tensor([[ 101, 1996, 2158, 2003, 3061, 2007],
        [ 101, 1996, 2450, 2038, 2146, 2601],
        [ 101, 1996, 2611, 2003, 4147, 2014],
        [ 101, 2023, 2450, 2003, 4147, 1037],
        [ 101, 1037, 2158, 4147, 1037, 2317],
        [ 101, 1037, 2711, 2003, 3788, 2185],
        [ 101, 1996, 2158, 2003, 4147, 1037],
        [ 101, 2023, 2711, 2038, 2460, 2601],
        [ 101, 1996, 2711, 2003, 4147, 1037],
        [ 101, 1996, 2450, 2038, 2601, 2606]])
labels = input_ids.clone()
probability_matrix = torch.full(labels.shape,0.15)
input_ids, labels = mask(input_ids=input_ids,vocab_size=30522, targets=labels,probability_matrix=probability_matrix)
print(input_ids)
print(labels)

#! find prediction and alpha:


# #! text1.attention_mask:
# class Attention_mask(torch.Tensor):
#     def __init__(self,size,dtype=torch.int64):
#         super().__init__()
#         self.data = torch.ones(size,dtype=dtype)

# attention_mask = Attention_mask(size=(10,6))
attention_mask = torch.ones(size=(10,6),dtype=torch.int64)
# class Image_atts(torch.Tensor):
#     def __init__(self,size,dtype=torch.int64):
#         super().__init__()
#         self.data = torch.ones(size,dtype=dtype)
# image_atts = Image_atts(size=(10,10))

logits_m = text_encoder_m(
    input_ids,
    attention_mask = attention_mask,
    encoder_hidden_states = image_embeds,
    encoder_attention_mask = image_atts,
    return_dict = True,
    return_logits = True
)
prediction = F.softmax(logits_m,dim=-1)

# #! image_res = 50

alpha = 0.4

mlm_output = text_encoder(input_ids,
                        attention_mask=attention_mask,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                        labels=labels,
                        soft_labels=prediction,
                        alpha=alpha
                        )
loss_mlm = mlm_output.loss
print("This is my loss mlm: {}".format(loss_mlm))


#! ----------------------------implement loss mrtd------------------------------
mrtd_input_ids = input_ids.clone()
probability_matrix = torch.full(size=labels.shape,fill_value=0.3)
mrtd_input_ids = mask(input_ids = mrtd_input_ids,vocab_size=30522,probability_matrix=probability_matrix)
#! momentum module use as generator:
mrtd_logits_m = text_encoder_m(
    mrtd_input_ids,
    attention_mask = attention_mask,
    encoder_hidden_states = image_embeds_m,
    encoder_attention_mask = image_atts,
    return_dict = True,
    return_logits = True
)
weights = F.softmax(mrtd_logits_m,dim=-1)
def mrtd_mask_modeling(mrtd_input_ids, ori_input_ids, attention_mask, weights):
    bs = mrtd_input_ids.size(0)
    weights = weights.view(-1,weights.size(-1))
    pred = torch.multinomial(weights,1).view(bs,-1)
    pred[:,0] = tokenizer.cls_token_id
    
    # pad token id is 0
    mrtd_input_ids = pred * attention_mask
    mrtd_labels = (pred != ori_input_ids)*(attention_mask)
    mrtd_labels[mrtd_input_ids == tokenizer.pad_token_id] = -100
    mrtd_labels[mrtd_input_ids == tokenizer.cls_token_id] = -100
    return mrtd_input_ids,mrtd_labels


#! tham số ori_input_ids = text1.input_id
text1_input_ids = torch.tensor([[  101,  1996,  2158,  2003,  5102,  1999],
        [  101,  1996,  2158,  2003,  4147,  1037],
        [  101,  1996,  2611, 11651,  2422,  6910],
        [  101,  1037,  2158,  4147,  1037,  2417],
        [  101,  1037,  2146,  2601, 10681,  2450],
        [  101,  1996,  2879,  2038,  2460,  2304],
        [  101,  2023,  2450,  2003,  4147,  1037],
        [  101,  2023,  2711,  2003,  5710,  2013],
        [  101,  1037,  2450,  4147,  1037,  6379],
        [  101,  2023,  2402,  2711,  2038,  3239]])
mrtd_input_ids, mrtd_labels = mrtd_mask_modeling(mrtd_input_ids,text1_input_ids,attention_mask,weights)


print("This is my mrtd_input_ids: {}".format(mrtd_input_ids))
print("This is my mrtd_labels: {}".format(mrtd_labels))


output_mrtd = text_encoder.bert(
    mrtd_input_ids,
    attention_mask = attention_mask,
    encoder_hidden_states = image_embeds,
    encoder_attention_mask = image_atts,
    return_dict = True
)

text_width = text_encoder.config.hidden_size
mrtd_head = nn.Linear(text_width,2)
mrtd_output = mrtd_head(output_mrtd.last_hidden_state.view(-1,text_width))

loss_mrtd = F.cross_entropy(mrtd_output,mrtd_labels.view(-1))
print("This is my loss_mrtd: {}".format(loss_mrtd))
