import sys, io
import numpy as np
import torch
from transformers import BartTokenizer, BartConfig, BartModel
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import pickle, random
from parabart import ParaBart



print("==== loading model ====")
config = BartConfig.from_pretrained('facebook/bart-base', cache_dir='../para-data/bart-base')

model = ParaBart(config)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir='../para-data/bart-base')

model.load_state_dict(torch.load("./model/model.pt", map_location='cpu'))

model = model.cuda()

def build_embeddings(model, tokenizer, sents):
    model.eval()
    embeddings = torch.ones((len(sents), model.config.d_model))
    with torch.no_grad():
        for i, sent in enumerate(sents):            
            sent_inputs = tokenizer(sent, return_tensors="pt")
            sent_token_ids = sent_inputs['input_ids']
            
            sent_embed = model.encoder.embed(sent_token_ids.cuda())
            embeddings[i] = sent_embed.detach().cpu().clone()
    return embeddings

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))



scores = []
labels = []
with open("qqp.pkl", "rb") as f:
    para_split = pickle.load(f)


pos_hard = para_split['pos_hard']
pos = para_split['pos']
neg = para_split['neg']

easy = pos + neg
hard = pos_hard + neg

scores = []
for i in tqdm(range(len(easy))):  
    embeds = build_embeddings(model, tokenizer, [easy[i][0], easy[i][1]])
    score = cosine(embeds[0], embeds[1]) 
    scores.append(score)

scores_hard = []
for i in tqdm(range(len(hard))):  
    embeds = build_embeddings(model, tokenizer, [hard[i][0], hard[i][1]])
    score = cosine(embeds[0], embeds[1]) 
    scores_hard.append(score)

   


best_acc = 0.0
best_thres = 0.0
scores = np.asarray(scores)
labels = [1]*len(pos) + [0]*len(neg)
labels = np.asarray(labels)  
for thres in range(-100, 100, 1):
    thres = thres / 100.0
    preds = scores > thres
    acc = sum(labels == preds)/len(labels)
    if acc > best_acc:
        best_acc = acc
        best_thres = thres
print('easy acc:', best_acc)


best_acc = 0.0
best_thres = 0.0
scores_hard = np.asarray(scores_hard)
labels_hard = [1]*len(pos_hard) + [0]*len(neg)
labels_hard = np.asarray(labels_hard) 
for thres in range(-100, 100, 1):
    thres = thres / 100.0
    preds = scores_hard > thres
    acc = sum(labels_hard == preds)/len(labels_hard)
    if acc > best_acc:
        best_acc = acc
        best_thres = thres
print('hard acc:', best_acc)

