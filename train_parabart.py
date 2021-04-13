import os, argparse, pickle, h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from utils import Timer, make_path, deleaf
from pprint import pprint
from tqdm import tqdm
from transformers import BartTokenizer, BartConfig, BartModel
from parabart import ParaBart

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="./model/")
parser.add_argument('--cache_dir', type=str, default="./bart-base/")
parser.add_argument('--data_dir', type=str, default="./data/")
parser.add_argument('--max_sent_len', type=int, default=40)
parser.add_argument('--max_synt_len', type=int, default=160)
parser.add_argument('--word_dropout', type=float, default=0.2)
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--valid_batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--fast_lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--log_interval', type=int, default=1000)
parser.add_argument('--temp', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
pprint(vars(args))
print()

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

def train(epoch, dataset, model, tokenizer, optimizer, args):
    timer = Timer()
    n_it = len(train_loader)
    optimizer.zero_grad()

    for it, idxs in enumerate(train_loader):
        total_loss = 0.0
        adv_total_loss = 0.0	    
        model.train()
 
        sent1_token_ids = dataset['sent1'][idxs].cuda()
        synt1_token_ids = dataset['synt1'][idxs].cuda()
        sent2_token_ids = dataset['sent2'][idxs].cuda()
        synt2_token_ids = dataset['synt2'][idxs].cuda()
        synt1_bow = dataset['synt1bow'][idxs].cuda()
        synt2_bow = dataset['synt2bow'][idxs].cuda()

        # optimize adv        
        # sent1 adv
        outputs = model.forward_adv(sent1_token_ids)
        targs = synt1_bow
        loss = adv_criterion(outputs, targs)
        loss.backward()
        adv_total_loss += loss.item()

        
        # sent2 adv
        outputs = model.forward_adv(sent2_token_ids)
        targs = synt2_bow
        loss = adv_criterion(outputs, targs)
        loss.backward()
        adv_total_loss += loss.item()
              
        if (it+1) % args.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if epoch > 1:
                adv_optimizer.step()
            adv_optimizer.zero_grad()

        # optimize model
        # sent1->sent2 para & sent1 adv        
        outputs, adv_outputs = model(torch.cat((sent1_token_ids, synt2_token_ids), 1), sent2_token_ids)
        targs = sent2_token_ids[:, 1:].contiguous().view(-1)
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        adv_targs = synt1_bow
        loss = para_criterion(outputs, targs) 
        if epoch > 1:
            loss -= 0.1 * adv_criterion(adv_outputs, adv_targs)
        loss.backward()        
        total_loss += loss.item()
        
        # sent2->sent1 para & sent2 adv
        outputs, adv_outputs = model(torch.cat((sent2_token_ids, synt1_token_ids), 1), sent1_token_ids)
        targs = sent1_token_ids[:, 1:].contiguous().view(-1)
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        adv_targs = synt2_bow
        loss = para_criterion(outputs, targs)
        if epoch > 1:
            loss -= 0.1 * adv_criterion(adv_outputs, adv_targs)
        loss.backward()        
        total_loss += loss.item()
        

        if (it+1) % args.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
       
        if (it+1) % args.log_interval == 0 or it == 0:
            para_1_2_loss, para_2_1_loss, adv_1_loss, adv_2_loss = evaluate(model, tokenizer, args)
            valid_loss = para_1_2_loss + para_2_1_loss - 0.1 * adv_1_loss - 0.1 * adv_2_loss
            print("| ep {:2d}/{} | it {:3d}/{} | {:5.2f} s | adv loss {:.4f} | loss {:.4f} | para 1-2 loss {:.4f} | para 2-1 loss {:.4f} | adv 1 loss {:.4f} | adv 2 loss {:.4f} | valid loss {:.4f} |".format(
                    epoch, args.n_epoch, it+1, n_it, timer.get_time_from_last(), adv_total_loss, total_loss, para_1_2_loss, para_2_1_loss, adv_1_loss, adv_2_loss, valid_loss))
            
			
            
def evaluate(model, tokenizer, args):
    model.eval()
    para_1_2_loss = 0.0
    para_2_1_loss = 0.0
    adv_1_loss = 0.0
    adv_2_loss = 0.0
    with torch.no_grad():
        for idxs in valid_loader:
            
            sent1_token_ids = dataset['sent1'][idxs].cuda()
            synt1_token_ids = dataset['synt1'][idxs].cuda()
            sent2_token_ids = dataset['sent2'][idxs].cuda()
            synt2_token_ids = dataset['synt2'][idxs].cuda()
            synt1_bow = dataset['synt1bow'][idxs].cuda()
            synt2_bow = dataset['synt2bow'][idxs].cuda()
            
            outputs, adv_outputs = model(torch.cat((sent1_token_ids, synt2_token_ids), 1), sent2_token_ids)
            targs = sent2_token_ids[:, 1:].contiguous().view(-1)
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            adv_targs = synt1_bow
            para_1_2_loss += para_criterion(outputs, targs) 
            adv_1_loss += adv_criterion(adv_outputs, adv_targs)
            
            outputs, adv_outputs = model(torch.cat((sent2_token_ids, synt1_token_ids), 1), sent1_token_ids)
            targs = sent1_token_ids[:, 1:].contiguous().view(-1)
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            adv_targs = synt2_bow
            para_2_1_loss += para_criterion(outputs, targs) 
            adv_2_loss += adv_criterion(adv_outputs, adv_targs)

    return para_1_2_loss / len(valid_loader), para_2_1_loss / len(valid_loader), adv_1_loss / len(valid_loader), adv_2_loss / len(valid_loader)


def prepare_dataset(para_data, tokenizer, num):
    sents1 = list(para_data['train_sents1'][:num])
    synts1 = list(para_data['train_synts1'][:num])
    sents2 = list(para_data['train_sents2'][:num])
    synts2 = list(para_data['train_synts2'][:num])

    sent1_token_ids = torch.ones((num, args.max_sent_len+2), dtype=torch.long) 
    sent2_token_ids = torch.ones((num, args.max_sent_len+2), dtype=torch.long)    		
    synt1_token_ids = torch.ones((num, args.max_synt_len+2), dtype=torch.long) 
    synt2_token_ids = torch.ones((num, args.max_synt_len+2), dtype=torch.long)
    synt1_bow = torch.ones((num, 74))
    synt2_bow = torch.ones((num, 74))
        
    bsz = 64
    
    for i in tqdm(range(0, num, bsz)):
        sent1_inputs = tokenizer(sents1[i:i+bsz], padding='max_length', truncation=True, max_length=args.max_sent_len+2, return_tensors="pt")
        sent2_inputs = tokenizer(sents2[i:i+bsz], padding='max_length', truncation=True, max_length=args.max_sent_len+2, return_tensors="pt")
        sent1_token_ids[i:i+bsz] = sent1_inputs['input_ids']
        sent2_token_ids[i:i+bsz] = sent2_inputs['input_ids']

    for i in tqdm(range(num)):
        synt1 = ['<s>'] + deleaf(synts1[i]) + ['</s>']
        synt1_token_ids[i, :len(synt1)] = torch.tensor([synt_vocab[tag] for tag in synt1])[:args.max_synt_len+2]
        synt2 = ['<s>'] + deleaf(synts2[i]) + ['</s>']
        synt2_token_ids[i, :len(synt2)] = torch.tensor([synt_vocab[tag] for tag in synt2])[:args.max_synt_len+2]
        
        for tag in synt1:
            if tag != '<s>' and tag != '</s>':
                synt1_bow[i][synt_vocab[tag]-3] += 1
        for tag in synt2:
            if tag != '<s>' and tag != '</s>':
                synt2_bow[i][synt_vocab[tag]-3] += 1

    synt1_bow /= synt1_bow.sum(1, keepdim=True)
    synt2_bow /= synt2_bow.sum(1, keepdim=True)
    
    sum = 0
    for i in range(num):
        if torch.equal(synt1_bow[i], synt2_bow[i]):
            sum += 1

    return {'sent1':sent1_token_ids, 'sent2':sent2_token_ids, 'synt1': synt1_token_ids, 'synt2': synt2_token_ids,
            'synt1bow': synt1_bow, 'synt2bow': synt2_bow}

print("==== loading data ====")
num = 1000000
para_data = h5py.File(os.path.join(args.data_dir, 'data.h5'), 'r')

train_idxs, valid_idxs = random_split(range(num), [num-5000, 5000], generator=torch.Generator().manual_seed(args.seed))

print(f"number of train examples: {len(train_idxs)}")
print(f"number of valid examples: {len(valid_idxs)}")

train_loader = DataLoader(train_idxs, batch_size=args.train_batch_size, shuffle=True)
valid_loader = DataLoader(valid_idxs, batch_size=args.valid_batch_size, shuffle=False)

print("==== preparing data ====")
make_path(args.cache_dir)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir=args.cache_dir)

with open('synt_vocab.pkl', 'rb') as f:
    synt_vocab = pickle.load(f)

dataset = prepare_dataset(para_data, tokenizer, num)

print("==== loading model ====")
config = BartConfig.from_pretrained('facebook/bart-base', cache_dir=args.cache_dir)
config.word_dropout = args.word_dropout
config.max_sent_len = args.max_sent_len
config.max_synt_len = args.max_synt_len

bart = BartModel.from_pretrained('facebook/bart-base', cache_dir=args.cache_dir)
model = ParaBart(config)
model.load_state_dict(bart.state_dict(), strict=False)
model.zero_grad()
del bart


no_decay_params = []
no_decay_fast_params = []
fast_params = []
all_other_params = []
adv_no_decay_params = []
adv_all_other_params = []

for n, p in model.named_parameters():
    if 'adv' in n:
        if 'norm' in n or 'bias' in n:
            adv_no_decay_params.append(p)
        else:
            adv_all_other_params.append(p)
    elif 'linear' in n or 'synt' in n or 'decoder' in n:
        if 'bias' in n:
            no_decay_fast_params.append(p)
        else:
            fast_params.append(p)
    elif 'norm' in n or 'bias' in n:
        no_decay_params.append(p)
    else:
        all_other_params.append(p)

optimizer = optim.AdamW([
        {'params': fast_params, 'lr': args.fast_lr},
        {'params': no_decay_fast_params, 'lr': args.fast_lr, 'weight_decay': 0.0},
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': all_other_params}
    ], lr=args.lr, weight_decay=args.weight_decay)

adv_optimizer = optim.AdamW([
        {'params': adv_no_decay_params, 'weight_decay': 0.0},
        {'params': adv_all_other_params}
    ], lr=args.lr, weight_decay=args.weight_decay)

para_criterion = nn.CrossEntropyLoss(ignore_index=model.config.pad_token_id).cuda()
adv_criterion = nn.BCEWithLogitsLoss().cuda()

model = model.cuda()

make_path(args.model_dir)

print("==== start training ====")

for epoch in range(1, args.n_epoch+1):
    train(epoch, dataset, model, tokenizer, optimizer, args)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model_epoch{:02d}.pt".format(epoch)))