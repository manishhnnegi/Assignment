# class impoorts 
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

#import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import torch
import torch.nn as nn
from config import get_config, latest_weights_file_path
#from train import get_model, get_ds, run_validation
# from translate import translate

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
config = get_config()

# configs
config
f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}"
# dataset download
ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

ds_raw
ds_raw.features
ds_raw['id'][:3],ds_raw['translation'][:3]
import pandas as pd
pd.DataFrame(ds_raw)
pd.DataFrame(ds_raw['translation'])
## tokenizer and its training
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
ds =ds_raw
lang = config['lang_src']
for i in get_all_sentences(ds, lang):
    break
i
tokenizer_path = Path(config['tokenizer_file'].format(lang))

tokenizer.pre_tokenizer = Whitespace()
trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
tokenizer.save(str(tokenizer_path))
tokenizer
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

# Build tokenizers
tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
# train test spilt
# Keep 90% for training, 10% for validation
train_ds_size = int(0.9 * len(ds_raw))
val_ds_size = len(ds_raw) - train_ds_size
train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

for i in train_ds_raw:
    break

i
train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

train_ds
for i in train_ds:
    break

i['encoder_input'].shape
i['decoder_input'].shape
i['encoder_mask'].shape

i
i['decoder_mask']
i['decoder_mask'].shape, i['encoder_input'].shape, i['decoder_input'].shape
i['src_text'], i['tgt_text'],i['label'].shape
i['label']
# Find the maximum length of each sentence in the source and target sentence
max_len_src = 0
max_len_tgt = 0

for item in ds_raw:
    src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
    tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

print(f'Max length of source sentence: {max_len_src}')
print(f'Max length of target sentence: {max_len_tgt}')
item
len(src_ids), len(tgt_ids)
# dataloader
train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

# traing steps
device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
if (device == 'cuda'):
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
elif (device == 'mps'):
    print(f"Device name: <mps>")
else:
    print("NOTE: If you have a GPU, consider using it for training.")
    print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
    print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
device = torch.device(device)

# Make sure the weights folder exists
Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
# Make sure the weights folder exists
Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
f"{config['datasource']}_{config['model_folder']}"
config['preload']
import torch.optim as optim

#train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
# # Tensorboard
# writer = SummaryWriter(config['experiment_name'])

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

# If the user specified a model to preload before training, load it
initial_epoch = 0
global_step = 0
preload = config['preload']
model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
if model_filename:
    print(f'Preloading model {model_filename}')
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']
else:
    print('No model to preload, starting from scratch')

loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

config['num_epochs'] = 5
config



for epoch in range(initial_epoch, config['num_epochs']):
    torch.cuda.empty_cache()
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
    for batch in batch_iterator:

        encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
        decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
        encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
        proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

        # Compare the output with the label
        label = batch['label'].to(device) # (B, seq_len)

        # Compute the loss using a simple cross entropy
        loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        # Log the loss
        # writer.add_scalar('train loss', loss.item(), global_step)
        # writer.flush()

        # Backpropagate the loss
        loss.backward()

        # Update the weights
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1

    # Run validation at the end of every epoch
    #run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

    # Save the model at the end of every epoch
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)

# calculation of no of parametrs in a model 
model
list(model.parameters())
# total no of parameters

# totoal no of numerics in the weight metrix as well as in bais matrix contributs in the total no of parameters

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
pytorch_total_params
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
count_parameters(model1)
# Processing Epoch 00: 100%|██████████████████████████████████████████████| 3638/3638 [09:33<00:00,  6.34it/s, loss=5.908]
# Processing Epoch 01: 100%|██████████████████████████████████████████████| 3638/3638 [09:18<00:00,  6.51it/s, loss=6.069]
# Processing Epoch 02: 100%|██████████████████████████████████████████████| 3638/3638 [09:30<00:00,  6.38it/s, loss=5.343]
# Processing Epoch 03: 100%|██████████████████████████████████████████████| 3638/3638 [09:19<00:00,  6.50it/s, loss=5.143]
# Processing Epoch 04: 100%|██████████████████████████████████████████████| 3638/3638 [09:40<00:00,  6.27it/s, loss=3.908]
# inferencing
val_dataloader
# validation
from train import *
run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), 0, None, num_examples=10)
# inferencing
from translate import translate
t = translate("Why do I need to translate this?")
t
# inferencing explainination
sentence = "Why do I need to translate this?"
seq_len = config['seq_len']
source = tokenizer_src.encode(sentence)

# source = torch.cat([
#     torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
#     torch.tensor(source.ids, dtype=torch.int64),
#     torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
#     torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
# ], dim=0).to(device)
# source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
seq_len
source
source.ids
[tokenizer_src.token_to_id('[SOS]')], [tokenizer_src.token_to_id('[EOS]')]
[tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2)
source = torch.cat([
    torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
    torch.tensor(source.ids, dtype=torch.int64),
    torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
    torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
], dim=0).to(device)
source
source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
(source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int()

model.eval()
with torch.no_grad():
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)


    
    # Print the source sentence and target start prompt
    if label != "": print(f"{f'ID: ':>12}{id}") 
    print(f"{f'SOURCE: ':>12}{sentence}")
    if label != "": print(f"{f'TARGET: ':>12}{label}") 
    print(f"{f'PREDICTED: ':>12}", end='')

    
    # Generate the translation word by word
    while decoder_input.size(1) < seq_len:

        print(f"decoder_input--------------------------------->{decoder_input}")
        
        # build mask for target and calculate output
        decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
        
        print(f"mask--------------------------------->{decoder_mask}")
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        print(f"output--------------------------------->{out.shape}")
    
        # project next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        print(f"next_word--------------------------------->{next_word}")
        
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)


        print(f"{tokenizer_tgt.decode([next_word.item()])}")
        # print the translated word
        print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')
    
        # break if we predict the end of sentence token
        if next_word == tokenizer_tgt.token_to_id('[EOS]'):
            break
# different type of mask used

    1. padding mask used in encoder 
    2. causal mask used in decoder
    
    causal mask is used during training only its not used in inferencing process stilll doubt is there code is used it for infrencing time but the structure of mask is different
.masked_fill_(mask == 0, -1e9)
decoder_mask
decoder_mask.masked_fill_(decoder_mask == 0, -1e9)
causal_mask(decoder_input.size(1)).type_as(source_mask)
torch.triu(torch.ones(8, 8) * float('-inf'), diagonal=1)
torch.triu(torch.ones(8,8)-float('inf'), diagonal=1)+torch.tril(torch.ones(8,8)-float('inf'), diagonal=-1)
# pytorch transformer decoder class

    it takes encoder input as memory decoder input as tgt apart from it it has optional parametrs like tgt_mask and other
decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)
transformer_decoder
tgt.shape
out.shape
decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
memory = torch.rand(32, 10, 512)
tgt = torch.rand(32, 20, 512)
out = decoder_layer(tgt, memory)
tgt.shape, out.shape, memory.shape
# sources
# # doubbt related to masking 
# https://ai.stackexchange.com/questions/42116/transformer-decoder-causal-masking-during-inference
# https://ai.stackexchange.com/questions/23889/what-is-the-purpose-of-decoder-mask-triangular-mask-in-transformer

# https://pytorch.org/tutorials/beginner/translation_transformer.html
# greedy decoder

model.eval()
with torch.no_grad():
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)


    
    # Print the source sentence and target start prompt
    if label != "": print(f"{f'ID: ':>12}{id}") 
    print(f"{f'SOURCE: ':>12}{sentence}")
    if label != "": print(f"{f'TARGET: ':>12}{label}") 
    print(f"{f'PREDICTED: ':>12}", end='')

    
    # Generate the translation word by word
    while decoder_input.size(1) < seq_len:

        print(f"decoder_input--------------------------------->{decoder_input}")
        
        # build mask for target and calculate output
        decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
        
        print(f"mask--------------------------------->{decoder_mask}")
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        print(f"output--------------------------------->{out.shape}")
    
        # project next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        print(f"next_word--------------------------------->{next_word}")
        
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)


        print(f"{tokenizer_tgt.decode([next_word.item()])}")
        # print the translated word
        print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')
    
        # break if we predict the end of sentence token
        if next_word == tokenizer_tgt.token_to_id('[EOS]'):
            break
sos_idx = tokenizer_tgt.token_to_id('[SOS]')
eos_idx = tokenizer_tgt.token_to_id('[EOS]')

# Precompute the encoder output and reuse it for every step
encoder_output = model.encode(source, source_mask)
# Initialize the decoder input with the sos token
decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

# Create a candidate list
candidates = [(decoder_initial_input, 1)]

while True:

    # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
    if any([cand.size(1) == max_len for cand, _ in candidates]):
        break

    # Create a new list of candidates
    new_candidates = []

    for candidate, score in candidates:

        # Do not expand candidates that have reached the eos token
        if candidate[0][-1].item() == eos_idx:
            continue

        # Build the candidate's mask
        candidate_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device)
        # calculate output
        out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
        # get next token probabilities
        prob = model.project(out[:, -1])






        
        # get the top k candidates
        topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
        for i in range(beam_size):
            # for each of the top k candidates, get the token and its probability
            token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
            token_prob = topk_prob[0][i].item()
            # create a new candidate by appending the token to the current candidate
            new_candidate = torch.cat([candidate, token], dim=1)
            # We sum the log probabilities because the probabilities are in log space
            new_candidates.append((new_candidate, score + token_prob))

    # Sort the new candidates by their score
    candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
    # Keep only the top k candidates
    candidates = candidates[:beam_size]

    # If all the candidates have reached the eos token, stop
    if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
        break

# Return the best candidate
return candidates[0][0].squeeze()
model.eval()
with torch.no_grad():
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)


    
    # Print the source sentence and target start prompt
    if label != "": print(f"{f'ID: ':>12}{id}") 
    print(f"{f'SOURCE: ':>12}{sentence}")
    if label != "": print(f"{f'TARGET: ':>12}{label}") 
    print(f"{f'PREDICTED: ':>12}", end='')

    
    # Generate the translation word by word
    while decoder_input.size(1) < seq_len:

        print(f"decoder_input--------------------------------->{decoder_input}")
        
        # build mask for target and calculate output
        decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
        
        print(f"mask--------------------------------->{decoder_mask}")
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        print(f"output--------------------------------->{out.shape}")
    
        # project next token
        prob = model.project(out[:, -1])
