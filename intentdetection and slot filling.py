# data loader
import os 
intent_labels = [i.strip() for i in open(os.path.join("intent_label.txt"), "r", encoding = "utf-8").readlines()]
intent_labels
slot_labels = [i.strip() for i in open(os.path.join("slot_label.txt"), "r", encoding = "utf-8").readlines()]
slot_labels
len(intent_labels), len(slot_labels)
def _read_file(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line.strip())
        return lines
with open("seq.in" , "r", encoding = "utf-8") as f:
   lines =  [line.strip() for line in f]
input_text_file = lines

with open("label" , "r", encoding = "utf-8") as f:
   lines =  [line.strip() for line in f]

intent_label_file = lines

with open("seq.out" , "r", encoding = "utf-8") as f:
   lines =  [line.strip() for line in f]

slot_label_file = lines



input_text_file
intent_label_file
slot_label_file
len(input_text_file), len(intent_label_file), len(slot_label_file)
texts = input_text_file
intents = intent_label_file
slots = slot_label_file

# example creation from text label and intent
# # intent_label, slot_label, words


#     def _create_examples(self, texts, intents, slots, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
#             guid = "%s-%s" % (set_type, i)
#             # 1. input_text
#             words = text.split()  # Some are spaced twice
#             # 2. intent
#             intent_label = (
#                 self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
#             )
#             # 3. slot
#             slot_labels = []
#             for s in slot.split():
#                 slot_labels.append(
#                     self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK")
#                 )

#             assert len(words) == len(slot_labels)
#             examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
#         return examples
for i ,(text,slot,intent) in enumerate(zip(texts,slots,intents)):
    break
 i, (text, intent, slot) 
import json
import copy
class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
examples = []
for i ,(text,slot,intent) in enumerate(zip(texts,slots,intents)):
    words = text.split()
    intent_label = (intent_labels.index(intent)  if intent in intent_labels else intent_labels.index('UNK'))
    slot_labels_tk = []
    for slot_tk in slot.split():
        slot_labels_tk.append(slot_labels.index(slot_tk) if slot_tk in slot_labels else slot_labels.index('UNK'))
    examples.append(InputExample(intent_label = intent_label,slot_labels = slot_labels_tk, words = words,guid = i))
examples
examples[-1]
texts[-1]
# tokenizer loading create features from excamples
from transformers import RobertaTokenizer, RobertaConfig

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer
cls_token = tokenizer.cls_token
sep_token = tokenizer.sep_token
unk_token = tokenizer.unk_token
pad_token_id = tokenizer.pad_token_id
pad_token_label_id=-100,
cls_token_segment_id=0,
pad_token_segment_id=0,
sequence_a_segment_id=0,
mask_padding_with_zero=True,
examples
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    pad_token_label_id=-100,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        
        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len
        )

        intent_label_id = int(example.intent_label)


        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                intent_label_id=intent_label_id,
                slot_labels_ids=slot_labels_ids,
            )
        )

    return features

max_seq_len = 60

features = convert_examples_to_features(
            examples,  max_seq_len, tokenizer, pad_token_label_id=0
        )
features
from torch.utils.data import TensorDataset
import torch
# Convert to Tensors and build dataset
all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

dataset = TensorDataset(
    all_input_ids, all_attention_mask, all_token_type_ids, all_intent_label_ids, all_slot_labels_ids
)
# final dataset
dataset
for i in dataset:
    break
i
len(dataset)
# model training
intent_labels
slot_labels
pad_token_label_id = 0
# model config
config = RobertaConfig.from_pretrained("roberta-base")

config
# model class
import torch.nn as nn


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
#joint bert

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

class JointRoberta(RobertaPreTrainedModel):
    def __init__(self, config, intent_label_lst, slot_label_lst):
        super(JointRoberta,self).__init__(config)
        self.use_crf = True
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained phobert
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, 0.1)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, 0.1)

        if self.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.roberta(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss


        use_crf = True
        ignore_index = 0
        slot_loss_coef = 1
        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
torch.cuda.set_device(0)
print(torch.cuda.current_device())
no_cuda = False
device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
device
model = JointRoberta(config = config, intent_label_lst=intent_labels,
                slot_label_lst=slot_labels)


model = JointRoberta.from_pretrained(config = config, intent_label_lst=intent_labels,
                slot_label_lst=slot_labels)
model = JointRoberta.from_pretrained(config = config, intent_label_lst=intent_labels,
                slot_label_lst=slot_labels, pretrained_model_name_or_path = "roberta-base")
model.to(device)
# training steps
#training
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
dataset
train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=16)
t_total = len(train_dataloader) // 1 * 5
t_total
from transformers import AdamW, get_linear_schedule_with_warmup

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)


from transformers import AdamW, get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=t_total
)

import os

import numpy as np
import torch


class EarlyStopping:
"""Early stops the training if validation loss doesn't improve after a given patience."""

def __init__(self, patience=7, verbose=False):
    """
    Args:
        patience (int): How long to wait after last time validation loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
                        Default: False
    """
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf

def __call__(self, val_loss, model, args):
    if args.tuning_metric == "loss":
        score = -val_loss
    else:
        score = val_loss
    if self.best_score is None:
        self.best_score = score
        self.save_checkpoint(val_loss, model, args)
    elif score < self.best_score:
        self.counter += 1
        print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
    else:
        self.best_score = score
        self.save_checkpoint(val_loss, model, args)
        self.counter = 0

# def save_checkpoint(self, val_loss, model, args): 
#     if self.verbose:
#         if args.tuning_metric == "loss":
#             print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
#         else:
#             print(
#                 f"{args.tuning_metric} increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
#             )
#     model.save_pretrained(args.model_dir)
#     torch.save(args, os.path.join(args.model_dir, "training_args.bin"))
#     self.val_loss_min = val_loss

#     # # Save model checkpoint (Overwrite)
#     # if not os.path.exists(self.args.model_dir):
#     #     os.makedirs(self.args.model_dir)
#     # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
#     # model_to_save.save_pretrained(self.args.model_dir)

#     # # Save training arguments together with the trained model
#     # torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
#     # logger.info("Saving model checkpoint to %s", self.args.model_dir)




global_step = 0
tr_loss = 0.0
model.zero_grad()

train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
    print("\nEpoch", _)

    for step, batch in enumerate(epoch_iterator):
        self.model.train()
        batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
from tqdm.auto import tqdm, trange
train_iterator = trange(10, desc="Epoch")
for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
    print("\nEpoch", _)
    for step, batch in enumerate(epoch_iterator):
        break
    break
step
batch
for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
    print("\nEpoch", _)

    for step, batch in enumerate(epoch_iterator):
        self.model.train()
        batch = tuple(t.to(device) for t in batch)  # GPU or CPU

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=8)
len(train_dataloader)
# simple triaining loop torxch

loss_lst = []
for epoch in  range(100):

    print("epoch----------------------",epoch)
    running_loss = 0.
    last_loss = 0.
    for i, batch in enumerate(train_dataloader):
        print("step",i)
        
        model.train()
        batch = tuple(t.to(device) for t in batch)  # GPU or CPU

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
        model_type = "abc"
        if model_type != "distilbert":
            inputs["token_type_ids"] = batch[2]
        model.zero_grad()    #or optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs[0]
        print("loss",loss)
        
        loss.backward()   # backpropgation
        optimizer.step()  # gradient decent
        
        
        # Gather data and report
        running_loss += loss.item()
        print(loss.item())
    print("running_loss----------",running_loss)
    loss_lst.append(running_loss)

   
loss_lst
import matplotlib.pyplot as plt
plt.plot(loss_lst)
# validation loop


model.eval()
with torch.no_grad():
    for i, batch in enumerate(train_dataloader):
        print("step",i)
        
        model.train()
        batch = tuple(t.to(device) for t in batch)  # GPU or CPU
    
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
        model_type = "abc"
        if model_type != "distilbert":
            inputs["token_type_ids"] = batch[2]
        #model.zero_grad()    #or optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs[0]
        print("loss",loss)
        

        # Gather data and report
        running_loss += loss.item()
        print(loss.item())
    print("running_loss----------",running_loss)
    loss_lst.append(running_loss)


train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=32)
len(train_dataloader)

# final concluded traing simple step
# trainng loop from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
#training
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

#dataset
train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=32)

import torch.optim as optim
optimizer  = optim.AdamW(model.parameters(),lr=0.00001)

loss_lst = []
val_loss_lst = []
for epoch in range(5):
    print("inside epoch---------------------------------------------------")
    model.train(True)
    running_loss = 0.
    last_loss = 0.

    for step, batch in enumerate(train_dataloader):
    
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)  # GPU or CPU
        
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
        model_type = "abc"
        if model_type != "distilbert":
            inputs["token_type_ids"] = batch[2]
        
        outputs = model(**inputs)
        loss = outputs[0]
        #print("loss",loss)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print("running_loss----------",running_loss)
    loss_lst.append(running_loss)




    running_vloss = 0.0
    model.eval()

    with torch.no_grad():
        
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)  # GPU or CPU
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "intent_label_ids": batch[3],
                "slot_labels_ids": batch[4],
            }
            model_type = "abc"
            if model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]
            
            outputs = model(**inputs)
            loss = outputs[0]
            running_vloss += loss.item()
        
        print("running_vloss----------",running_vloss)
        val_loss_lst.append(running_vloss)
# trainng loop from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
#training
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

#dataset
train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=32)

import torch.optim as optim
optimizer  = optim.AdamW(model.parameters(),lr=0.00001)

loss_lst = []
val_loss_lst = []
for epoch in range(50):
    print("inside epoch---------------------------------------------------")
    model.train(True)
    running_loss = 0.
    last_loss = 0.

    for step, batch in enumerate(train_dataloader):
    
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)  # GPU or CPU
        
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
        model_type = "abc"
        if model_type != "distilbert":
            inputs["token_type_ids"] = batch[2]
        
        outputs = model(**inputs)
        loss = outputs[0]
        #print("loss",loss)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print("running_loss----------",running_loss)
    loss_lst.append(running_loss)




    running_vloss = 0.0
    model.eval()

    with torch.no_grad():
        
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)  # GPU or CPU
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "intent_label_ids": batch[3],
                "slot_labels_ids": batch[4],
            }
            model_type = "abc"
            if model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]
            
            outputs = model(**inputs)
            loss = outputs[0]
            running_vloss += loss.item()
        
        print("running_vloss----------",running_vloss)
        val_loss_lst.append(running_vloss)
import matplotlib.pyplot as plt
plt.plot(loss_lst)
plt.plot(val_loss_lst)
# save model
model_dir = "model"

if not os.path.join(model_dir):
    os.makedirs(model_dir)

model_to_save = model.module if hasattr(model, "module") else model
model_to_save.save_pretrained(model_dir)


# # Save training arguments together with the trained model
# torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
# save model using torch
torch.save(model.state_dict(), 'model_weights.pth')   # correct way to save 

#torch.save(model, "torch_model.pkl")
# load model 
model_from_local  = JointRoberta.from_pretrained(config = config, intent_label_lst=intent_labels,pretrained_model_name_or_path = "model",
                slot_label_lst=slot_labels)
model_from_local  
model_from_local.to(device)
running_vloss = 0.0
model_from_local.eval()

with torch.no_grad():
    
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)  # GPU or CPU
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
        model_type = "abc"
        if model_type != "distilbert":
            inputs["token_type_ids"] = batch[2]
        
        outputs = model_from_local(**inputs)
        loss = outputs[0]
        running_vloss += loss.item()
    
    print("running_vloss----------",running_vloss)
    val_loss_lst.append(running_vloss)
# load model from pytorch 
from_torch_model = JointRoberta(config = config, intent_label_lst=intent_labels,
                slot_label_lst=slot_labels)
from_torch_model .load_state_dict(torch.load('torch_model.pkl').state_dict())

from_torch_model.to(device)zz
running_vloss = 0.0
model_from_local.eval()

with torch.no_grad():
    
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)  # GPU or CPU
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
        model_type = "abc"
        if model_type != "distilbert":
            inputs["token_type_ids"] = batch[2]
        
        outputs = from_torch_model(**inputs)
        loss = outputs[0]
        running_vloss += loss.item()
    
    print("running_vloss----------",running_vloss)
    val_loss_lst.append(running_vloss)
# correct model loading from pytorch
from_torch_model2 = JointRoberta(config = config, intent_label_lst=intent_labels,
                slot_label_lst=slot_labels)
from_torch_model2.load_state_dict(torch.load('model_weights.pth'))
from_torch_model2.to(device)

running_vloss = 0.0
from_torch_model2.eval()

with torch.no_grad():
    
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)  # GPU or CPU
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
        model_type = "abc"
        if model_type != "distilbert":
            inputs["token_type_ids"] = batch[2]
        
        outputs = from_torch_model2(**inputs)
        loss = outputs[0]
        running_vloss += loss.item()
    
    print("running_vloss----------",running_vloss)
    val_loss_lst.append(running_vloss)

# saving ways of model 
# torch.save(model, "torch_model.pkl")
# model.load_state_dict(torch.load('torch_model.pkl').state_dict())
#CTRAN

# # save model

# torch.save(bert_layer,f"models/ctran{_fn}-bertlayer.pkl")
# torch.save(encoder,f"models/ctran{_fn}-encoder.pkl")
# torch.save(middle,f"models/ctran{_fn}-middle.pkl")
# torch.save(decoder,f"models/ctran{_fn}-decoder.pkl")

# # load model 
# bert_layer.load_state_dict(torch.load(f'models/ctran{_fn}-bertlayer.pkl').state_dict())


# # pytorh model save and load
# torch.save(model.state_dict(), PATH)

# #load model
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

# def save_model(self):
#     # Save model checkpoint (Overwrite)
#     if not os.path.exists(self.args.model_dir):
#         os.makedirs(self.args.model_dir)
#     model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
#     model_to_save.save_pretrained(self.args.model_dir)

#     # Save training arguments together with the trained model
#     torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
#     logger.info("Saving model checkpoint to %s", self.args.model_dir)


# def load_model(self):
#         # Check whether model exists
#         if not os.path.exists(self.args.model_dir):
#             raise Exception("Model doesn't exists! Train first!")

#         try:
#             self.model = self.model_class.from_pretrained(
#                 self.args.model_dir,
#                 args=self.args,
#                 intent_label_lst=self.intent_label_lst,
#                 slot_label_lst=self.slot_label_lst,
#             )
#             self.model.to(self.device)
#             logger.info("***** Model Loaded *****")
#         except Exception:
#             raise Exception("Some model files might be missing...")
model = model.from_pretrained
# extra parameters in model
from transformers import AdamW, get_linear_schedule_with_warmup

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)


from transformers import AdamW, get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=t_total
)


global_step = 0
tr_loss = 0.0
model.zero_grad()

train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)



import os

import numpy as np
import torch


class EarlyStopping:
"""Early stops the training if validation loss doesn't improve after a given patience."""

def __init__(self, patience=7, verbose=False):
    """
    Args:
        patience (int): How long to wait after last time validation loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
                        Default: False
    """
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf

def __call__(self, val_loss, model, args):
    if args.tuning_metric == "loss":
        score = -val_loss
    else:
        score = val_loss
    if self.best_score is None:
        self.best_score = score
        self.save_checkpoint(val_loss, model, args)
    elif score < self.best_score:
        self.counter += 1
        print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
    else:
        self.best_score = score
        self.save_checkpoint(val_loss, model, args)
        self.counter = 0

def save_checkpoint(self, val_loss, model, args): 
    if self.verbose:
        if args.tuning_metric == "loss":
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        else:
            print(
                f"{args.tuning_metric} increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
    model.save_pretrained(args.model_dir)
    torch.save(args, os.path.join(args.model_dir, "training_args.bin"))
    self.val_loss_min = val_loss

    # # Save model checkpoint (Overwrite)
    # if not os.path.exists(self.args.model_dir):
    #     os.makedirs(self.args.model_dir)
    # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
    # model_to_save.save_pretrained(self.args.model_dir)

    # # Save training arguments together with the trained model
    # torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
    # logger.info("Saving model checkpoint to %s", self.args.model_dir)


# rough
#joint bert

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

class JointRoberta(RobertaPreTrainedModel):
    def __init__(self, config,args, intent_label_lst, slot_label_lst):
        super(JointRoberta,self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained phobert
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits


import torch
import torch.nn as nn
from torchcrf import CRF
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from .module import IntentClassifier, SlotClassifier


class JointPhoBERT(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointPhoBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained phobert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)

        self.slot_classifier = SlotClassifier(
            config.hidden_size,
            self.num_intent_labels,
            self.num_slot_labels,
            self.args.use_intent_context_concat,
            self.args.use_intent_context_attention,
            self.args.max_seq_len,
            self.args.attention_embedding_size,
            args.dropout_rate,
        )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)------------------------------------------->If you have given a sequence, "You are on StackOverflow". The sequence_output will give 768 embeddings of these four words. But, the pooled output will just give you one embedding of 768, it will pool the embeddings of these four words.
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        if not self.args.use_attention_mask:
            tmp_attention_mask = None
        else:
            tmp_attention_mask = attention_mask

        if self.args.embedding_type == "hard":
            hard_intent_logits = torch.zeros(intent_logits.shape)
            for i, sample in enumerate(intent_logits):
                max_idx = torch.argmax(sample)
                hard_intent_logits[i][max_idx] = 1
            slot_logits = self.slot_classifier(sequence_output, hard_intent_logits, tmp_attention_mask)
        else:
            slot_logits = self.slot_classifier(sequence_output, intent_logits, tmp_attention_mask)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )
            total_loss += self.args.intent_loss_coef * intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction="mean")
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += (1 - self.args.intent_loss_coef) * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
