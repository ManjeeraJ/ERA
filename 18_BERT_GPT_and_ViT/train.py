from model import build_transformer
from dataset import BillingualDataset, casual_mask
from config_file import get_config, get_weights_file_path

import torchtext.datasets as datasets
import torch
torch.cuda.amp.autocast(enabled = True)

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:12240"  # Ignore. This was specific to RS's system
config = get_config()

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    
    
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    encoder_output = model.encode(source, source_mask)
    #Initialize the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:  # config['seq_len']
            break
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        '''
        Since model.decode has to take decoder_mask as input during training, now also we have to give as input. Otherwise it is not required
        during inference, right??
        first loop : decoder_mask = torch.tensor([
    [1]
]).type_as(source_mask).to(device)
        second loop : decoder_mask = torch.tensor([
    [1, 0],
    [1, 1]
]).type_as(source_mask).to(device)

        '''
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source_mask).fill_(next_word.item()).to(device)
            ],
            dim =  1
        )
        
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)

'''
We have a different code for validation as in training we use parallel prediction of decoders (by masking future) which we cannot do here. Here, we loop
'''
def run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0  # count of batches, also here batch size = 1
    source_texts = []
    expected = []
    predicted = []
    
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80
        
    with torch.no_grad():
        for batch in val_dataloader:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            assert encoder_input.size(0)==1, "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)  # The output basically contains ids, because the next part it getting the text out using the ids
            
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.text.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.text.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.text.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentenses(ds, lang):
    for item in ds:
        yield item['translation'][lang]
        
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))  # If the tokenizer file doesn't exist, the function creates a new Tokenizer object with a WordLevel model.
        tokenizer.pre_tokenizer = Whitespace()   # The tokenizer's pre_tokenizer is set to split input text by whitespace.
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentenses(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    

    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split = 'train')
    '''
    print (ds_raw) --->
    Dataset({
    features: ['id', 'translation'],
    num_rows: 32332
})
    '''
    
    src_lang = config["lang_src"]
    tgt_lang = config["lang_tgt"]
    seq_len = config["seq_len"]
    
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, src_lang)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, tgt_lang)
    
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len)
    val_ds = BillingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len)
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][src_lang]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][tgt_lang]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max length of the source sentence : {max_len_src}")
    print(f"Max length of the source target : {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size = config["batch_size"], shuffle = True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)  # important to note that batch size is 1 here
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

'''
Implement smart batching. Below reduces the length to the maximum of that particular batch. Can be improved further
'''
def collate_fn(batch):

    encoder_input_max = max(x['encoder_str_length'] for x in batch)
    decoder_input_max = max(x['decoder_str_length'] for x in batch)
    
    encoder_inputs = []
    decoder_inputs = []
    encoder_mask = []
    decoder_mask = []
    label = []
    src_text = []
    tgt_text = []

    for b in batch:
        encoder_inputs.append(b["encoder_input"][:encoder_input_max])
        decoder_inputs.append(b["decoder_input"][:decoder_input_max])
        encoder_mask.append((b["encoder_mask"][0,0,:encoder_input_max]).unsqueeze(0).unsqueeze(0).unsqueeze(0).int())
        decoder_mask.append((b["decoder_mask"][0,:decoder_input_max,:decoder_input_max]).unsqueeze(0).unsqueeze(0))
        label.append(b["label"][:decoder_input_max])
        src_text.append(b["src_text"])
        tgt_text.append(b["tgt_text"])
        


    return {
        "encoder_input": torch.vstack(encoder_inputs), 
        "decoder_input": torch.vstack(decoder_inputs),  
        "encoder_mask": torch.vstack(encoder_mask),
        "decoder_mask": torch.vstack(decoder_mask),
        "label": torch.vstack(label), 
        "src_text": src_text,
        "tgt_text": tgt_text,
    }


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, config["seq_len"], config["seq_len"], d_model=config['d_model'])
    return model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")
    
    '''
    In summary, thr below line ensures that the directory specified in config["model_folder"] exists, creating any necessary parent directories(if path given is models/checkpoints) along the way and not raising an error if the directory already exists.
    '''
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    #Tensorboard
    writer = SummaryWriter(config["experiment_name"])
    
    #Adam is used to train each feature with a different learning rate. 
    #If some feature is appearing less, adam takes care of it
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], eps = 1e-9)

    # --------------------------
    MAX_LR = 10**-3  # Use 10 times higher value than the lr used in the paper(LLM)
    STEPS_PER_EPOCH = len(train_dataloader)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr = MAX_LR,
                                                    steps_per_epoch = STEPS_PER_EPOCH,
                                                    epochs=config['num_epochs'],
                                                    pct_start = int (0.3*config['num_epochs'])/config['num_epochs'] if config['num_epochs'] != 1 else 0.5,  # 30% of total epochs
                                                    div_factor =100,
                                                    three_phase = False,
                                                    final_div_factor = 100,  # If not assignment, =10, anneal to 10 to the power of -5, and further maybe 20 more epochs
                                                    anneal_strategy="linear"
                                                    )

    #-----------------------------
    
    initial_epoch = 0
    global_step = 0  # I think this is counter for batches which is not refreshed for every epoch
    
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print("Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        print("preloaded")
        
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)
    '''
    ignore_index=tokenizer_src.token_to_id("[PAD]") ---> ignore_index: This parameter specifies a target value that should be ignored when computing the loss. It's often used to ignore padding tokens in sequence data.
    tokenizer_src.token_to_id("[PAD]"): This converts the padding token [PAD] to its corresponding integer ID using the source tokenizer tokenizer_src. The ID returned by this method is used as the ignore_index.
    It is important to note that the loss is calculated between predicted and true labels. To have a look at how labels look, refer to dataset.py
    '''
    scaler = torch.cuda.amp.GradScaler()  # What??
    lr = [0.0]  # What??
    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        print(epoch)
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            
            with torch.autocast(device_type = 'cuda', dtype = torch.float16):  # What??
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # It is important to note here that decoder predictions are hapenning in parallel rather in loop because of masking. Need to verify it in code. 
                proj_output = model.project(decoder_output)
                
                label = batch["label"].to(device)
                
                #Compute loss using cross entropy
                tgt_vocab_size = tokenizer_tgt.get_vocab_size()
                loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            '''
            Components
It is critical to note that cross entropy loss doe logsoftmax,NLL inside its function itself. Therefore the size on the input going in for predicted lables can be [batch_size * seq_length, tgt_vocab_size]

1. tgt_vocab_size = tokenizer_tgt.get_vocab_size()

This line retrieves the size of the target vocabulary using the target tokenizer. The vocabulary size is needed to correctly reshape the model's output for the loss calculation.
loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))

2. proj_output: This is the output from the model, typically a tensor of shape [batch_size, seq_length, tgt_vocab_size], where tgt_vocab_size is the number of classes (vocabulary size).
proj_output.view(-1, tgt_vocab_size): This reshapes the proj_output tensor to [batch_size * seq_length, tgt_vocab_size], flattening the batch and sequence dimensions to a single dimension. This is necessary because nn.CrossEntropyLoss expects the input tensor to be of shape [N, C], where N is the number of instances and C is the number of classes.
3. label: This is the ground truth tensor, typically of shape [batch_size, seq_length].
label.view(-1): This flattens the label tensor to [batch_size * seq_length], making it a 1D tensor of target class indices.
loss_fn: This is the loss function (defined earlier as nn.CrossEntropyLoss). It computes the cross-entropy loss between the flattened model outputs and the flattened ground truth labels.
batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

4. batch_iterator: This is likely an instance of tqdm, a popular library for creating progress bars in Python.
.set_postfix({"loss": f"{loss.item():6.3f}"}): This updates the progress bar with the current loss value. loss.item() extracts the scalar value from the loss tensor, and f"{loss.item():6.3f}" formats the loss value to a string with three decimal places, ensuring it fits neatly in the progress bar display.
            '''

            #Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()
            
            #Backpropogate loss
            # loss.backward()
            # WHAT ID GOING ON BELOW??
            scaler.scale(loss).backward()
            
            #Update weights
            # optimizer.step()
            # optimizer.zero_grad(set_to_none=True)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            if not skip_lr_sched:
                scheduler.step
            lr.append(scheduler.get_last_lr())
            '''
            set_to_none=True:

            This parameter, when set to True, sets the gradients to None instead of zero. This can have some performance benefits:
            Memory Efficiency: Setting gradients to None can save memory, as it avoids the need to store zeros.
            Speed: It can also potentially speed up the training process because setting a variable to None can be faster than setting it to zero.
                        
            '''
            # scheduler.step()

            global_step+=1
            
        #run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, writer, global_step)
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        # torch.save(
        #     {
        #         "epoch": epoch,
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "global_step": global_step
        #     },
        #     model_filename
        # )
        
            
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    
    