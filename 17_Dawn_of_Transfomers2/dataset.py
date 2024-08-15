import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BillingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype = torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    """
    Returns a pair of english-italian sentence. The reason why we are sending both is explained in notes
    """
    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        '''
        src_tgt_pair : {'id': '7708', 'translation': {'en': '"Yes; but the time is of no consequence: what followed is the strange point.', 'it': '— Sì, ma poco importa il giorno.'}}
        '''
        src_text = src_tgt_pair['translation'][self.src_lang]
        '''
        src_text : "Yes; but the time is of no consequence: what followed is the strange point.
        '''
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]
        '''
        tgt_text : — Sì, ma poco importa il giorno.
        '''
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        '''
        enc_input_tokens : [35, 185, 18, 31, 5, 94, 34, 10, 67, 1778, 38, 55, 480, 34, 5, 389, 490, 7]
        '''
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        '''
        dec_input_tokens : [9, 176, 4, 26, 161, 1477, 14, 147, 5]
        '''
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        #For encoding, we PAD both SOS and EOS. For decoding, we only pad SOS.
        #THe model is required to predict EOS and stop on its own.
        
        #Make sure that padding is not negative (ie the sentance is too long)
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")
            
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype = torch.int64)
            ],
            dim =  0,
        )
        '''
        encoder_input : tensor([   1,   35,  185,   18,   31,    5,   94,   34,   10,   67, 1778,   38,
          55,  480,   34,    5,  389,  490,    7,    2,    3,    3,    3,    3,
           3,    3,    3,    3,    3,    3,    3,    3,....
           3,    3]). It is 512d where 1 - start, 2-end, and 3 is pad
        '''
        
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0,
        )
        '''
        similar as above for decoder_input
        '''
        
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64),
            ],
            dim = 0,
        )        
        '''
        similar as above for label
        '''
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), 
            # encoder mask: (1, 1, seq_len) -> Has 1 when there is text and 0 when there is pad (no text)
            '''
            Why masking? - 1. In encoder, we mask the pad tokens for 2 reasons 1. we dont want them to be taken into account during
            attention values calculation, so making them 0 will ensure their weigtage is 0 2. in loss calculation, id of pad token is ignored
            2. In decoder, same thing plus 1. we dont want to take future values into consideration and 2. we want to perform loss calculation for decoder predictions parallely
            '''
            '''
            encoder_mask : tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0]]], dtype=torch.int32)
            '''

            '''
            decoder_mask : tensor([[[1, 0, 0,  ..., 0, 0, 0],
            [1, 1, 0,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            ...,
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0]]], dtype=torch.int32)
            '''
            
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            # (1, seq_len) and (1, seq_len, seq_len)
            # Will get 0 for all pads. And 0 for earlier text.
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
            }
    
def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal = 1).type(torch.int)
    #This will get the upper traingle values
    return mask == 0
    
    
    