### implemented 1. amp 2. batching 3. one cycle

1. for this error '/content/gdrive/MyDrive/model.py in attention(query, key, value, mask, dropout)
    115         if mask is not None:
    116             # Write a very low value (indicating -inf) to the positions where mask == 0
--> 117             attention_scores.masked_fill_(mask == 0, -1e9)
    118         attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
    119         if dropout is not None:

RuntimeError: value cannot be converted to type at::Half without overflow'- changed attention mask to -1e4

2. got 8 mins/epoch - batch size is too less - 6. increasing it, will decrease runtime further