#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Norod78/hebrew_stories-gpt_neo-small")
model = AutoModelForCausalLM.from_pretrained("Norod78/hebrew_stories-gpt_neo-small")

prompt_text = "פעם אחת, לפני שנים רבות "
stop_token = "<|endoftext|>"
generated_max_length = 50
seed = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = 0 if torch.cuda.is_available()==False else torch.cuda.device_count()


np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

model.to(device)
#model.half()


encoded_prompt = tokenizer.encode(
    prompt_text, add_special_tokens=False, return_tensors="pt")

encoded_prompt = encoded_prompt.to(device)

if encoded_prompt.size()[-1] == 0:
        input_ids = None
else:
        input_ids = encoded_prompt

output_sequences = model.generate(
    input_ids=input_ids,
    max_length=generated_max_length + len(encoded_prompt[0]),
    top_k=50, 
    top_p=0.95, 
    do_sample=True,
    num_return_sequences=2
)

# Remove the batch dimension when returning multiple sequences
if len(output_sequences.shape) > 2:
    output_sequences.squeeze_()

generated_sequences = []

for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    
    generated_sequence = generated_sequence.tolist()

    # Decode text
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    # Remove all text after the stop token
    text = text[: text.find(stop_token) if stop_token else None]

    # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
    total_sequence = (
        prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
    )

    generated_sequences.append(total_sequence)
print(total_sequence)
print("------")

