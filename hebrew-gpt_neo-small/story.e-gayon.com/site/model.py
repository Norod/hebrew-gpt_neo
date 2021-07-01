# -*- coding: utf-8 -*-

import argparse
import re

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Norod78/hebrew_stories-gpt_neo-small")
model = AutoModelForCausalLM.from_pretrained("Norod78/hebrew_stories-gpt_neo-small")

stop_token = "<|endoftext|>"
new_lines = "\n\n\n"

np.random.seed(None)
random_seed = np.random.randint(10000,size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = 0 if torch.cuda.is_available()==False else torch.cuda.device_count()

torch.manual_seed(random_seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(random_seed)

model.to(device)

def extend(input_text, max_size=20):
    if len(input_text) == 0:
        input_text = "האיש האחרון בעולם ישב לבד בחדרו, לפתע נשמע נקישה"

    encoded_prompt = tokenizer.encode(
    input_text, add_special_tokens=False, return_tensors="pt")

    encoded_prompt = encoded_prompt.to(device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt
    
    output_sequences = model.generate(
    input_ids=input_ids,
    max_length=max_size + len(encoded_prompt[0]),
    top_k=50, 
    top_p=0.95, 
    do_sample=True,
    num_return_sequences=1)

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

        # Remove all text after 3 newlines
        text = text[: text.find(new_lines) if new_lines else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            input_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
    
    parsed_text = total_sequence.replace("<|startoftext|>", "").replace("\t","").replace("\r","").replace("\n\n", "\n")
    if len(parsed_text) == 0:
        parsed_text = "שגיאה"
    return parsed_text

if __name__ == "__main__":
    test_text = ''
    extended = extend(test_text, 96)
    print(extended)
