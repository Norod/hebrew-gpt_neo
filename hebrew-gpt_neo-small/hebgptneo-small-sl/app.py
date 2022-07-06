# -*- coding: utf-8 -*-

import argparse
import re
import os

import streamlit as st
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tokenizers

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

random.seed(None)
suggested_text_list = ['פעם אחת, לפני שנים רבות','שלום, קוראים לי דורון ואני','בוקר טוב לכולם','ואז הפרתי את כל כללי הטקס כש']

@st.cache(hash_funcs={tokenizers.Tokenizer: id, tokenizers.AddedToken: id})
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def extend(input_text, max_size=20, top_k=50, top_p=0.95):
    if len(input_text) == 0:
        input_text = ""

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
    top_k=top_k, 
    top_p=top_p, 
    do_sample=True,
    repetition_penalty=25.0,
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
    
    parsed_text = total_sequence.replace("<|startoftext|>", "").replace("\r","").replace("\n\n", "\n")
    if len(parsed_text) == 0:
        parsed_text = "שגיאה"
    return parsed_text

if __name__ == "__main__":
    st.title("Hebrew GPT Neo (Small)")

    pre_model_path = './hebrew-gpt_neo-small'

    model, tokenizer = load_model(pre_model_path)

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

    text_area = st.text_area("Enter the first few words (or leave blank), tap on \"Generate Text\" below. Tapping again will produce a different result.", 'האיש האחרון בעולם ישב לבד בחדרו כשלפתע נשמעה נקישה')

    st.sidebar.subheader("Configurable parameters")

    max_len = st.sidebar.slider("Max-Length", 0, 256, 192,help="The maximum length of the sequence to be generated.")
    top_k = st.sidebar.slider("Top-K", 0, 100, 40, help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    top_p = st.sidebar.slider("Top-P", 0.0, 1.0, 0.92, help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.")

    if st.button("Generate Text"):
        with st.spinner(text="Generating results..."):
            st.subheader("Result")
            print(f"device:{device}, n_gpu:{n_gpu}, random_seed:{random_seed}, maxlen:{max_len}, top_k:{top_k}, top_p:{top_p}")
            if len(text_area.strip()) == 0:
                text_area = random.choice(suggested_text_list)
            result = extend(input_text=text_area,                         
                            max_size=int(max_len),                         
                            top_k=int(top_k),
                            top_p=float(top_p))

            print("Done length: " + str(len(result)) + " bytes") 
            #<div class="rtl" dir="rtl" style="text-align:right;">
            st.markdown(f"<p dir=\"rtl\" style=\"text-align:right;\"> {result} </p>", unsafe_allow_html=True)
            st.write("\n\nResult length: " + str(len(result)) + " bytes")
            print(f"\"{result}\"")      
    
    st.markdown(    
        """Hebrew text generation model (125M parameters) based on EleutherAI's gpt-neo architecture. Originally trained on a TPUv3-8 which was made avilable to me via the [TPU Research Cloud Program](https://sites.research.google/trc/)."""
    )

    st.markdown("<footer><hr><p style=\"font-size:14px\">Enjoy</p><p style=\"font-size:12px\">Created by <a href=\"https://linktr.ee/Norod78\">Doron Adler</a></p></footer> ", unsafe_allow_html=True)