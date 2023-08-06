import spacy
from scipy.spatial.distance import cosine

import os
import ast
import random
import logging

from tqdm.auto import tqdm
import pandas as pd

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tqdm.pandas()


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"device: {device}")

# Constants
HOME_PATH = os.getcwd()
logger.info(f"HOME_PATH: {HOME_PATH}")

SPLIT_DATA_PATH = os.path.join(HOME_PATH,"data","processed","split_data.csv")
logger.info(f"SPLIT_DATA_PATH: {SPLIT_DATA_PATH}")

# Set the path to save gpt2 model
MODEL_PATH = os.path.join(HOME_PATH, "models")
logger.info(f"model_path: {MODEL_PATH}")

FINAL_RES = os.path.join(HOME_PATH,"data","results_data.csv")
logger.info(f"FINAL_RES: {FINAL_RES}")

# GPT Inference constants
MAX_LENGTH= 100
NUM_RETURN_SEQUENCE= 1
NO_REPEAT_NGRAM_SIZE= 2
REPETITION_PENALTY= 1.5
TOP_P= 0.92
TEMPERATURE=.85
DO_SAMPLE= True
TOP_K= 125
EARLY_STOPPING= True

# Prep data for inference by taking away original sentence all words except 2-3 words randomly
def truncate_text(text: str):
    
    ran_num = random.randint(5,10)
    ran_num = 4
    
    # Split by space
    text_list_split = text.split(" ")
    
    # Select randomly 2-4 words to retain
    text_list_trunc = text_list_split[:ran_num]
    
    # Return
    return " ".join(text_list_trunc)

def get_inference_gpt2(text: str,  model, tokenizer):
    
    # Encode the text using tokenizer
    text_ids = tokenizer.encode(text, return_tensors = 'pt')
    
    generated_text_samples = model.generate(
    text_ids, 
    max_length= MAX_LENGTH,  
    num_return_sequences= NUM_RETURN_SEQUENCE,
    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE ,
    repetition_penalty=REPETITION_PENALTY,
    top_p=TOP_P,
    temperature=TEMPERATURE,
    do_sample= DO_SAMPLE,
    top_k= TOP_K,
    early_stopping= EARLY_STOPPING)

    return tokenizer.decode(generated_text_samples[0], skip_special_tokens=True)

# Helper function
def jaccard_similarity(x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    
    return intersection_cardinality/float(union_cardinality)

def corpus(text):
    text_list = text.split()
    return text_list

def count_words(text_list: str):
    # text_list_format = ast.literal_eval(text_list)
    return len(text_list)

# Printing some examples
def view_generated_samples(index: int, data: pd.DataFrame):  
    index = index
    # original_text = (" ").join(ast.literal_eval(data.iloc[index]["text_lists"]))
    original_text = (" ").join(data.iloc[index]["text_lists"])
    print(f"Original text: {original_text}")
    input_words = data.iloc[index]["trunc_text"]
    print(f"input_words: {input_words}")
    gpt2_text = data.iloc[index]["gpt_text_gen"]
    print(f"gpt2_text generated: {gpt2_text}")
    print(f"\n")

# Create embeddings using simply word2vec
def generate_word2vec_embedding(sentence: str):
    # generate the average of word embeddings
    return nlp(sentence).vector

def calculate_cosine_similarity_score(sentence_one: str, sentence_two: str):
    # encode the sentences into embeddings
    sentence_one_emb = generate_word2vec_embedding(sentence_one)
    sentence_two_emb = generate_word2vec_embedding(sentence_two)
    
    # calculate cosine similarity score
    cos_sim_score = 1 - cosine(sentence_one_emb, sentence_two_emb)
    return cos_sim_score

def predict_gpt(val_data_path: str):
    # Load Validation data
    logging.info(f"Loading validation dataset ...")
    data = pd.read_csv(val_data_path)
    data_val = data[data["split"] == "val"]
    data_val["text"] = data_val["text"].astype(str)

    # Loading trained model and tokenizer
    logging.info(f"Loading trained gpt2 model and tokenizer ...")
    gpt2_model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

    data_val["trunc_text"] = data_val["text"].progress_apply(lambda x: truncate_text(x))

    # Generate inference
    logging.info(f"Performing inference ...")
    # Create a list for trunc text
    trunc_list = data_val["trunc_text"].to_list()

    # Get res
    res = []

    for review in tqdm(trunc_list):
        res.append(get_inference_gpt2(review, gpt2_model, gpt2_tokenizer))

    # Add back to original dataframe
    data_val["gpt_text_gen"] = res

    # Split the original text into list of words then count
    data_val["text_lists"] = data_val["text"].progress_apply(corpus)
    data_val["word_count"] = data_val["text_lists"].progress_apply(count_words)

    # Calculate jaccard similarity
    logging.info(f"Calculating jaccard similarity score ...")
    data_val["jaccard_score"] = data_val.progress_apply(lambda x: jaccard_similarity(x["text"],x["gpt_text_gen"]),axis=1)

    # Load word2vec pretrained model
    nlp = spacy.load("en_core_web_sm")

    # Calculate cosine similarity score
    logging.info(f"Calculating cosine similarity score ...")
    data_val["cos_sim_score"] = data_val.progress_apply(lambda x: calculate_cosine_similarity_score(x["text"], x["gpt_text_gen"]), axis=1)

    # Save final results
    logging.info(f"Saving results ...")
    data_val.to_csv(FINAL_RES, index=False)
    logging.info(f"Inference using gpt2 on validation dataset complete. Final results saved to:{FINAL_RES} ...")


if __name__ == '__main__':
        
    # Train gpt2 model
    predict_gpt(SPLIT_DATA_PATH)
    