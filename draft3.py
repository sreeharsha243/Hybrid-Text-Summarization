from transformers import BartForConditionalGeneration, BartTokenizer
import networkx as nx # for graph representation, here it is used for rank awarding for textrank. 
import numpy as np    # for array/matrix creation
from nltk.corpus import stopwords # loading stopwords from nltk.corpus toolkit 
from nltk.stem import WordNetLemmatizer # for stemming WordNetLemmatizer is used
from nltk.tokenize import sent_tokenize, word_tokenize # for tokenizing sentences and words using nltk.tokenize 
from sklearn.metrics.pairwise import cosine_similarity # calculating cosiner similarity
from sklearn.cluster import KMeans #for kmean clustering purpose. 
from scipy.spatial import distance # for finding the euclidean distance for clustering in kmeans
from evaluate import load # evaluation using metrics
# Load the ROUGE metric
import evaluate 
import pandas as pd # for dataframe creation
import re # regular expression usage
import os # file/ folder handling
import matplotlib.pyplot as plt # plotting graphs
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns # for heatmap creation
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

wl = WordNetLemmatizer()

def funforfrontend(input_text):

    def generate_summary(input_text, max_length=1024):
        # Load the model and tokenizer
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

        # Tokenize the input text into smaller chunks
        input_chunks = [input_text[i:i+max_length] for i in range(0, len(input_text), max_length)]

        # Generate summaries for each chunk and concatenate them
        summaries = []
        for chunk in input_chunks:
            inputs = tokenizer(chunk, return_tensors="pt", max_length=max_length, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], num_beams=3, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        return ' '.join(summaries)

    # Read input text from file
    # input_file_path = gsummary_filename
    # with open(input_file_path, "r", encoding="utf-8") as file:
    # input_text = ''
    # doc = input_text
    # Generate summary
    summary = generate_summary(input_text)

    # Write summary to output file
    # output_file_path = "output.txt"
    # with open(output_file_path, "w", encoding="utf-8") as file:
    #     file.write(summary)

    # print("Summary generated and saved to output.txt")

    # # Accept input text from the user
    # doc=text
    
    # # Convert the input text to uppercase
    # uppercase_text = text.upper()
    
    # # Print the uppercase text
    # # print("Uppercase text:", uppercase_text)

    return summary, doc