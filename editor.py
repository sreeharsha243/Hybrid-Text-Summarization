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

def funforfrontend(input_file):

    doc = input_file
    def extract_word_vectors() -> dict:
        """
        Extracting word embeddings. These are the n vector representation of words.
        """
        #print('Extracting word vectors')

        word_embeddings = {}
        # Here we use glove word embeddings of 100 dimension
        f = open(r"C:\Users\sriha\Documents\builds\project\MAIN\MAIN\glove.6B.100d.txt", encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs

        f.close()
        return word_embeddings


    def text_preprocessing(sentences: list) -> list:
        """
        Pre processing text to remove unnecessary words.
        """
        #print('Preprocessing text')

        stop_words = set(stopwords.words('english'))
        for i in range(len(sentences)):
            sen = re.sub('[^a-zA-Z0-9£₹]', " ", sentences[i])  
            #sen = sen.lower()  
            sen=sentences[i]
            sen=sen.split()                         
            sen = ' '.join([i for i in sen if i not in stopwords.words('english')])
            sentences.append(sen)


        clean_words = None
        for sent in sentences:
            words = word_tokenize(sent)
            #words = [wl.lemmatize(word.lower()) for word in words if word.isalnum()]
            clean_words = [word for word in words if word not in stop_words]

        return clean_words


    def sentence_vector_representation(sentences: list, word_embeddings: dict) -> list:
        """
        Creating sentence vectors from word embeddings.
        """
        #print('Sentence embedding vector representations')

        sentence_vectors = []
        for sent in sentences:
            clean_words = text_preprocessing([sent])
            # Averaging the sum of word embeddings of the sentence to get sentence embedding vector
            v = sum([word_embeddings.get(word, np.zeros(100, )) for word in clean_words]) / (len(clean_words) + 0.001)
            sentence_vectors.append(v)

        return sentence_vectors


    def create_similarity_matrix(sentences: list, sentence_vectors: list) -> np.ndarray:
        """
        Using cosine similarity, generate similarity matrix.
        """
        #print('Creating similarity matrix')
        # Vectorize sentences
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)
        
        # Calculate cosine similarity between sentence vectors
        sim_mat = cosine_similarity(X, X)
        # Defining a zero matrix of dimension n * n
        #sim_mat = np.zeros([len(sentences), len(sentences)])
        #for i in range(len(sentences)):
        #    for j in range(len(sentences)):
        #        if i != j:
        #            # Replacing array value with similarity value.
        #            # Not replacing the diagonal values because it represents similarity with its own sentence.
        #            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

        return sim_mat


    def determine_sentence_rank(sentences: list, sim_mat: np.ndarray):
        """
        Determining sentence rank using Page Rank algorithm.
        """
        #print('Determining sentence ranks')
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted([(scores[i], s[:15]) for i, s in enumerate(sentences)], reverse=True)
        return ranked_sentences


    def generate_summary_textrank(sentences: list, ranked_sentences: list):
        """
        Generate a sentence for sentence score greater than average.
        """
        #print('Generating summary')

        # Get top 1/3 th ranked sentences
        top_ranked_sentences = ranked_sentences[:int(len(sentences) / 3)] if len(sentences) >= 3 else ranked_sentences

        sentence_count = 0
        summary = ''

        for i in sentences:
            for j in top_ranked_sentences:
                if i[:15] == j[1]:
                    summary += i + ' '
                    sentence_count += 1
                    break
        #summary = ''.join(summary.split())
        return summary

    def generate_summary_kmeans(sentences_textrank: list, sen_vectors_textrank: list)->str:
        # Calculate WCSS for different values of n_clusters
        #wcss = []
        #for i in range(1, 11):
        #    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        #    kmeans.fit(sen_vectors_textrank)
        #    wcss.append(kmeans.inertia_)

        # Calculate the differences between consecutive WCSS values
        #differences = np.diff(wcss)

        # Calculate the percentage change in WCSS
        #percent_change = (differences / wcss[:-1]) * 100

        # Find the index corresponding to the maximum percentage change
        #optimal_index = np.argmax(percent_change)

        # Plot the elbow curve
        #plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
        #plt.title('Elbow Method')
        #plt.xlabel('Number of clusters (n_clusters)')
        #plt.ylabel('Within-cluster sum of squares (WCSS)')
        #plt.show()

        # The optimal value for n_clusters is one more than the index with maximum percentage change
        optimal_n_clusters = 8 #optimal_index + 1
        #print("Optimal number of clusters (n_clusters):", optimal_n_clusters)
        n_clusters = optimal_n_clusters # int(input("Number of clusters: "))
        kmeans = KMeans(n_clusters, init = 'k-means++', random_state = 42)
        y_kmeans = kmeans.fit_predict(sen_vectors_textrank)

        #finding and printing the nearest sentence vector from cluster centroid

        my_list=[]
        for i in range(n_clusters):
            my_dict={}
        
            for j in range(len(y_kmeans)):
                if y_kmeans[j]==i:
                    my_dict[j] =  distance.euclidean(kmeans.cluster_centers_[i],sen_vectors_textrank[j])
            #min_distance = min(my_dict.values())
            #my_list.append(min(my_dict, key=my_dict.get))
            min_distances = sorted(my_dict.values())[:4]  # Select only 4 sentences from each cluster
            selected_indices = [index for index, distance_value in my_dict.items() if distance_value in min_distances]
            my_list.extend(selected_indices)
        print(f'No. of sentences in the kmeans:{len(my_list)}\n')
        summary_kmeans = ''.join(sentences_textrank[i] for i in sorted(my_list))
        #summary_kmeans = ''.join(summary_kmeans.split())
        return summary_kmeans

    # gsummary_filename = None  # Global variable to store summary filenames

    def main():
        global gsummary_filename  # Declare the use of the global variable
        
        # text = 'touching on themes of community diversity and the juxtaposition of activity and tranquilityheres another expansion of the text as the sun rises over the city skyline casting its golden rays upon the waking metropolis a sense of anticipation fills the airin quiet cafes and hidden gardens individuals seek refuge from the noise and clamor of the outside worldhere amidst the serenity of nature and the gentle hum of conversation they find solace and peaceas the day gives way to night the city takes on a new persona bathed in the soft glow of streetlights and neon signsand amidst the chaos and the cacophony there is beauty  the beauty of human connection of shared experiences of life lived to the fullestthis expansion further explores the daily rhythms and nuances of city life touching on themes of community diversity and the juxtaposition of activity and tranquilityheres another expansion of the text as the sun rises over the city skyline casting its golden rays upon the waking metropolis a sense of anticipation fills the airin quiet cafes and hidden gardens individuals seek refuge from the noise and clamor of the outside worldhere amidst the serenity of nature and the gentle hum of conversation they find solace and peaceas the day gives way to night the city takes on a new persona bathed in the soft glow of streetlights and neon signsand as the clock ticks on the city never sleeps its heartbeat echoing through the streets a constant reminder of the vitality of urban lifeit is a place of endless exploration and discovery where every corner holds the promise of adventureand amidst the chaos and the cacophony there is beauty  the beauty of human connection of shared experiences of life lived to the fullestthis expansion further explores the daily rhythms and nuances of city life touching on themes of community diversity and the juxtaposition of activity and tranquilityheres another expansion of the text'

        #print('Original Text::-----------------\n')

        sentences = sent_tokenize(input_file.strip())
        #print('Sentences:---------------------\n',sentences)

        word_embeddings = extract_word_vectors()
        #print('Word embeddings:---------------------\n',len(word_embeddings))

        sentence_vectors = sentence_vector_representation(sentences, word_embeddings)
        #print('Sentence vectors:--------------------\n', len(sentence_vectors), sentence_vectors)

        sim_mat = create_similarity_matrix(sentences, sentence_vectors)
        #print('Similarity matrix:-------------------\n', sim_mat.shape, sim_mat)

        ranked_sentences = determine_sentence_rank(sentences, sim_mat)
        #print('Length of Ranked Sentences-------------\n', len(ranked_sentences))
        #print('Ranked sentences:--------------------\n', ranked_sentences)

        summary = generate_summary_textrank(sentences, ranked_sentences)
        #print('Length of summary after TestRank-------------\n', len(summary))
        #print('Summary generated by TextRank--------------------\n',summary)

        sentences_textrank = sent_tokenize(summary.strip())
        #print("Length of the text rank: \n", len(sentences_textrank))
        #print('Sentences:--------------------------------------\n',sentences_textrank)

        word_embeddings_textrank = extract_word_vectors()
        #print('Word embeddings:-----------------------------------\n',len(word_embeddings_textrank))

        sentence_vectors_textrank = sentence_vector_representation(sentences_textrank, word_embeddings)
        #print('Sentence vectors of summary generated by textrank ---------------------------\n', len(sentence_vectors_textrank), sentence_vectors_textrank)

        similarity_matrix = cosine_similarity(sentence_vectors_textrank, sentence_vectors)
        max_similarity_indices = similarity_matrix.argmax(axis=1)
        #print(f'Length of Indices is: {len(max_similarity_indices)}')
        #print(f'Indices of comparision:{max_similarity_indices}')
        sentence_vectors_textrank=sorted(sentence_vectors_textrank, key=lambda x: x[0])
        summary_final = generate_summary_kmeans(sentences_textrank, sentence_vectors_textrank)
        #print('Summary generated by combining TextRank and K-means--------------------\n',summary_final)
        # summary_filename = os.path.join(directory_path_write, filename.replace(".txt", "_1.txt"))
        # gsummary_filename = summary_filename  # Add to the global list

        # with open(summary_filename, 'w', encoding='utf-8') as summary_file:
        #     summary_file.write(summary_final)
        # print(f"Summary for '{filename}' written to '{summary_filename}'.")
        return summary_final

    # if __name__ == "__main__":
    #     main()

    def text_to_paragraphs(input_text):
        # Read the input text from the file
        # with open(input_file_path, 'r', encoding='utf-8') as file:
        #     text = file.read()

        # Split the text into lines
        lines = input_text.split("\n")
        
        # Initialize an empty list to store paragraphs
        paragraphs = []
        
        # Initialize an empty string to accumulate lines for a paragraph
        current_paragraph = ""
        
        # Iterate through each line in the text
        for line in lines:
            # If the line is empty (i.e., contains only whitespace), it indicates the end of a paragraph
            if not line.strip():
                # Add the accumulated lines to the paragraphs list
                if current_paragraph:
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = ""
            else:
                # Add the line to the current paragraph
                current_paragraph += line.strip() + " "
        
        # Add the last accumulated paragraph if any
        if current_paragraph:
            paragraphs.append(current_paragraph.strip())
        
        # Write the paragraphs back to the input file
        # with open(input_file_path, 'w', encoding='utf-8') as file:
        #     for paragraph in paragraphs:
        #         file.write(paragraph + "\n\n")
        return paragraphs
    # Specify the input file path
    input_text_for_this_fun = main()

    # Convert text to paragraphs and overwrite the input file with paragraphs
    paragraph = str(text_to_paragraphs(input_text_for_this_fun))
    # print(paragraph)

    # print(f"The paragraphs have been saved to {output_file_path}.")

    def remove_non_alphabetic(input_text):
        # Read the input text from the file
        # with open(input_file_path, 'r', encoding='utf-8') as file:
        #     text = file.read()

        # Remove all characters except alphabetic characters and whitespace
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', input_text)

        # Write the cleaned text back to the same file
        # with open(input_file_path, 'w', encoding='utf-8') as file:
        #     file.write(cleaned_text)
        return cleaned_text

    # Specify the input file path
    # input_file_path_for_cleaning = text_to_paragraphs

    # Call the function to remove non-alphabetic characters
    clean_text = remove_non_alphabetic(paragraph)

    # print(clean_text)

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
    summary = generate_summary(clean_text)

    return summary, doc