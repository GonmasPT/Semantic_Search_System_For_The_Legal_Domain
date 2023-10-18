from collections import defaultdict
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import string
import os.path
import pickle
from sentence_transformers import InputExample
import random
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
import torch

def retrieve_segmented_corpus():
    with open('data/dataset.txt', 'r') as file:
        lines = file.readlines()
        articles = defaultdict(list)
        article_id = ''

        # Iterate over each line in the file
        for line in lines:
            # Check if the line starts with "Article"
            if line.startswith("Article") and not line.startswith("Articles"):
                # Get article id
                match = re.match(r"Article\s+(\d+(?:-\d+)?)", line)
                article_id = match.group(1)
                # Remove id from segment
                line = re.sub(match.group(0), "", line)
                # Add segment to the corresponding id
                articles[article_id].append(line.strip())
            elif re.match("\((\d+|[ivxlcdm]+)\)", line):
                articles[article_id].append(line.strip())
        return articles


def retrieve_training_data():
    pickle_file = "segment_training_data.pkl"
    train_id__query = defaultdict(list)
    train_id__rel_docs = defaultdict(list)

    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as f:
            train_id__query, train_id__rel_docs = pickle.load(f)
    else:
        file_names = ["data/riteval_H18_en.xml", "data/riteval_H19_en.xml", "data/riteval_H20_en.xml", "data/riteval_H21_en.xml", "data/riteval_H22_en.xml", "data/riteval_H23_en.xml", "data/riteval_H24_en.xml", "data/riteval_H25_en.xml", "data/riteval_H26_en.xml", "data/riteval_H27_en.xml", "data/riteval_H28_en.xml", "data/riteval_H29_en.xml", "data/riteval_H30_en.xml", "data/riteval_R01_en.xml"]
        #file_names = ["data/riteval_H18_en.xml"]
        for file in file_names:
            # Reading the data inside the xml file
            with open(file, 'r') as f:
                training_data = f.read()
                # Passing the stored data inside the beautifulsoup parser, storing the returned object
                Bs_data = BeautifulSoup(training_data, "xml")
                # Finding all instances of tag 't1'
                t1_tag = Bs_data.find_all('t1')
                # Finding all instances of tag `t2`
                t2_tag = Bs_data.find_all('t2')
                # Find pairs (ids)
                pairs = Bs_data.find_all('pair')

                for pair, t1, t2 in zip(pairs, t1_tag, t2_tag):
                    train_id__query[pair.get('id')].append(t2.text.strip())
                    t1 = t1.text.strip()
                    t1_split = t1.splitlines()

                    for line in t1_split:
                        if len(line) == 0: continue
                        elif line.startswith("Article"):
                            # Get id
                            match = re.match(r"Article\s+(\d+(?:-\d+)?)", line)
                            train_id__rel_docs[pair.get('id')].append(match.group(1))
                        else: continue

        with open(pickle_file, "wb") as f:
            pickle.dump((train_id__query, train_id__rel_docs), f)
    return train_id__query, train_id__rel_docs


def print_corpus(corpus):
    # Open the output file for writing
    with open('OriginalArticles.txt', 'w') as output_file:
        # Write the articles to the output file
        for article in corpus:
            output_file.write(article)
            output_file.write("\n")


def lexical_process_corpus(articles):
    #nltk.download('stopwords')
    #nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    processed_articles = []
    for i, article in enumerate(articles):
        # Remove special characters
        processed_articles.append(unidecode(article))
        # Remove punctuation
        processed_articles[i] = re.sub(r'[^\w\s-]|_', '', processed_articles[i])
        # Lowercase all words
        processed_articles[i] = processed_articles[i].lower()
        # Remove stop words
        processed_articles[i] = ' '.join([word for word in processed_articles[i].split() if word not in stop_words])
        # Perform word tokenization
        processed_articles[i] = word_tokenize(processed_articles[i])
    return processed_articles


def lexical_process_queries(queries):
    #nltk.download('stopwords')
    #nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    processed_queries = []
    for i, query in enumerate(queries):
        # Remove special characters
        processed_queries.append(unidecode(query))
        # Remove punctuation
        processed_queries[i] = processed_queries[i].translate(str.maketrans('', '', string.punctuation))
        # Lowercase all words
        processed_queries[i] = processed_queries[i].lower()
        # Remove stop words
        processed_queries[i] = ' '.join([word for word in processed_queries[i].split() if word not in stop_words])
        # Perform word tokenization
        processed_queries[i] = word_tokenize(processed_queries[i])
    return processed_queries


def split_training_data(ids, t1s, t2s):
    # Initialize empty lists to store the split data
    new_ids = []
    new_t1s = []
    new_t2s = []
    article_id = ''

    for id, t1, t2 in zip(ids, t1s, t2s):
        t1_split = t1.splitlines()
        for line in t1_split:
            if len(line) == 0: continue
            elif line.startswith("Article"):
                # Get id
                match = re.search(r"Article\s+(\d+(?:-\d+)?)", line)
                article_id = "Article " + match.group(1)
            else:
                temp = article_id + " " + line
                new_ids.append(id)
                new_t1s.append(temp)
                new_t2s.append(t2)

    return new_ids, new_t1s, new_t2s


def create_labeled_data(corpus, relevant_docs, queries):
    data_label0 = []
    data_label1 = []
    for relevant_doc, query in zip(relevant_docs, queries):
        relevant_doc_id = re.search(r"Article\s+(\d+(?:-\d+)?)", relevant_doc)
        for doc in corpus:
            doc_id = re.search(r"Article\s+(\d+(?:-\d+)?)", doc)
            if doc_id.group(1) == relevant_doc_id.group(1): data_label1.append(InputExample(texts=[query, doc], label=1.0))
            else : data_label0.append(InputExample(texts=[query, doc], label=0.0))
    sampled_data_label_0 = random.sample(data_label0, int(len(data_label0) * 0.05))
    data = data_label1 + sampled_data_label_0
    return data


def join_query_docs_sets(queries, relevant_docs):
    query_doc_dict = {}
    
    for i, query in enumerate(queries):
        if query not in query_doc_dict:
            query_doc_dict[query] = set()
        query_doc_dict[query].add(relevant_docs[i])

    return query_doc_dict


def average_words_per_segment(segments):
    total_words = sum(len(segment[0].split()) for segment in segments)
    average_words = total_words / len(segments)
    return round(average_words)


def normalize_scores(tuple_list):
    scores = [float(t[2]) for t in tuple_list]

    min_value = min(scores)
    max_value = max(scores)

    normalized_values = ((t[0], t[1], (float(t[2]) - min_value) / (max_value - min_value)) for t in tuple_list)

    return list(normalized_values)