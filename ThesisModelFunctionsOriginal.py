from collections import defaultdict
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import string
import os.path
import pickle


def retrieve_corpus():
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
                articles[article_id] = line.strip()
            elif re.match("\((\d+|[ivxlcdm]+)\)", line):
                articles[article_id] = articles[article_id] + " " + line.strip()
        return articles


def retrieve_training_data():
    pickle_file = "training_data.pkl"
    query_id__query = defaultdict(list)
    query_id__rel_docs = defaultdict(list)

    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as f:
            query_id__query, query_id__rel_docs = pickle.load(f)
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
                    query_id__query[pair.get('id')] = t2.text.strip()
                    t1 = t1.text.strip()
                    t1_split = t1.splitlines()

                    for line in t1_split:
                        if len(line) == 0: continue
                        elif line.startswith("Article"):
                            # Get id
                            match = re.match(r"Article\s+(\d+(?:-\d+)?)", line)
                            query_id__rel_docs[pair.get('id')].append(match.group(1))
                        else: continue

        with open(pickle_file, "wb") as f:
            pickle.dump((query_id__query, query_id__rel_docs), f)
    return query_id__query, query_id__rel_docs


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


def normalize_scores(tuple_list):
    scores = [float(t[2]) for t in tuple_list]

    min_value = min(scores)
    max_value = max(scores)

    epsilon = 1e-7  # A small constant value
    normalized_values = ((t[0], t[1], (float(t[2]) - min_value) / (max_value - min_value + epsilon)) for t in tuple_list)

    return list(normalized_values)


def callback_model_score(score, epoch, steps):
    # This function will be called after each epoch. 'score' is the evaluator's score for this epoch.
    print(f"After epoch {epoch}, the evaluator score is {score}")


def combine_scores(semantic_scores, lexical_scores=None):
    normalized_semantic_scores = semantic_scores
    if lexical_scores: normalized_lexical_scores = normalize_scores(lexical_scores)

    # Combine the two lists
    combined_scores = []
    if lexical_scores:
        for lexical_score, semantic_score in zip(normalized_lexical_scores, normalized_semantic_scores):
            combined_scores.append((lexical_score[0], lexical_score[1], (lexical_score[2] + float(semantic_score[2])) / 2))
    else:
        for semantic_score in normalized_semantic_scores:
            combined_scores.append(semantic_score[0], semantic_score[1], float(semantic_score[2]))

    #combined_scores = normalized_semantic_scores
    #combined_scores = normalized_lexical_scores

    # Group the data by query
    query_id_group = defaultdict(list)
    for tup in combined_scores:
        query_id_group[tup[0]].append(tup)

    # Sort the tuples within each group by score and select the top 4 segments
    top_articles_filtered = {}

    for query_id, tuples in query_id_group.items():
        sorted_tuples = sorted(tuples, key=lambda x: x[2], reverse=True)
        #filtered_articles = sorted_tuples[:2]
        filtered_articles = [sorted_tuples[0]]  # Start with the top segment
        top_1_score = sorted_tuples[0][2]  # Get the score of the top segment
        for tup in sorted_tuples[1:]:  # Iterate over the remaining sorted_tuples
            if len(filtered_articles) >= 4:  # Ensure there are no more than 4 segments per query
                break

            if float(tup[2]) >= float(top_1_score) * 0.91 and len(filtered_articles) < 2:
                filtered_articles.append(tup)
            elif float(tup[2]) >= float(top_1_score) * 0.85 and len(filtered_articles) >= 2:
                filtered_articles.append(tup)
            else:
                break  # Break the loop since the tuples are sorted in descending order, and the remaining ones won't meet the conditions

        top_articles_filtered[query_id] = filtered_articles

    return top_articles_filtered