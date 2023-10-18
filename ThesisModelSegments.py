from ThesisModelFunctionsOriginal import *
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, losses, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import torch
import random
import pickle
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.air import session
import json
import os
import sys


#model = SentenceTransformer('all-MiniLM-L6-v2')
#model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
#model = SentenceTransformer('multi-qa-distilbert-cos-v1')
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')


if __name__ == '__main__':

    torch.cuda.empty_cache()

    article_id__segments = retrieve_segmented_corpus() # replaced by article_id__segment
    query_id__query, query_id__rel_docs_id = retrieve_training_data()

    # Arrange data in the evaluator format
    article_id__segment = {}
    for art_id, segments in zip(list(article_id__segments.keys()), list(article_id__segments.values())):
        segment_count = 0
        for segment in segments:
            article_id__segment[art_id + '--' + str(segment_count)] = segment
            segment_count += 1

    art_id_list = list(article_id__segment.keys())
    segment_list = list(article_id__segment.values())
    query_id__rel_segments_id = {} # Create a new dictionary to store the updated data

    for query_id, rel_docs in query_id__rel_docs_id.items():
        updated_rel_docs = []
        
        for id in rel_docs:
            num_segments = 0
            for art_id in art_id_list:
                if art_id.split('--')[0] == id: num_segments += 1

            if num_segments < 1: sys.exit("num_segments < 1")
            updated_rel_docs.extend([f"{id}--{i}" for i in range(0, num_segments)])
        
        query_id__rel_segments_id[query_id] = updated_rel_docs


    pickle_file = "segment_split_data.pkl"
    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as f:
            training_data, validation_data, test_data, val_query_id__query, test_query_id__query, test_query_id__rel_segments_id = pickle.load(f)
    else:
        # Shuffle queries
        keys = list(query_id__query.keys())
        random.shuffle(keys)
        shuffled_query_id__query = {key: query_id__query[key] for key in keys}
        shuffled_query_id__rel_segments_id = {key: query_id__rel_segments_id[key] for key in keys}

        # Split the data training, validation and test sets
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        len_train = int(train_ratio * len(shuffled_query_id__query))
        len_val = int(val_ratio * len(shuffled_query_id__query))
        len_test = int(test_ratio * len(shuffled_query_id__query))

        train_query_id__query = {key: value for idx, (key, value) in enumerate(shuffled_query_id__query.items()) if idx < len_train}
        val_query_id__query = {key: value for idx, (key, value) in enumerate(shuffled_query_id__query.items()) if idx >= len_train and idx < len_train+len_val}
        test_query_id__query = {key: value for idx, (key, value) in enumerate(shuffled_query_id__query.items()) if idx >= len_train+len_val}

        train_query_id__rel_segments_id = {key: value for idx, (key, value) in enumerate(shuffled_query_id__rel_segments_id.items()) if idx < len_train}
        val_query_id__rel_segments_id = {key: value for idx, (key, value) in enumerate(shuffled_query_id__rel_segments_id.items()) if idx >= len_train and idx < len_train+len_val}
        test_query_id__rel_segments_id = {key: value for idx, (key, value) in enumerate(shuffled_query_id__rel_segments_id.items()) if idx >= len_train+len_val}
        
        # Create the training InputExamples of all data
        training_data = []
        for query, rel_docs_id in zip(list(train_query_id__query.values()), list(train_query_id__rel_segments_id.values())):
            for rel_doc in rel_docs_id:
                    training_data.append(InputExample(texts=[query, article_id__segment[rel_doc]]))

        # Create validation list
        validation_data = defaultdict()
        for query_id, rel_docs_id in zip(list(val_query_id__query.keys()), list(val_query_id__rel_segments_id.values())):
            rel_docs_set = set()
            for rel_doc in rel_docs_id:
                    rel_docs_set.add(rel_doc)
            validation_data[query_id] = rel_docs_set

        # Create test list
        test_data = []
        for query_id, rel_docs_id in zip(list(test_query_id__query.keys()), list(test_query_id__rel_segments_id.values())):
            for rel_doc in rel_docs_id:
                    test_data.append(InputExample(texts=[query_id, article_id__segment[rel_doc]]))

        with open(pickle_file, "wb") as f:
            pickle.dump((training_data, validation_data, test_data, val_query_id__query, test_query_id__query, test_query_id__rel_segments_id), f)

    #--------------------------------------------------------------------------
    # Get BERT ready
    #--------------------------------------------------------------------------

    user_input = int(input("Please choose mode (1-> Use existing model; 2-> Train new model; 3-> Run hyperparameter tuning): "))
    output_dir = 'model_segments'

    if user_input == 1: # Use existing model

        if os.path.exists(output_dir):
            model = SentenceTransformer(output_dir)
        else:
            sys.exit("Terminating the program. No model found, choose another mode.")    

    elif user_input == 2: # Train new model

        if os.path.exists(output_dir):
            sys.exit("Terminating the program. Delete existing model or choose another mode.")
        else:
            train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)
            loss_function = losses.MultipleNegativesRankingLoss(model=model)
            evaluator = InformationRetrievalEvaluator(val_query_id__query, article_id__segment, validation_data, batch_size=16)
            model = model.to('cuda')

            # Configure the training
            model.fit(train_objectives=[(train_dataloader, loss_function)],
                    evaluator=evaluator,
                    epochs=50,
                    warmup_steps=20,
                    #optimizer_params={'lr': 0.00011718867818415094, 'weight_decay': 0.03416775724773422},
                    #output_path=output_dir,
                    callback=callback_model_score,
                    show_progress_bar=True)
            model.save(output_dir)

    elif user_input == 3: # Perform hyper-parameter tuning

        if os.path.exists(output_dir):
            sys.exit("Terminating the program. Delete existing model or choose another mode.")
        else:
            def train_transformer(config):
                train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)
                loss_function = losses.MultipleNegativesRankingLoss(model=model)
                evaluator = InformationRetrievalEvaluator(val_query_id__query, article_id__segment, validation_data, batch_size=16)

                # Configure the training
                model.fit(train_objectives=[(train_dataloader, loss_function)],
                        evaluator=evaluator,
                        epochs=30,
                        warmup_steps=config["warmup_steps"],
                        optimizer_params={'lr': config["lr"], 'weight_decay': config["weight_decay"]},
                        show_progress_bar=True)

                score = evaluator(model)
                session.report({"score": score})

            # Define search space
            space = {
                "lr": tune.loguniform(1e-6, 1e-3),
                "warmup_steps": tune.choice([15, 20, 25, 30]),
                "weight_decay": tune.loguniform(1e-6, 1e-1)
            }

            pbt = PopulationBasedTraining(
                time_attr="training_iteration",
                metric="score",
                mode="max",
                perturbation_interval=1,
                hyperparam_mutations={
                "lr": tune.loguniform(1e-6, 1e-3),
                "warmup_steps": tune.choice([15, 20, 25, 30]),
                "weight_decay": tune.loguniform(1e-6, 1e-2)
                }
            )

            # Initialize Ray
            ray.init()

            # Execute the hyperparameter search
            analysis = tune.run(
                train_transformer,
                resources_per_trial={"cpu": 1, "gpu": 1},
                config=space,
                num_samples=15,
                storage_path='./ray_results_segments',
                scheduler=pbt  # Use the PBT scheduler
                #reuse_actors=True
            )

            # Get the best trial
            best_config = analysis.get_best_config("score", "max", "last")

            # Save all trial data to a file
            with open('trial_data_segments.json', 'w') as f:
                json.dump(best_config, f)

            sys.exit("Tuning complete. Terminating the program.")

    else:
        sys.exit("Terminating the program. Please choose a valid mode.")


    #--------------------------------------------------------------------------
    # Get BM25 ready
    #--------------------------------------------------------------------------

    corpus_processed = lexical_process_corpus(segment_list)
    queries_processed = lexical_process_queries(list(test_query_id__query.values()))
    bm25 = BM25Okapi(corpus_processed)

    #--------------------------------------------------------------------------
    # Semantic Search
    #--------------------------------------------------------------------------

    corpus_embeddings = model.encode(segment_list, batch_size=16, convert_to_tensor=True)
    semantic_scores = []

    user_input = int(input("Please choose mode (4-> Use test data queries; 5-> Read new input query): "))

    if user_input == 4: # Use test data queries

        # Find the closest 4 articles for each query based on cosine similarity
        for query_id, query in zip(list(test_query_id__query.keys()), list(test_query_id__query.values())):
            query_embedding = model.encode(query, batch_size=16, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].tolist()

            for i, score in enumerate(cos_scores):
                semantic_scores.append((query_id, art_id_list[i], "{:.2f}".format(score)))

    elif user_input == 5: # Read input

        input_query = input("Write a query: ")
        query_embedding = model.encode(input_query, batch_size=16, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].tolist()
        
        for i, score in enumerate(cos_scores):
            semantic_scores.append((input_query, art_id_list[i], "{:.2f}".format(score)))

    else:
        sys.exit("Terminating the program. Please choose a valid mode.")
        
    #--------------------------------------------------------------------------
    # Lexical Search
    #--------------------------------------------------------------------------

    lexical_scores = []

    if user_input == 4:

        for query_id, query in zip(list(test_query_id__query.keys()), queries_processed):
            # Search and Rank the most similar articles to the query
            bm25_scores = bm25.get_scores(query).tolist()

            for i, score in enumerate(bm25_scores):
                lexical_scores.append((query_id, art_id_list[i], "{:.2f}".format(score)))

    elif user_input == 5:

            bm25_scores = bm25.get_scores(input_query).tolist()
            
            for i, score in enumerate(bm25_scores):
                lexical_scores.append((input_query, art_id_list[i], "{:.2f}".format(score)))

    #--------------------------------------------------------------------------
    # Combine scores
    #--------------------------------------------------------------------------

    normalized_semantic_scores = semantic_scores
    normalized_lexical_scores = normalize_scores(lexical_scores)

    """ with open('TopArticlesSemantic.txt', 'w') as output_file:
        for score in normalized_semantic_scores:
            output_file.write(score[0] + " " + score[1] + " " + score[2] + '\n')

    with open('TopArticlesLexical.txt', 'w') as output_file:
        for score in normalized_lexical_scores:
            output_file.write(score[0] + " " + score[1] + " " + "{:.2f}".format(score[2]) + '\n') """


    # Combine the two lists
    combined_scores = []
    for lexical_score, semantic_score in zip(normalized_lexical_scores, normalized_semantic_scores):
        combined_scores.append((lexical_score[0], lexical_score[1], (lexical_score[2] + float(semantic_score[2])) / 2))

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

            if tup[2] >= top_1_score * 0.91 and len(filtered_articles) < 2:
                filtered_articles.append(tup)
            elif tup[2] >= top_1_score * 0.85 and len(filtered_articles) >= 2:
                filtered_articles.append(tup)
            else:
                break  # Break the loop since the tuples are sorted in descending order, and the remaining ones won't meet the conditions

        top_articles_filtered[query_id] = filtered_articles

    # Print the results
    if user_input == 4:
        with open('FinalScores(Segments).txt', 'w') as output_file:
            for query_id, articles in top_articles_filtered.items():
                output_file.write(f"{articles}" + '\n')

    elif user_input == 5:
        print("\nSEARCH RESULTS:\n")
        for query_id, articles in top_articles_filtered.items():
            for article in articles:
                print("Article " + article[1] + " with score " + "{:.2f}".format(article[2]))
                print("\nArticle " + article[1] + ": " + article_id__segment[article[1]] + "\n\n")

    #--------------------------------------------------------------------------
    # Evaluate model
    #--------------------------------------------------------------------------

    if user_input == 4: # Evaluate if using test data

        # Calculate evaluation metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        ac = 0

        for query_id, top_articles_tuples in top_articles_filtered.items():
            top_articles_set = {tup[1] for tup in top_articles_tuples} # Top articles retrieved by the search engine
            relevant_docs = set(test_query_id__rel_segments_id[query_id]) # query its the query id / these are the relevant articles from the dataset (not from the search)

            tp = len(top_articles_set.intersection(relevant_docs))
            fp = len(top_articles_set.difference(relevant_docs))
            fn = len(relevant_docs.difference(top_articles_set))

            if tp > 0:
                ac += 1
            true_positives += tp
            false_positives += fp
            false_negatives += fn

        accuracy = ac / len(top_articles_filtered)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        beta = 2
        f2_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F2-score: {f2_score:.2f}')