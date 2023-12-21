from ThesisModelFunctionsOriginal import *
from ChatModelHelperFunctions import *
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, losses, util, InputExample
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
import panel as pn
from bokeh.models.widgets import Button
from bokeh.models import TextInput
from bokeh.models import Div
from openai.embeddings_utils import get_embedding, cosine_similarity


models = ["all-roberta-large-v1",
          "all-MiniLM-L6-v2",
          "all-mpnet-base-v2",
          "multi-qa-mpnet-base-dot-v1",
          "OpenAI",
]
chat_models = ["OpenAI",
             "Anthropic"
]


def main():
    torch.cuda.empty_cache()
    model_name = models[3] # Choose embedding model
    chat_model_name = chat_models[0] # Choose chat model

    # Retrieve all data and queries
    article_id__article = retrieve_corpus()
    query_id__query, query_id__rel_docs_id = retrieve_training_data()
    art_id_list = list(article_id__article.keys())
    article_list = list(article_id__article.values())

    #--------------------------------------------------------------------------
    # Prepare train, val and test data
    #--------------------------------------------------------------------------

    pickle_file = "original_split_data.pkl"
    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as f:
            training_data, validation_data, test_data, val_query_id__query, test_query_id__query, test_query_id__rel_docs_id = pickle.load(f)
    else:
        # Shuffle queries
        keys = list(query_id__query.keys())
        random.shuffle(keys)
        shuffled_query_id__query = {key: query_id__query[key] for key in keys}
        shuffled_query_id__rel_docs_id = {key: query_id__rel_docs_id[key] for key in keys}

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

        train_query_id__rel_docs_id = {key: value for idx, (key, value) in enumerate(shuffled_query_id__rel_docs_id.items()) if idx < len_train}
        val_query_id__rel_docs_id = {key: value for idx, (key, value) in enumerate(shuffled_query_id__rel_docs_id.items()) if idx >= len_train and idx < len_train+len_val}
        test_query_id__rel_docs_id = {key: value for idx, (key, value) in enumerate(shuffled_query_id__rel_docs_id.items()) if idx >= len_train+len_val}
        
        # Create the training list
        training_data = []
        for query, rel_docs_id in zip(list(train_query_id__query.values()), list(train_query_id__rel_docs_id.values())):
            for rel_doc in rel_docs_id:
                    training_data.append(InputExample(texts=[query, article_id__article[rel_doc]]))

        # Create validation list
        validation_data = defaultdict()
        for query_id, rel_docs_id in zip(list(val_query_id__query.keys()), list(val_query_id__rel_docs_id.values())):
            rel_docs_set = set()
            for rel_doc in rel_docs_id:
                    rel_docs_set.add(rel_doc)
            validation_data[query_id] = rel_docs_set

        # Create test list
        test_data = []
        for query_id, rel_docs_id in zip(list(test_query_id__query.keys()), list(test_query_id__rel_docs_id.values())):
            for rel_doc in rel_docs_id:
                    test_data.append(InputExample(texts=[query_id, article_id__article[rel_doc]]))

        with open(pickle_file, "wb") as f:
            pickle.dump((training_data, validation_data, test_data, val_query_id__query, test_query_id__query, test_query_id__rel_docs_id), f)

    #--------------------------------------------------------------------------
    # Prepare BM25
    #--------------------------------------------------------------------------

    corpus_processed = lexical_process_corpus(article_list)
    queries_processed = lexical_process_queries(list(test_query_id__query.values()))
    bm25 = BM25Okapi(corpus_processed)

    #--------------------------------------------------------------------------
    # Prepare BERT
    #--------------------------------------------------------------------------

    user_model = int(input("1 -> Local Model; 2 -> Openai Model\nType 1 or 2: "))
    output_dir = 'model_original'
    model = None

    if user_model == 1: # Use Local model
        if os.path.exists(output_dir):
            model = SentenceTransformer(output_dir)
        else:
            train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)
            loss_function = losses.MultipleNegativesRankingLoss(model=model_name)
            evaluator = InformationRetrievalEvaluator(val_query_id__query, article_id__article, validation_data, batch_size=16)
            model = SentenceTransformer(model_name)
            model = model.to('cuda')

            # Configure the training
            model.fit(train_objectives=[(train_dataloader, loss_function)],
                    evaluator=evaluator,
                    epochs=50,
                    warmup_steps=20,
                    optimizer_params={'lr': 0.00011718867818415094, 'weight_decay': 0.03416775724773422},
                    #output_path=output_dir,
                    callback=callback_model_score,
                    show_progress_bar=True)
            model.save(output_dir)

    elif user_model == 2: # Use Openai Model
        pass

    elif user_model == 3: # Perform hyper-parameter tuning
        if os.path.exists(output_dir):
            sys.exit("Terminating the program. Delete existing model or choose another mode.")
        else:
            def train_transformer(config):
                train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)
                loss_function = losses.MultipleNegativesRankingLoss(model=model_name)
                evaluator = InformationRetrievalEvaluator(val_query_id__query, article_id__article, validation_data, batch_size=16)
                model = SentenceTransformer(model_name)

                # Configure the training
                model.fit(train_objectives=[(train_dataloader, loss_function)],
                        evaluator=evaluator,
                        epochs=1,
                        warmup_steps=config["warmup_steps"],
                        optimizer_params={'lr': config["lr"], 'weight_decay': config["weight_decay"]},
                        show_progress_bar=True)

                score = evaluator(model)
                session.report({"score": score})

            # Define search space
            space = {
                "lr": tune.loguniform(1e-6, 1e-1),
                "warmup_steps": tune.choice([20, 25, 30, 35]),
                "weight_decay": tune.loguniform(1e-6, 1e-1)
            }
            pbt = PopulationBasedTraining(
                time_attr="training_iteration",
                metric="score",
                mode="max",
                perturbation_interval=1,
                hyperparam_mutations={
                "lr": tune.loguniform(1e-6, 1e-2),
                "warmup_steps": tune.choice([20, 25, 30, 35]),
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
                num_samples=10,
                storage_path='./ray_results_original',
                scheduler=pbt  # Use the PBT scheduler
                #reuse_actors=True
            )
            # Get the best trial
            best_config = analysis.get_best_config("score", "max", "last")

            # Save all trial data to a file
            with open('trial_data_v2.json', 'w') as f:
                json.dump(best_config, f)

            sys.exit("Tuning complete. Terminating the program.")

    else:
        sys.exit("Terminating the program. Please choose a valid mode.")


    #--------------------------------------------------------------------------
    # Perform Search
    #--------------------------------------------------------------------------

    pickle_file = "corpus_embeddings_local.pkl"
    if os.path.isfile(pickle_file):
        with open(pickle_file, "rb") as f:
            corpus_embeddings = pickle.load(f)
    else:
        corpus_embeddings = model.encode(article_list, batch_size=16, convert_to_tensor=True)
        with open(pickle_file, "wb") as f:
            pickle.dump(corpus_embeddings, f)

    semantic_scores = []
    lexical_scores = []

    user_input = int(input("Please choose mode (4-> Use test data queries; 5-> Read new input query): "))

    if user_input == 4: # Use test data queries
        if user_model == 1 or user_model == 3:
            # Semantic Scores
            for query_id, query in zip(list(test_query_id__query.keys()), list(test_query_id__query.values())):
                query_embedding = model.encode(query, batch_size=16, convert_to_tensor=True)
                cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].tolist()
                for i, score in enumerate(cos_scores):
                    semantic_scores.append((query_id, art_id_list[i], "{:.2f}".format(score)))

            #Lexical Scores
            for query_id, query in zip(list(test_query_id__query.keys()), queries_processed):
                # Search and Rank the most similar articles to the query
                bm25_scores = bm25.get_scores(query).tolist()
                for i, score in enumerate(bm25_scores):
                    lexical_scores.append((query_id, art_id_list[i], "{:.2f}".format(score)))

            # Combine scores
            top_articles_filtered = combine_scores(semantic_scores, lexical_scores)

        else:
            pickle_file = "corpus_embeddings_openai.pkl"
            if os.path.isfile(pickle_file):
                with open(pickle_file, "rb") as f:
                    corpus_embeddings = pickle.load(f)
            else:
                corpus_embeddings = [get_embedding(article, engine='text-embedding-ada-002') for article in article_list]
                with open(pickle_file, "wb") as f:
                    pickle.dump(corpus_embeddings, f)

            for query_id, query in zip(list(test_query_id__query.keys()), list(test_query_id__query.values())):
                query_embedding = get_embedding(query, engine='text-embedding-ada-002')
                cos_scores = [cosine_similarity(corpus_embedding, query_embedding) for corpus_embedding in corpus_embeddings]
                for i, score in enumerate(cos_scores):
                    semantic_scores.append((query_id, art_id_list[i], "{:.2f}".format(score)))

            top_articles_filtered = combine_scores(semantic_scores)

        # Output final scores
        with open('FinalScores.txt', 'w') as output_file:
            for query_id, articles in top_articles_filtered.items():
                output_file.write(f"{articles}" + '\n')

        # Calculate evaluation metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for query_id, top_articles_tuples in top_articles_filtered.items():
            top_articles_set = {tup[1] for tup in top_articles_tuples} # Top articles retrieved by the search engine
            relevant_docs = set(test_query_id__rel_docs_id[query_id]) # query its the query id / these are the relevant articles from the dataset (not from the search)

            tp = len(top_articles_set.intersection(relevant_docs))
            fp = len(top_articles_set.difference(relevant_docs))
            fn = len(relevant_docs.difference(top_articles_set))

            true_positives += tp
            false_positives += fp
            false_negatives += fn

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        beta = 2
        f2_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F2-score: {f2_score:.2f}')

    elif user_input == 5: # Read input
        # Create a Panel TextInput widget for user input
        input_field = TextInput(value="Type something here", align='center', sizing_mode="scale_width")
        output_text = Div(text="Response will be displayed here", align='center')
        input_query = ""
        response = ""

        def on_button_click():
            global input_query, response
            input_query = input_field.value
            response = ""
            semantic_scores = []
            lexical_scores = []

            if input_query.lower() == 'exit':
                response = "A terminar o programa."
            else:
                #Semantic Scores
                query_embedding = model.encode(input_query, batch_size=16, convert_to_tensor=True)
                cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].tolist()
                for i, score in enumerate(cos_scores):
                    semantic_scores.append((input_query, art_id_list[i], "{:.2f}".format(score)))

                #Lexical Scores
                bm25_scores = bm25.get_scores(input_query).tolist()
                for i, score in enumerate(bm25_scores):
                    lexical_scores.append((input_query, art_id_list[i], "{:.2f}".format(score)))

                # Combine scores
                top_articles_filtered = combine_scores(semantic_scores, lexical_scores)

                # Output final scores
                article_set = ""
                lines = ""
                print("\RELEVANT RESULTS:\n")
                for query_id, articles in top_articles_filtered.items():
                    for article in articles:
                        article_set += f"#### {article_id__article[article[1]]} ####\n"
                        lines += "Article " + article[1] + "\n\n"
                        print("Article " + article[1] + " with score " + "{:.2f}".format(article[2]))
                        print("\nArticle " + article[1] + ": " + article_id__article[article[1]] + "\n\n")

                # Chatmodel
                if chat_model_name == "OpenAI":
                    messages = [{"role":"system", "content":system_message}]
                    messages.append({"role":"user", "content":f"User query:\n<{input_query}>"})
                    messages.append({"role":"assistant", "content":f"Set of relevant articles:\n{article_set}"})
                    chat_response = get_completion_from_messages(messages, temperature=0)

                response = chat_response + "\n\n" + "Relevant Articles:\n\n" + lines

            input_field.value = ""  # Clear the input field after submission
            output_text.text = response.replace("\n", "<br>")

        # Create a Panel button to submit user input
        submit_button = Button(label="Submit", align='center')
        submit_button.on_click(on_button_click)

        # Create a Panel app with the widgets
        app = pn.Column(
            input_field,
            submit_button,
            output_text,
            sizing_mode="stretch_width",  # Adjust the sizing_mode for the entire layout
        )

        # Show the app in a browser
        app.show()

    else:
        sys.exit("Terminating the program. Please choose a valid mode.")


if __name__ == '__main__':
    main()