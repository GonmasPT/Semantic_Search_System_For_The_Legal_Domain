This repository contains the code used to build the semantic search system. The main is located in the file ThesisModelOriginal.py

The first time you run it will take a while, because the SBERT model needs to be fine-tuned before you can used it. For future runs it will only take a few seconds for the system to load everything.

The system will give you two options: you either choose to run the test queries or you choose to manually insert queries. If you choose the second option then you must first create a .env file with an Openai api key, since the system uses an openai chatmodel to generate the response based on the articles that the search system retrieved.
