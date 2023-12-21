from dotenv import load_dotenv
import openai
import os

load_dotenv()  # take environment variables from .env.
openai.api_key = os.getenv("OPENAI_API_KEY")

system_message = f"""
You are a friendly chatbot tasked with helping users regarding the Japanese Civil Law. \
You will be provided with a set of relevant articles from the Japanese Civil Law \
to help you answer the user query. Each article is delimited with #### characters. \
There can be in total 4 articles in the set.
You must use these articles to formulate your own answer to the query \
asked by the user. Your task is answer if you agree or not with the query and provide helpfull information to the user regarding the context of the query. \
The user query is delimited by <> characters.
If the context given by the articles does not contain the information necessary to answer the \
user query, then just say 'I do not possess the information to provide an answer', dont try to make up the answer.
"""

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
#     print(str(response.choices[0].message))
    return response.choices[0].message["content"]