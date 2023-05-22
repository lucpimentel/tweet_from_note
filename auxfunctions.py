import openai
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.document_loaders import UnstructuredFileLoader
import re




def format_text(input_string: str) -> str:
    """
    Formats the input string by adding line breaks based on periods and question marks.
    
    Args:
        input_string: The input string to be formatted.
        
    Returns:
        The formatted string with line breaks added.
    """
    formatted_lines = []

    sentences = re.split(r'(?<=[?.])', input_string)  # Split the input string based on periods and question marks

    period_count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence.endswith('.'):
            period_count += 1
            if period_count % 2 == 0:  # Jump a line every two periods
                formatted_lines.append(sentence + '\n\n')
            else:
                formatted_lines.append(sentence + ' ')
        elif sentence.endswith('?'):
            formatted_lines.append(sentence + '\n\n')  # Add the sentence with two line breaks at the end
            period_count = 0  # Reset period count after a question mark
        else:
            formatted_lines.append(sentence + ' ')  # Add the sentence with a space at the end

    formatted_text = ''.join(formatted_lines)  # Join the formatted sentences into a single string
    return formatted_text.strip()




def load_vectorstore_db(api_key):
    openai_embeddings = OpenAIEmbeddings(openai_api_key = api_key)
    db = FAISS.load_local('faiss_index',openai_embeddings)
    return db


def get_all_txt_files():
    """
    Loop through all txt files in the current directory and load them using UnstructuredFileLoader.
    """
    
    # Loop through all files in the current directory
    for filename in os.listdir():
        # If the file is a text file, load it using a UnstructuredFileLoader
        if filename.endswith(".txt"):
            loader = UnstructuredFileLoader(os.path.join(filename))
            txts = loader.load()
    
    # Return the list of loaded text files
    return txts


def openai_api_call(prompt: str, template: str, model = 'gpt-3.5-turbo', temperature: int = 0.5, top_p:int = 0.5, max_tokens:int = 1000) -> str:
        """
        Calls the OpenAI API with a given prompt

        Args:
            prompt (str): The prompt to use for generating the text.
        
        Returns:
            str: The generated text.
        """
        # Create the completion call using the OpenAI API
        
        
        response = openai.ChatCompletion.create(model=model,
                                                temperature = temperature,
                                                top_p = top_p,
                                                max_tokens = max_tokens,
                    messages=[{"role": "system", "content": template},
                            {"role": "user", "content": prompt}]#f'Please create a tweet based on this note {prompt}:'}]
                            )
        return response['choices'][0]['message']['content']
        


def generate_tweet_from_note(note, llm_chain):
    try:
        tweet = llm_chain.run(input= note)
    except:
         tweet = llm_chain({'style':style,'input':note})['text']
    return tweet


def generate_tweets(list_of_personal_notes, llm_chain) -> pd.DataFrame:
    """
    Generate tweets from a list of personal notes using a language model chain.

    Args:
        list_of_personal_notes (List[str]): A list of personal notes.
        llm_chain (Any): The language model chain used to generate tweets.

    Returns:
        pd.DataFrame: A DataFrame containing the generated tweets with their corresponding personal notes.
    """

    results_list = []

    for personal_note in list_of_personal_notes:
        for i in range(2):
            unformatted_tweet = generate_tweet_from_note(personal_note, llm_chain)  # Call your function to generate an unformatted tweet
            tweet = format_text(unformatted_tweet)  # Call your function to format the tweet

            results_list.append({'Personal Note': personal_note, 'Tweet': tweet})

    generated_tweets = pd.DataFrame(results_list)
    return generated_tweets

