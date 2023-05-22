import streamlit as st
import os
import re
import openai
import pandas as pd
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from useful_variables import tweet_from_note_template, tweet_editor_template
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from auxfunctions import (openai_api_call, load_vectorstore_db, get_all_txt_files, format_text, generate_tweets)

def create_llmchain():
    llm = ChatOpenAI(temperature = 1,openai_api_key = openai_api_key)
    prompt = PromptTemplate(input_variables = ["input"],template = tweet_from_note_template)
    llm_chain = LLMChain(llm = llm, prompt = prompt)
    return llm_chain

def create_sequential_llmchain():

    # Creating generate tweet chain
    llm = ChatOpenAI(temperature = 0.5,openai_api_key = openai_api_key)
    generate_tweet_prompt = PromptTemplate(input_variables = ["style","input"],template = tweet_from_note_template)
    generate_tweet_chain = LLMChain(llm = llm, prompt = generate_tweet_prompt,output_key='tweet')
    

    # Creating tweet editor chain
    tweet_editor_prompt = PromptTemplate(input_variables = ['tweet'], template = tweet_editor_template)
    editor_chain = LLMChain(llm = llm, prompt = tweet_editor_prompt)
    
    # creating sequential_chain
    sequential_chain = SequentialChain(chains = [generate_tweet_chain,editor_chain], input_variables = ['style','input'])
    return sequential_chain




st.title('Welcome to the Tweets From Notes GPT App!')
st.write('---')
st.write('''

This small app serves to organize all of your notes and ramblings ino readable and understandable format.
Simply upload a .txt file with each one of your notes surrounded by triple accents (```) to have them edited.

''')

openai_api_key = st.text_input('What is you OpenAI API key?')


text = st.file_uploader("Upload a .txt file with all of your notes")

if text and openai_api_key:
    block_of_text = str(text.read())


    list_of_personal_notes = re.findall(r'```([\s\S]*?)```', block_of_text)
    


    llm_chain = create_llmchain()
    #llm_chain = create_sequential_llmchain()


    if st.button("Create Tweets"):
        # Show a loading message while the code snippet is executing
        with st.spinner("Running code..."):
            generated_tweets = generate_tweets(list_of_personal_notes[:2],llm_chain)
    
        # Get the current date and time as a string
        now = datetime.now().strftime("%d-%m-%Y")

        # Write the dataframe to an Excel file with the current timestamp in the file name
        file_name = f"Generated tweets {now}.xlsx"
        #excel_file = generated_tweets.to_excel(index = False)

    
try:
    st.dataframe(generated_tweets)
except:
    pass