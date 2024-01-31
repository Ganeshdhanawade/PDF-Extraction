import google.generativeai as genai
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from PyPDF2 import PdfReader,PdfFileReader
import os
import io

from langchain.vectorstores import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class utils:
    ## reade each pdf and get text insite it
    # def get_pdf_text(pdf_docs):
    #     text=''
    #     for pdf in pdf_docs:
    #         string_io = io.StringIO(pdf.decode('utf-8')) #getvalues())
    #         pdf_reader=PdfReader(string_io)
    #         for page in pdf_reader.pages:
    #             text+=page.extract_text()
    #         return text
        
    
    def get_pdf_text(pdf_docs):
        text = ''
        for pdf in pdf_docs:
            with io.BytesIO(pdf) as pdf_file:  # Create file-like object
                    pdf_reader = PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    return text

    ## find the chunks of pdf
    def get_text_chunks(text):
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
        chunks=text_splitter.split_text(text)
        return chunks

    ### store that chunks in vectors with text embedding
    def get_vector_store(text_chunks):
        embeddings=GoogleGenerativeAI(model='models/embedding-001')
        vector_store=faiss.FAISS.from_texts(text_chunks,embedding=embeddings)
        vector_store.save_local('faiss_index')

    ### create that quation answer of pdf containt
    def get_conversational_chain():

        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        model=ChatGoogleGenerativeAI(model='gemini_pro',temperature=0.2)

        Prompt=PromptTemplate(template=prompt_template,input_variables=['context','quation'])
        chain=load_qa_chain(model,chain_type='stuff',prompt=Prompt)
        return chain