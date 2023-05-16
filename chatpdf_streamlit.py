from dotenv import load_dotenv 
import os 
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain import ConversationChain
from langchain.chains import RetrievalQA




def main():
    load_dotenv() 
    st.set_page_config(page_title='Chat with PDF')
    st.header('PDF Chat 💬')

    # upload the pdf 
    pdf = st.file_uploader('Upload your PDF', type = 'pdf')

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() 
        
        # split the text into chunks 
        text_splitter = CharacterTextSplitter(
            separator = '\n',
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text) 

        # create embeddings 
        embeddings = OpenAIEmbeddings() 
        knowledge_db = FAISS.from_texts(chunks, embeddings)
        retriever = knowledge_db.as_retriever() 
        # building user ui
        question = st.text_input('Ask me a question')

        if question:
            docs = knowledge_db.similarity_search(question)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type = 'map_reduce')
            response = chain.run(input_documents = docs, question = question)
            st.write(response) 

    
if __name__ == "__main__":
    main() 

