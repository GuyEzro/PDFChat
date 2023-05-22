import os
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ##About
    This app in an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    -[LangChain](https://python.langchain.com/)
    -[OpenAI](https://platform.openai.com/docs/models) LLM nodel
    ''')
    st.write('Made with love by [Prompt Engineer](https://youtube.com@engineerprompt)')

def main():
    st.header('Chat with PDF')
    load_dotenv()
    pdf = st.file_uploader("upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                text += page_text
            except:
                st.warning(f"Text extraction failed for page "
                           f"{page.page_number + 1}. Consider alternative methods for extracting text from this page.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
        else:
            embedding = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embedding)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = vector_store.similarity_search(query=query, k=3)
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                 response = chain.run(input_documents=docs, question=query)
                 st.write(cb)
            st.write(response)


if __name__ == '__main__':
    main()


