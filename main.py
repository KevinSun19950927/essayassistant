from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

openai_api_key = st.text_input('Enter your openai api key here:')

upload_file = st.file_uploader("Upload a PDF file", type=['pdf'])

chain_type = st.selectbox(
    "Please choose your chain type",
    ('stuff', 'map_reduce', 'refine', 'map_rerank')
)

num_chunks = st.slider('Number of chunks:', 0, 100, 20)


query = st.text_input('What can essay assistant help you?', 'What are examples of good data science teams')


def qa(file_path, query, chain_type, num_chunks):

    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()

    # Chunk data into smaller documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    # Creating embeddings of your documents to get ready for semantic search
    # docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    # query = st.text_input('What can essay assistant help you?', 'What are examples of good data science teams')
    # docs = docsearch.similarity_search(query, include_metadata=True)
    # st.write(docs[0].page_content[:250])
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": num_chunks})
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    print(result['result'])
    return result


def qa_result():
    os.environ["OPENAI_API_KEY"] = openai_api_key
    if upload_file is not None:
        saved_path = os.path.join("../", upload_file.name)

        with open(saved_path, 'wb') as f:
            f.write(upload_file.getbuffer())
        if query:
            response = qa(saved_path, query, chain_type, num_chunks)
            st.write(response["result"])


run = st.button('Run')
if run:
    qa_result()






















