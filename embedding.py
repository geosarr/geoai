import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from constants import CHROMA_PATH
from langchain_core.documents import Document


def create_chroma_database(
    data: list[Document] | list[str],
    embedding: OpenAIEmbeddings,
    override: bool = False,
    path: str = CHROMA_PATH,
):
    if len(data) == 0:
        print("No data to save.")
        return
    if os.path.exists(path) and not override:
        print(f"It seems a chroma database already exists at {path}.")
        try:
            return Chroma(persist_directory=path, embedding_function=embedding)
        except Exception as e:
            print(f"Error loading existing database: {e}")
            print("Please delete the existing database or set override=True.")
            return
    data_element = data[0]
    printer = lambda: print(f"Saved {len(data)} documents to {path}.")
    if isinstance(data_element, Document):
        db = Chroma.from_documents(data, embedding, persist_directory=path)
        printer()
        return db
    if isinstance(data_element, str):
        db = Chroma.from_texts(data, embedding, persist_directory=path)
        printer()
        return db


def get_retriever(embedding: OpenAIEmbeddings, path: str = CHROMA_PATH):
    ## Reading the persisted Chroma database
    vectordb = Chroma(persist_directory=path, embedding_function=embedding)
    return vectordb.as_retriever()
