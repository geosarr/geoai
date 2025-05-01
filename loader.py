import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm.auto import tqdm
from pypdf import PdfReader

from constants import CHROMA_PATH
from embedding import create_chroma_database
from splitter import recursive_splitter, split_documents


def load_pdf(path: str):
    return PyPDFLoader(path).load()


def lazy_load_pdf(path: str):
    return PyPDFLoader(path).lazy_load()


def load_pdf_lazily(
    path: str, max_nb_docs: int | None = None, write_stdout: bool = False
):
    total_nb_pages = len(PdfReader(path).pages)
    nb_pages = (
        min(max_nb_docs, total_nb_pages)
        if isinstance(max_nb_docs, int)
        else total_nb_pages
    )
    lazy_docs = lazy_load_pdf(path)
    documents = [Document("")] * nb_pages
    for page, doc in tqdm(
        enumerate(lazy_docs),
        total=nb_pages,
        desc=f"Loading {nb_pages} pages from {path}",
    ):
        if page >= nb_pages:
            break
        documents[page] = doc
        if write_stdout:
            tqdm.write(doc.page_content)
            tqdm.write("=======================================================\n")
    return documents


def pre_process(
    path: str, chunk_size: int = 1000, chunk_overlap: int = 200, override: bool = False
):
    if os.path.exists(CHROMA_PATH) and not override:
        load_dotenv()
        embedding = OpenAIEmbeddings()
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
    docs = load_pdf_lazily(
        path
    )  # TODO use here a load function that loads any common document format (.pdf, .docx, .md, etc)

    ## Split the data
    text_splitter = recursive_splitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = split_documents(docs, text_splitter)

    ## Loading OPENAI_API_KEY and OpenAI embeddings
    load_dotenv()
    embedding = OpenAIEmbeddings()
    return create_chroma_database(documents, embedding, override, CHROMA_PATH)


if __name__ == "__main__":
    path = "data/s3-api.pdf"
    # docs = lazy_load_pdf(path)
    # print(next(docs).page_content)

    vector_db = pre_process(
        path=path, chunk_size=1000, chunk_overlap=200, override=False
    )
    response = vector_db.as_retriever().invoke("Which languages are supported by s3?")
    for doc in response:
        print(doc.page_content)
        print("=======================================================\n")
