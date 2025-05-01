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
) -> bool:
    if len(data) == 0:
        print("No data to save.")
        return False
    if os.path.exists(path) and not override:
        print(f"It seems a chroma database already exists at {path}.")
        return False
    data_element = data[0]
    printer = lambda: print(f"Saved {len(data)} documents to {path}.")
    if isinstance(data_element, Document):
        Chroma.from_documents(data, embedding, persist_directory=path)
        printer()
        return True
    if isinstance(data_element, str):
        Chroma.from_texts(data, embedding, persist_directory=path)
        printer()
        return True
    return False


if __name__ == "__main__":
    from loader import load_pdf_lazily

    # from pypdf import PdfReader
    from splitter import recursive_splitter, split_documents
    from dotenv import load_dotenv

    path = "data/s3-api.pdf"
    ## Reading to Document
    docs = load_pdf_lazily(path, max_nb_docs=4000, write_stdout=False)

    # ## Reading to str
    # reader = PdfReader(path)
    # docs = [reader.pages[i].extract_text() for i in range(10)]

    text_splitter = recursive_splitter(chunk_size=1000, chunk_overlap=200)
    documents = split_documents(docs, text_splitter)

    ## Loading OPENAI_API_KEY and OpenAI embeddings
    load_dotenv()
    embedding = OpenAIEmbeddings()
    create_chroma_database(documents, embedding, True, CHROMA_PATH)

    ## Reading the persisted Chroma database
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
    retriever = vectordb.as_retriever()

    ## Retrieving most relevant answers using stored vector embeddings
    response = retriever.invoke("What request to use to send a DELETE HTTPS request ?")
    for doc in response:
        print(doc.page_content)
        print("=======================================================\n")
