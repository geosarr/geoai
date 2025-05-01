import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from constants import CHROMA_PATH
from langchain_core.documents import Document


def create_chroma_database(
    documents: list[Document],
    embedding: OpenAIEmbeddings,
    override: bool = False,
) -> bool:
    if os.path.exists(CHROMA_PATH) and not override:
        print(f"A chroma database already exists at {CHROMA_PATH}.")
        return False
    Chroma.from_documents(documents, embedding, persist_directory=CHROMA_PATH)
    print(f"Saved {len(documents)} documents to {CHROMA_PATH}.")
    return True


if __name__ == "__main__":
    from loader import load_pdf_lazily
    from dotenv import load_dotenv

    path = "data/s3-api.pdf"
    documents = load_pdf_lazily(path, max_nb_docs=3497, write_stdout=False)

    load_dotenv()
    embedding = OpenAIEmbeddings()
    if create_chroma_database(documents, embedding, True):
        print("Chroma database created.")

    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
    retriever = vectordb.as_retriever()
    response = retriever.invoke("What request to use to send a DELETE HTTPS request ?")
    for doc in response:
        print(doc.page_content)
        print("=======================================================\n")
