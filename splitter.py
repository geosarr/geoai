from typing import Iterable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def recursive_splitter(
    chunk_size: int, chunk_overlap: int, add_start_index: bool = True
):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=add_start_index,
    )


def split_documents(
    documents: Iterable[Document], splitter: RecursiveCharacterTextSplitter
) -> Iterable[Document]:
    return splitter.split_documents(documents)


if __name__ == "__main__":
    from loader import lazy_load_pdf

    path = "data/s3-api.pdf"
    docs = lazy_load_pdf(path)
    documents = [next(docs) for _ in range(200)]
    text_splitter = recursive_splitter(chunk_size=1000, chunk_overlap=200)
    split_docs = split_documents(documents, text_splitter)
    for doc in split_docs:
        print(doc.page_content)
        print("=======================================================\n")
