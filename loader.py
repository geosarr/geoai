from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from tqdm.auto import tqdm


def load_pdf(path: str):
    return PyPDFLoader(path).load()


def lazy_load_pdf(path: str):
    return PyPDFLoader(path).lazy_load()


def load_pdf_lazily(path: str, max_nb_docs: int = 100, write_stdout: bool = False):
    lazy_docs = lazy_load_pdf(path)
    documents = [Document("")] * max_nb_docs
    for page, doc in tqdm(
        enumerate(lazy_docs),
        total=max_nb_docs,
        desc=f"Loading at most {max_nb_docs} pages from {path}",
    ):
        if page >= max_nb_docs:
            break
        documents[page] = doc
        if write_stdout:
            tqdm.write(doc.page_content)
            tqdm.write("=======================================================\n")
    return documents


async def async_load_pdf(path: str):
    return await PyPDFLoader(path).aload()


async def async_lazy_load_pdf(path: str):
    return await PyPDFLoader(path).alazy_load()


if __name__ == "__main__":
    path = "data/s3-api.pdf"
    docs = lazy_load_pdf(path)
    print(next(docs).page_content)
