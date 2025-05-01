from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from tqdm.auto import tqdm
from pypdf import PdfReader


def load_pdf(path: str):
    return PyPDFLoader(path).load()


def lazy_load_pdf(path: str):
    return PyPDFLoader(path).lazy_load()


def load_pdf_lazily(path: str, max_nb_docs: int = 100, write_stdout: bool = False):
    total_nb_pages = len(PdfReader(path).pages)
    nb_pages = min(max_nb_docs, total_nb_pages)
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


async def async_load_pdf(path: str):
    return await PyPDFLoader(path).aload()


async def async_lazy_load_pdf(path: str):
    return await PyPDFLoader(path).alazy_load()


if __name__ == "__main__":
    path = "data/s3-api.pdf"
    docs = lazy_load_pdf(path)
    print(next(docs).page_content)
