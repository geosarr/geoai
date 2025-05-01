from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path: str):
    return PyPDFLoader(path).load()


def lazy_load_pdf(path: str):
    return PyPDFLoader(path).lazy_load()


async def async_load_pdf(path: str):
    return await PyPDFLoader(path).aload()


async def async_lazy_load_pdf(path: str):
    return await PyPDFLoader(path).alazy_load()


if __name__ == "__main__":
    path = "data/s3-api.pdf"
    docs = lazy_load_pdf(path)
    print(next(docs).page_content)
