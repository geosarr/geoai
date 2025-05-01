from constants import CHROMA_PATH
from embedding import get_retriever


try:
    RETRIEVER = get_retriever(CHROMA_PATH)
except Exception as e:
    raise ValueError(
        f"Error loading retriever: \n{e}\nmake sure the database is created and the path is correct."
    )


def create_prompt(request: str) -> str:
    more_relevant_docs = RETRIEVER.invoke(request)
    context = "\n===================================\n".join(
        map(lambda doc: doc.page_content, more_relevant_docs)
    )
    question = f"Answer the question: {request}"
    prompt = (
        f"""Use the texts below as input of your answer.\n\n{context}\n\n{question}"""
        if len(more_relevant_docs)
        else question  # Fallback to the question if no relevant docs are found
    )
    return prompt


if __name__ == "__main__":
    print(create_prompt("Which language is better for s3 programming ?"))
