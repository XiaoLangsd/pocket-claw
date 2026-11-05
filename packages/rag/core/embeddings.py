from langchain_openai import OpenAIEmbeddings
from packages.common.config import settings

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=settings.embedding.api_key,
        base_url=settings.embedding.base_url,
        model=settings.embedding.model,
    )