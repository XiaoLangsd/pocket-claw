import uuid
from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from packages.rag.config import chunk_size, chunk_overlap, chroma_path
from packages.rag.core.embeddings import get_embeddings
from langchain_chroma import Chroma

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap,
    separators=["\n\n", "\n", "。","！", "？", "：", "，", " ", ""]
)

def get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext in (".txt", ".md"):
        return TextLoader(file_path, autodetect_encoding=True)
    if ext == ".csv":
        return CSVLoader(file_path, csv_args={"encoding": "utf-8"})
    if ext == ".pdf":
        return PyMuPDFLoader(file_path)
    if ext == ".docx":
        # noinspection abstract-class
        return Docx2txtLoader(file_path)
    raise ValueError(f"不支持的文件格式: {ext}")

async def process_and_store(file_path: str, file_name: str):
    """加载文档 -> 切片 -> 向量化 -> 存入 ChromaDB"""
    file_id = str(uuid.uuid4())

    loader = get_loader(file_path)
    documents = loader.load()

    chunks = SPLITTER.split_documents(documents)
    # 给每个块添加元数据
    for chunk in chunks:
        chunk.metadata.update({
            "file_id": file_id,
            "file_name": file_name,
            "source": file_name,
        })

    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)

    return {
        "file_id": file_id,
        "file_name": file_name,
        "chunks": len(chunks),
        "status": "success",
    }