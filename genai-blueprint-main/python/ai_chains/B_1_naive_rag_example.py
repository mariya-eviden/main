""" 
A simple RAG chain
"""

from pathlib import Path

from devtools import debug  # noqa: F401
from langchain_community.document_loaders.text import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from python.ai_core.chain_registry import Example, RunnableItem, register_runnable
from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.llm import get_llm
from python.ai_core.prompts import def_prompt
from python.ai_core.vector_store import VectorStoreFactory


def get_retriever(config: dict):
    debug(config)
    path = config.get("path")
    if path is None:
        raise ValueError("Config should have a 'path' key")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path in config does not exists: {path}")
    vector_store = VectorStoreFactory(
        id= None,
        collection_name="test_rag",
        embeddings_factory=EmbeddingsFactory(),  # take default one
    ).vector_store

    logger.info(f"indexing text document  {path} in VectorStore")
    loader = TextLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=splits)

    return vector_store.as_retriever(search_kwargs={"k": 3})


prompt_system = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    """
prompt_user = """
    Question: {question} 
    Context: {context} 

    Answer:
    """
prompt = def_prompt(system=prompt_system, user=prompt_user)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain(config: dict):
    chain = (
        {
            "context": get_retriever(config) | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain


register_runnable(
    RunnableItem(
        tag="RAG",
        name="Simple RAG chain",
        runnable=get_chain,
        examples=[
            Example(
                path=Path("use_case_data/maintenance/maintenance_procedure_1.txt"),
                query=[
                    "What are the tools required for the maintenance task 'Repair of Faulty Switchgear' "
                ],
            )
        ],
    )
)
