# !pip install --upgrade langchain
# !pip install tiktoken
# !pip install docarray
# !pip install pypdf
# !pip install chromadb
# !pip install -U sentence-transformers

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.llms import HuggingFacePipeline
# import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

from langchain.chains import RetrievalQA

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.retrievers.document_compressors import EmbeddingsFilter

from langchain.prompts import PromptTemplate