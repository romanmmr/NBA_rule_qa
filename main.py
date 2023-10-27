import os

from import_pdf import *
from utils import *


# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def build_qa():
    return print('Start Building Q&A system for NBA rules')


def get_embeddings(embbedings_model_name):
    print('Getting Embeddings')
    my_embeddings = SentenceTransformerEmbeddings(model_name=embbedings_model_name)
    print('Embeddings completed')
    return my_embeddings


def get_tokenizer(model_name):
    print('Getting Tokenizer')
    my_tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('Tokenizer completed')
    return my_tokenizer


def get_model(model_name):
    print('Getting Model')
    my_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print('Model completed')
    return my_model


def make_pipeline(task, model, tokenizer, max_len):
    print('Generating Pipeline')
    my_pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_length=max_len
    )
    print('Pipeline completed')
    return my_pipe


def get_llm(pipe_line, temperature=0, max_length=1024):
    print('Getting Local llm')
    my_local_llm = HuggingFacePipeline(
        pipeline=pipe_line,
        model_kwargs={"temperature": temperature, "max_length": max_length}
    )
    print('Local llm completed')
    return my_local_llm


def load_pages(pdf_name):
    print('Loading pages')
    my_loader = PyPDFLoader(get_pdf_path(pdf_name))
    my_pages = my_loader.load()
    print('Loading pages completed')
    return my_pages


def make_splits(chunk_size=500, chunk_overlap=300):
    print('Getting splits')
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        separators=["\n", "(?<=\. )", " ", ""]
    )
    r_splits = rec_splitter.split_documents(pages)
    print('Splits completed')
    return r_splits


def make_vector_database(vdb_splits, vdb_embeddings):
    print('Creating vector database')
    persist_directory = 'docs/chroma/'
    if os.path.exists('docs/chroma/'):

        if len(os.listdir('docs/chroma/')) != 0:
            os.remove(os.path.join('docs/chroma/', os.listdir('docs/chroma/')[0]))

        vdb = Chroma.from_documents(
            documents=vdb_splits,
            embedding=vdb_embeddings,
            persist_directory=persist_directory
        )
    print('Vector database created')
    return vdb


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    embedding_model = 'all-MiniLM-L6-v2'
    model_id = 'google/flan-t5-large'
    build_qa()
    embeddings = get_embeddings(embedding_model)
    tokenizer = get_tokenizer(model_id)
    model = get_model(model_id)
    pipe = make_pipeline(
        task='text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_len=1024
    )
    local_llm = get_llm(
        pipe_line=pipe
    )
    pages = load_pages('Official-Playing-Rules-2022-23-NBA-Season.pdf')
    splits = make_splits(
        chunk_size=500,
        chunk_overlap=300
    )
    print(f'Number of splits: {len(splits)}')
    vectordb = make_vector_database(
        vdb_splits=splits,
        vdb_embeddings=embeddings
    )



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
