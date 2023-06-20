import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# openai / langchain Const
RETRIEVER_K_ARG = 3
OPENA_AI_MODEL = "gpt-4-0314"
PRE_PROMPT_INSTRUCTIONS = "Use the context to answer the prompt"
PERSIST_DIRECTORY = "db"
HUGGINGFACE_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_KWARGS = {"device": "cuda"}

# Google Const
CLIENT_SECRET_FILE = "credentials.json"
TOKEN_FILE = 'token.json'
GOOGLE_DRIVER_FOLDER_ID = "YOUR_GOOGLE_DRIVER_FOLDER_ID_HERE"

os.environ["OPENAI_API_KEY"] ="sk-YOUR_OPEN_AI_API_KEY"


def load_documents():
    loader = GoogleDriveLoader(
        credentials_path=CLIENT_SECRET_FILE,
        token_path=TOKEN_FILE,
        folder_id=GOOGLE_DRIVER_FOLDER_ID,
        recursive=False,
        file_types=["sheet", "document", "pdf"],
    )
    return loader.load()


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"])
    return text_splitter.split_documents(docs)


def generate_embeddings():
    return HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL, model_kwargs=MODEL_KWARGS)


def create_chroma_db(texts, embeddings):
    if not os.path.exists(PERSIST_DIRECTORY):
        return Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
    else:
        return Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)


def create_retriever(db):
    return db.as_retriever(search_kwargs={"k": RETRIEVER_K_ARG})


def create_index(llm, retriever):
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def create_llm():
    return ChatOpenAI(temperature=0, model_name=OPENA_AI_MODEL)


def main():
    docs = load_documents()
    texts = split_documents(docs)
    embeddings = generate_embeddings()
    db = create_chroma_db(texts, embeddings)
    retriever = create_retriever(db)
    llm = create_llm()
    qa = create_index(llm, retriever)

    while True:
        query = input("> ")
        if query.lower() == "exit":
            exit()
        answer = qa({"query": f"### Instructions. {PRE_PROMPT_INSTRUCTIONS} ###Prompt {query}"})
        print(answer['result'])


if __name__ == "__main__":
    main()
