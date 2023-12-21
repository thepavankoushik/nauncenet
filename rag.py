from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import pandas as pd

tweets = pd.read_csv('data/train.csv', sep=',')
selected_columns = ['text', 'target']
twitter_df = tweets[selected_columns]

DOCUMENT="text"
TOPIC="target"



df_loader = DataFrameLoader(twitter_df, page_content_column=DOCUMENT)
df_document = df_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(df_document)


model_name = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(model_name=model_name)


persist_directory = 'db'
vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

