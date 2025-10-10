import os
from langchain_astradb import AstraDBVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

dotenv.load_dotenv()

ASTRA_DB_API_ENDPOINT=os.environ["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_APPLICATION_TOKEN=os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_NAMESPACE=os.environ["ASTRA_DB_NAMESPACE"]

embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=os.environ["NVIDIA_KEY"],
    truncate="NONE",
)

vector_store = AstraDBVectorStore(
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    collection_name="food_demo",
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)


# Load and chunk contents of the blog
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()
# print(docs)

file_path = "data.csv"

# Load CSV — each row becomes a Document
loader = CSVLoader(file_path=file_path, encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)