import json
import os
import chromadb
import dotenv
from astrapy import DataAPIClient
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

dotenv.load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ['ASTRA_DB_APPLICATION_TOKEN']
ASTRA_DB_API_ENDPOINT = os.environ['ASTRA_DB_API_ENDPOINT']
keyspace = os.environ['ASTRA_DB_NAMESPACE']

my_client = DataAPIClient()
my_database = my_client.get_database(
    ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=os.environ["NVIDIA_KEY"],
    truncate="NONE",
)

#Astra db
my_collection = my_database.get_collection('food_demo', keyspace=keyspace)

#chroma db
chroma_client = chromadb.PersistentClient("./chroma_food_storage_data")
collection = chroma_client.get_or_create_collection(name="food_demo_local")

def clean_content(text: str) -> str:
    text = text.replace("\n", "; ").replace("  ", " ").strip()
    return text

def download_content(batch_size = 100):
    try:
        cursor = my_collection.find(
            {},
        )
        astra_docs = list(cursor)
        for i in range(0, len(astra_docs), batch_size):
            batch = astra_docs[i:i + batch_size]

            # Clean up the text
            docs = [clean_content(d["content"]) for d in batch]
            metas = [d.get("metadata", {}) for d in batch]
            ids = [d["_id"] for d in batch]

            # Generate embeddings using NVIDIA model
            vectors = embeddings.embed_documents(docs)

            # Insert into Chroma
            collection.add(
                documents=docs,
                metadatas=metas,
                ids=ids,
                embeddings=vectors
            )

            print(f"✅ Inserted batch {i // batch_size + 1}")
    except Exception as e:
        print(e)


def query_offline(query: str) -> list[Document]:
    results = collection.query(
        query_embeddings=[embeddings.embed_query(query)],
        n_results=1,
        include=['documents', 'metadatas', 'distances']
    )
    document_contents = results.get('documents')[0]
    metadatas = results.get('metadatas')[0]

    # 2. Convert the raw data into a list of Document objects
    documents = []
    for content, metadata in zip(document_contents, metadatas):
        doc = Document(
            page_content=content,
            metadata=metadata if metadata else {}  # Use an empty dict if metadata is None
        )
        documents.append(doc)

    return documents


if __name__=='__main__':
    print(query_offline('How to store apple'))