import json
import os
import chromadb
import dotenv
from astrapy import DataAPIClient
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from typing import List, Dict, Any

# Load environment variables
dotenv.load_dotenv()


class VectorStoreManager:
    """
    Manages connections and operations for both Astra DB and Chroma DB
    vector stores, including document downloading, embedding, and querying.
    """

    def __init__(self):
        """
        Initializes the database clients, collections, and the embedding model.
        """
        # --- Environment Variables ---
        self.ASTRA_DB_APPLICATION_TOKEN = os.environ.get('ASTRA_DB_APPLICATION_TOKEN')
        self.ASTRA_DB_API_ENDPOINT = os.environ.get('ASTRA_DB_API_ENDPOINT')
        self.keyspace = os.environ.get('ASTRA_DB_NAMESPACE')
        self.NVIDIA_KEY = os.environ.get("NVIDIA_KEY")

        # Basic check for essential environment variables
        if not all([self.ASTRA_DB_APPLICATION_TOKEN, self.ASTRA_DB_API_ENDPOINT, self.keyspace, self.NVIDIA_KEY]):
            raise ValueError(
                "Missing one or more required environment variables (ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_API_ENDPOINT, ASTRA_DB_NAMESPACE, NVIDIA_KEY).")

        # 2. Astra DB Client/Collection (Source DB)
        self._init_astra_db()

        # 3. Chroma DB Client/Collection (Local DB)
        self._init_chroma_db()

    def _init_astra_db(self):
        """Initializes the Astra DB client and collection."""
        my_client = DataAPIClient()
        my_database = my_client.get_database(
            self.ASTRA_DB_API_ENDPOINT,
            token=self.ASTRA_DB_APPLICATION_TOKEN,
        )
        # Astra DB collection name is 'food_demo'
        self.astra_collection = my_database.get_collection('food_demo')
        print("✅ Astra DB connection established.")

    def _init_chroma_db(self):
        """Initializes the Chroma DB persistent client and collection."""
        # Persistent storage directory
        chroma_path = "./chroma_food_storage_data"
        self.chroma_client = chromadb.PersistentClient(chroma_path)
        # Chroma DB collection name is 'food_demo_local'
        self.chroma_collection = self.chroma_client.get_or_create_collection(name="food_demo_local")
        print(f"✅ Chroma DB connection established. Storage path: {chroma_path}")

    @staticmethod
    def _clean_content(text: str) -> str:
        """Helper method to clean and standardize document content."""
        text = text.replace("\n", "; ").replace("  ", " ").strip()
        return text

    def download_and_populate(self, batch_size: int = 100):
        """
        Downloads documents from Astra DB, generates embeddings,
        and populates the Chroma DB vector store.
        """
        print("Starting document download and population...")
        try:
            # 1. Fetch all documents from Astra DB
            cursor = self.astra_collection.find({})
            astra_docs = list(cursor)
            total_docs = len(astra_docs)

            if total_docs == 0:
                print("⚠️ No documents found in Astra DB collection.")
                return

            print(f"Found {total_docs} documents in Astra DB. Processing in batches of {batch_size}...")

            # 2. Process and insert documents in batches
            for i in range(0, total_docs, batch_size):
                batch: List[Dict[str, Any]] = astra_docs[i:i + batch_size]

                # Prepare data for Chroma
                docs = [self._clean_content(d["content"]) for d in batch]
                metas = [d.get("metadata", {}) for d in batch]
                ids = [str(d["_id"]) for d in batch]  # Ensure IDs are strings

                # Insert into Chroma
                self.chroma_collection.add(
                    documents=docs,
                    metadatas=metas,
                    ids=ids,
                )

                print(f"✅ Inserted batch {(i // batch_size) + 1} / {((total_docs - 1) // batch_size) + 1}")

            final_count = self.chroma_collection.count()
            print(f"✨ All documents successfully downloaded and populated to Chroma DB. Total count: {final_count}")

        except Exception as e:
            print(f"❌ An error occurred during download_and_populate: {e}")

    def query_offline(self, query: str, n_results: int = 1) -> List[Document]:
        """
        Queries the local Chroma DB using the specified query,
        generates an embedding, and returns results as LangChain Documents.

        :param query: The natural language query string.
        :param n_results: The number of top results to return.
        :return: A list of LangChain Document objects.
        """
        if self.chroma_collection.count() == 0:
            self.download_and_populate()
        try:
            # query_vector = self.embeddings.embed_query(query)

            # Query Chroma
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )

            # Prepare results as a list of LangChain Document objects
            documents: List[Document] = []
            if not results or not results.get('documents'):
                return documents

            # Results are nested lists, so we use the first element [0]
            document_contents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]


            for content, metadata, distance in zip(document_contents, metadatas, distances):
                # Add distance to metadata for inspection
                doc_metadata = metadata if metadata else {}
                doc_metadata['chroma_distance'] = distance

                doc = Document(
                    page_content=content,
                    metadata=doc_metadata
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"❌ An error occurred during query_offline: {e}")
            return []


# --- Example Usage ---
if __name__ == "__main__":
    try:
        # 1. Instantiate the Manager
        manager = VectorStoreManager()

        # 2. Download and populate (Run this once to sync data)
        # manager.download_and_populate(batch_size=50) # Uncomment to run sync

        # 3. Query the local Chroma DB
        search_query = "how to store apples??"
        results = manager.query_offline(search_query, n_results=1)

        # 4. Print Results
        print("\n--- Query Results ---")
        if results:
            for i, doc in enumerate(results):
                print(f"\nResult {i + 1}: (Distance: {doc.metadata.get('chroma_distance'):.4f})")
                print(f"Content: {doc.page_content[:150]}...")
                print(f"Metadata: {doc.metadata}")
        else:
            print("No documents were retrieved.")
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    # print(not os.path.isdir('chroma_food_storage_data'))