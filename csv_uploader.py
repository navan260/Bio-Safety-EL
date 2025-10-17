import os
import csv
from dotenv import load_dotenv
from typing import List, Optional

# LangChain Imports
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from langchain_core.embeddings import Embeddings

# NEW: Import HuggingFaceEmbeddings instead of OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader

# Load environment variables from .env file
load_dotenv()


class AstraDBCSVUploader:
    """
    A class to handle connecting to Astra DB and uploading data from a CSV file
    using the LangChain CSVLoader and AstraDBVectorStore.
    """

    def __init__(
            self,
            collection_name: str,
            embedding_model: Optional[Embeddings] = None,
            astra_db_api_endpoint: Optional[str] = None,
            astra_db_token: Optional[str] = None,
            astra_db_keyspace: Optional[str] = None,
    ):

        # 1. Get connection parameters
        self.api_endpoint = astra_db_api_endpoint or os.environ.get("ASTRA_DB_API_ENDPOINT")
        self.token = astra_db_token or os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
        self.keyspace = astra_db_keyspace or os.environ.get("ASTRA_DB_KEYSPACE")
        self.collection_name = collection_name

        # 2. Initialize Embeddings Model
        # This will default to 'sentence-transformers/all-MiniLM-L6-v2'
        default_hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embeddings = embedding_model or default_hf_embeddings

        if not all([self.api_endpoint, self.token]):
            raise ValueError(
                "Astra DB API endpoint and token must be provided or set in environment variables."
            )

        # 3. Initialize AstraDBVectorStore
        self.vector_store = AstraDBVectorStore(
            embedding=self.embeddings,  # Passes the HuggingFace model
            collection_name=self.collection_name,
            api_endpoint=self.api_endpoint,
            token=self.token,
            namespace=self.keyspace,
        )
        print(f"AstraDBVectorStore initialized for collection: '{self.collection_name}'")
        print(f"Using Embedding Model: {self.embeddings.model_name}")

    # The rest of the methods (upload_documents, load_and_upload_csv) remain the same.

    def load_and_upload_csv(
            self,
            file_path: str,
            encoding: str = 'utf-8',
            csv_args: Optional[dict] = None
    ) -> List[str]:
        # ... (Implementation is the same as before)
        print(f"Loading CSV from: {file_path}...")
        try:
            loader = CSVLoader(
                file_path=file_path,
                encoding=encoding,
                csv_args=csv_args
            )
            documents = loader.load()

        except FileNotFoundError:
            raise FileNotFoundError(f"Error: CSV file not found at path: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading CSV file: {e}")

        print(f"Successfully loaded {len(documents)} documents from CSV.")

        # 2. Upload documents to Astra DB Vector Store
        print("Starting embedding and upload to Astra DB...")
        inserted_ids = self.vector_store.add_documents(documents)

        print(f"Successfully uploaded {len(inserted_ids)} documents to Astra DB collection: '{self.collection_name}'.")
        return inserted_ids


# --- Example Usage (will now use all-MiniLM-L6-v2) ---
if __name__ == "__main__":


    # Instantiate the uploader - it will now use HuggingFaceEmbeddings by default
    try:
        csv_file_path = 'data.csv'
        uploader = AstraDBCSVUploader(
            collection_name="food_demo"
        )

        # Load and upload the CSV data
        inserted_ids = uploader.load_and_upload_csv(csv_file_path)

        # Example: Show retrieval result
        print("\nPerforming a similarity search...")
        query_results = uploader.vector_store.similarity_search(
            "how to store poha?",
            k=1
        )

    except Exception as e:
        print(f"A fatal error occurred: {e}")