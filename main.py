import getpass
import logging
import os
import subprocess
import time

from langchain_astradb import AstraDBVectorStore
from langchain import hub
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
# Renamed for clarity: langgraph.graph.END is more common
from langgraph.graph import START, END, StateGraph
from typing import List, TypedDict, Literal
import dotenv
from langchain.chat_models import init_chat_model
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from offline_data_fetch_store import VectorStoreManager
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()

# Your environment variable setup remains the same
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_NAMESPACE = os.environ["ASTRA_DB_NAMESPACE"]


# 1. State definition is correct, using 'offline: bool' for the decision
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    offline: bool

class RagFetcher:
    def __init__(self):
        self.OLLAMA_MODEL = "llama3"
        self.OLLAMA_URL = "http://localhost:11434"

        self.llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
        """

        self.prompt = PromptTemplate(input_variables=["question", "context"], template=template)
        self.ollama_llm = ChatOllama(
            model=self.OLLAMA_MODEL,
            base_url=self.OLLAMA_URL,
            # Ensure the model is pulled beforehand using 'ollama pull llama3'
        )
        self.hg_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Astra Vector Store (Used for ONLINE retrieval)
        self.astra_vector_store = AstraDBVectorStore(
            embedding=self.hg_embeddings,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            collection_name="food_demo",
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_NAMESPACE,
        )
        # Vector Store Manager (Handles OFFLINE/Chroma and population logic)
        self.manager = VectorStoreManager()

        # Build the Graph
        self._build_graph()

    # ----------------------------------------------------
    # 2. Retrieval Nodes (One for each source)
    # ----------------------------------------------------

    # Node for retrieving from Astra DB (ONLINE)
    def retrieve_online(self, state: State):
        """Retrieves documents from the remote AstraDB (ONLINE)."""
        logging.log(0, 'Retrieving')
        retrieved_docs = self.astra_vector_store.similarity_search(state["question"], k=1)
        return {"context": retrieved_docs}

    # Node for retrieving from Chroma DB (OFFLINE)
    def retrieve_offline(self, state: State):
        """Retrieves documents from the local ChromaDB (OFFLINE)."""
        context = self.manager.query_offline(state['question'])
        return {'context': context}

    # ----------------------------------------------------
    # 3. Decision Node
    # ----------------------------------------------------
    def decide_offline_online(self, state: State):
        """Decides which retrieval path to take based on the 'offline' flag."""
        print(f"[DEBUG] decide_offline_online called. offline={state['offline']}")
        if state['offline']:
            return 'retrieve_offline'
        else:
            return 'retrieve_online'

    # ----------------------------------------------------
    # 4. Generation Node
    # ----------------------------------------------------
    def generate(self, state: State):
        """Generates the final answer using the retrieved context."""
        print("[DEBUG] generate called")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        print(f"[DEBUG] Context length: {len(docs_content)}")
        if not docs_content:
            print("[DEBUG] No context found")
            return {"answer": "I couldn't find relevant information in the knowledge base (either online or offline)."}
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        if state['offline']:
            try:
                print(f"-> Running OFFLINE Generation (Ollama: {self.OLLAMA_MODEL}) 💻")
                response = self.ollama_llm.invoke(messages)
                response_content = response.content
            except Exception as e:
                response_content = (
                    f"LLM Generation Error (Ollama): Could not connect to the local server. "
                    f"Please run 'ollama serve' and pull the '{self.OLLAMA_MODEL}' model. Error: {e}"
                )

            return {"answer": response_content}
        else:
            print("-> Running ONLINE Generation (Gemini) 🌐")
            response = self.llm.invoke(messages)
            return {"answer": response.content}

    # ----------------------------------------------------
    # 5. Graph Building
    # ----------------------------------------------------
    def _build_graph(self):
        """Initializes and compiles the LangGraph state machine."""
        graph_builder = StateGraph(State)

        graph_builder.add_node("retrieve_offline", self.retrieve_offline)  # Chroma retrieval
        graph_builder.add_node("retrieve_online", self.retrieve_online)  # Astra retrieval
        graph_builder.add_node("generate_answer", self.generate)  # LLM Generation

        graph_builder.add_conditional_edges(
            # Source Node: START
            START,
            # Conditional Function: The function that returns the next node name
            self.decide_offline_online,
            # Mapping: Maps function output (string) to the next node
            {
                'retrieve_offline': 'retrieve_offline',
                'retrieve_online': 'retrieve_online',
            },
        )


        graph_builder.add_edge('retrieve_offline', 'generate_answer')
        graph_builder.add_edge('retrieve_online', 'generate_answer')

        graph_builder.add_edge('generate_answer', END)

        self.graph = graph_builder.compile()

    # ----------------------------------------------------
    # 6. Public Query Method
    # ----------------------------------------------------
    def query(self, query: str, mode: Literal['offline', 'online'] = 'offline'):
        """
        Public entry point to run the RAG chain.

        :param mode: 'offline' to use ChromaDB, 'online' to use AstraDB.
        """
        # Set the initial state with the query and the decision flag
        initial_state = {
            "question": query,
            "offline": (mode == 'offline'),  # Convert string 'offline' to boolean True/False
            "context": [],
            "answer": ""
        }
        print(f"[DEBUG] Starting graph execution with mode={mode}")
        try:
            response = self.graph.invoke(initial_state)
            print("[DEBUG] Graph execution finished")
            return response["answer"]
        except Exception as e:
            print(f"[ERROR] Graph execution failed: {e}")
            raise e


if __name__ == '__main__':
    rg = RagFetcher()

    question = "How to store poha? And how many days can I store it for? Also explain its sensitivity."

    # --- MODE 1: OFFLINE QUERY (CHROMA) ---
    print("\n--- Running OFFLINE Query (Chroma) ---")
    offline_answer = rg.query(question, mode='offline')
    print(f"\nQuestion: {question}")
    print(f"Offline Answer: {offline_answer}")

    # --- MODE 2: ONLINE QUERY (ASTRA) ---
    print("\n\n--- Running ONLINE Query (AstraDB) ---")
    online_answer = rg.query(question, mode='online')
    print(f"\nQuestion: {question}")
    print(f"Online Answer: {online_answer}")