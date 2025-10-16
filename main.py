import getpass
import os
from langchain_astradb import AstraDBVectorStore
from langchain import hub
from langchain_core.documents import Document
# Renamed for clarity: langgraph.graph.END is more common
from langgraph.graph import START, END, StateGraph
from typing import List, TypedDict, Literal
import dotenv
from langchain.chat_models import init_chat_model
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from offline_data_fetch_store import VectorStoreManager

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


class RAG_Fetcher:
    def __init__(self):
        # Initializations
        self.llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        self.prompt = hub.pull("rlm/rag-prompt")
        self.embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=os.environ["NVIDIA_KEY"],
            truncate="NONE",
        )
        # Astra Vector Store (Used for ONLINE retrieval)
        self.astra_vector_store = AstraDBVectorStore(
            embedding=self.embeddings,
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
        print("-> Running ONLINE Retrieval (AstraDB) 📡")
        retrieved_docs = self.astra_vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    # Node for retrieving from Chroma DB (OFFLINE)
    def retrieve_offline(self, state: State):
        """Retrieves documents from the local ChromaDB (OFFLINE)."""
        print("-> Running OFFLINE Retrieval (ChromaDB) 💾")
        # Use the manager's query method, specifying the mode for Chroma
        # NOTE: We assume the manager's internal logic handles population check.
        # Alternatively, use manager.query_offline directly after a manual check.

        # Using manager.query_offline directly (assuming it's the Chroma query)
        context = self.manager.query_offline(state['question'])
        return {'context': context}

    # ----------------------------------------------------
    # 3. Decision Node
    # ----------------------------------------------------
    def decide_offline_online(self, state: State):
        """Decides which retrieval path to take based on the 'offline' flag."""
        if state['offline']:
            return 'retrieve_offline'
        else:
            return 'retrieve_online'

    # ----------------------------------------------------
    # 4. Generation Node (Shared)
    # ----------------------------------------------------
    def generate(self, state: State):
        """Generates the final answer using the retrieved context."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        if not docs_content:
            return {"answer": "I couldn't find relevant information in the knowledge base (either online or offline)."}

        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    # ----------------------------------------------------
    # 5. Graph Building
    # ----------------------------------------------------
    def _build_graph(self):
        """Initializes and compiles the LangGraph state machine."""
        graph_builder = StateGraph(State)

        # 1. Add Nodes (Only retrieval and generation nodes are needed)
        graph_builder.add_node("retrieve_offline", self.retrieve_offline)  # Chroma retrieval
        graph_builder.add_node("retrieve_online", self.retrieve_online)  # Astra retrieval
        graph_builder.add_node("generate_answer", self.generate)  # LLM Generation

        # 2. Define Conditional Edges (The Core Fix)
        # The graph starts, executes self.decide_offline_online, and then
        # routes the flow to the node name (string) returned by that function.
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

        # 3. Connect Retrieval to Generation
        graph_builder.add_edge('retrieve_offline', 'generate_answer')
        graph_builder.add_edge('retrieve_online', 'generate_answer')

        # 4. Connect Generation to END
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

        response = self.graph.invoke(initial_state)
        return response["answer"]


if __name__ == '__main__':
    rg = RAG_Fetcher()

    question = "How to store poha?"

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