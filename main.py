import getpass
import os
from langchain_astradb import AstraDBVectorStore
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import dotenv
from langchain.chat_models import init_chat_model
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from offline_data_fetch_store import query_offline

dotenv.load_dotenv()

embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=os.environ["NVIDIA_KEY"],
    truncate="NONE",
)

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

ASTRA_DB_API_ENDPOINT=os.environ["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_APPLICATION_TOKEN=os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_NAMESPACE=os.environ["ASTRA_DB_NAMESPACE"]


llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
vector_store = AstraDBVectorStore(
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    collection_name="food_demo",
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)

prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    print(retrieved_docs)
    print("Retrieval done")
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    print("Generation done")
    return {"answer": response.content}


def outer(state: State):
    context = query_offline(state['question'])
    return {'context': context}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([outer, generate])
graph_builder.add_edge(START, "outer")
graph = graph_builder.compile()
response = graph.invoke({"question": "What is the best preservation method for poha?"})
print(response["answer"])