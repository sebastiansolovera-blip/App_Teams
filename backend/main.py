import uvicorn
import os
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Request, Response
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from contextlib import asynccontextmanager

from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext, ActivityHandler, MessageFactory
from botbuilder.schema import Activity

# --- Rutas absolutas para Render ---
BASE_DIR = os.path.dirname(__file__)  # apunta a /backend
DATA_PATH = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.environ.get("CHROMA_PATH", os.path.join(BASE_DIR, "chroma_db"))

# --- Globals ---
rag_chain = None
vectorstore = None

# --- Request body ---
class QueryRequest(BaseModel):
    query: str

# --- Document Loading ---
def load_documents(data_path: str):
    documents = []
    if not os.path.exists(data_path):
        print(f"Error: Data directory not found at {data_path}")
        return []

    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Attempting to load file: {file_path}")
            try:
                if file.endswith(".txt"):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                elif file.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                else:
                    print(f"Skipping unsupported file type: {file_path}")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    if not documents:
        print("No documents loaded from data folder.")
        return []

    print(f"Loaded {len(documents)} raw documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

# --- RAG Setup ---
def setup_rag_pipeline(llm, vectorstore):
    global rag_chain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    print("RAG pipeline setup complete.")

# --- Teams Bot ---
class RagBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        global rag_chain
        user_query = turn_context.activity.text

        if rag_chain is None:
            await turn_context.send_activity(MessageFactory.text("RAG pipeline is not initialized. Please run /indexar-documentos."))
            return

        print(f"Received query from Teams: '{user_query}'")
        try:
            response = rag_chain({"question": user_query, "chat_history": []})
            rag_response = response.get('answer', 'Could not retrieve a specific answer.')
            await turn_context.send_activity(MessageFactory.text(rag_response))
            print("Sent response to Teams.")
        except Exception as e:
            print(f"Error during RAG query from Teams: {e}")
            await turn_context.send_activity(MessageFactory.text(f"An error occurred: {e}"))

# --- FastAPI App ---
load_dotenv()
MICROSOFT_APP_ID = os.getenv("MicrosoftAppId")
MICROSOFT_APP_PASSWORD = os.getenv("MicrosoftAppPassword")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not all([MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD, GOOGLE_API_KEY]):
    print("WARNING: Missing one or more environment variables.")

SETTINGS = BotFrameworkAdapterSettings(
    app_id=MICROSOFT_APP_ID,
    app_password=MICROSOFT_APP_PASSWORD
)
ADAPTER = BotFrameworkAdapter(SETTINGS)
BOT = RagBot()

# Lifespan (lightweight para Render)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Startup OK (light). RAG pipeline will be initialized solo on demand.")
    yield
    print("Shutdown complete.")

app = FastAPI(lifespan=lifespan)
router = APIRouter()

# --- Indexación de documentos ---
@router.post("/indexar-documentos")
async def index_documents():
    global vectorstore, rag_chain
    print("Starting document indexation process...")
    try:
        document_chunks = load_documents(DATA_PATH)
        if not document_chunks:
            return {"status": "error", "message": "No documents found to index."}

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = Chroma.from_documents(
            documents=document_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        print("Indexation complete. RAG pipeline is ready.")
        setup_rag_pipeline(llm, vectorstore)

        return {"status": "success", "message": "Indexation completed. The RAG pipeline is now ready to handle queries."}
    except Exception as e:
        print(f"Error during indexation: {e}")
        return {"status": "error", "message": f"An error occurred: {e}"}

# --- Endpoint Teams ---
@router.post("/api/messages")
async def messages(req: Request):
    if "application/json" not in req.headers.get("Content-Type", ""):
        return Response(status_code=415)
    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")
    await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
    return Response(status_code=200)

# --- Healthcheck ---
@router.get("/healthz")
async def healthz():
    return {"ok": True}

# --- Include router ---
app.include_router(router)

# --- Render entrypoint ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)
