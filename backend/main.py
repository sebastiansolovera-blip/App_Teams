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
from langchain.text_splitter import RecursiveCharacterCharacterTextSplitter
from contextlib import asynccontextmanager

# Importa CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

# Importaciones condicionales para el bot de Teams
try:
    from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext, ActivityHandler, MessageFactory
    from botbuilder.schema import Activity
    TEAMS_BOT_AVAILABLE = True
except ImportError:
    print("BotBuilder libraries not found. Teams bot functionality will be disabled.")
    TEAMS_BOT_AVAILABLE = False
    BotFrameworkAdapter = None
    BotFrameworkAdapterSettings = None
    TurnContext = None
    ActivityHandler = object
    MessageFactory = None
    Activity = None


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
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    print("RAG pipeline setup complete.")

# --- Teams Bot (Condicional) ---
if TEAMS_BOT_AVAILABLE:
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
else:
    class RagBot:
        def __init__(self):
            print("Teams bot functionality is disabled.")
        async def on_turn(self, turn_context):
            pass


# --- FastAPI App ---
load_dotenv()
MICROSOFT_APP_ID = os.getenv("MicrosoftAppId")
MICROSOFT_APP_PASSWORD = os.getenv("MicrosoftAppPassword")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if TEAMS_BOT_AVAILABLE and MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD:
    print("Initializing Teams Bot Adapter...")
    SETTINGS = BotFrameworkAdapterSettings(
        app_id=MICROSOFT_APP_ID,
        app_password=MICROSOFT_APP_PASSWORD
    )
    ADAPTER = BotFrameworkAdapter(SETTINGS)
    BOT = RagBot()
    TEAMS_BOT_ENABLED = True
else:
    print("Microsoft App ID or Password not found. Teams bot endpoint will not be active.")
    ADAPTER = None
    BOT = None
    TEAMS_BOT_ENABLED = False


# Lifespan (lightweight para Render)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Startup OK (light). RAG pipeline will be initialized solo on demand.")
    yield
    print("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

# Define los orígenes permitidos para CORS.
# ¡ACTUALIZA ESTO CON LA URL EXACTA DE TU SITIO DE GITHUB PAGES!
origins = [
    "https://<tu-usuario-github>.github.io", # Reemplaza con tu usuario de GitHub
    "https://<tu-usuario-github>.github.io/<nombre-repo>", # Si es un sitio de proyecto
    # Si necesitas probar localmente con el frontend:
    # "http://localhost",
    # "http://localhost:8080", # Ejemplo de puerto local, ajústalo si es necesario
    # "http://127.0.0.1",
    # "http://127.0.0.1:8080", # Ejemplo de puerto local
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permite todos los métodos
    allow_headers=["*"], # Permite todos los encabezados
)


router = APIRouter()

# --- Add a root endpoint ---
@router.get("/")
async def read_root():
    return {"message": "Knowledge Management Backend is running"}


# --- Indexación de documentos ---
@router.post("/indexar-documentos")
async def index_documents():
    global vectorstore, rag_chain
    print("Starting document indexation process...")
    try:
        if not GOOGLE_API_KEY:
             return {"status": "error", "message": "GOOGLE_API_KEY environment variable not set."}

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

# --- Endpoint Teams (Condicional) ---
if TEAMS_BOT_ENABLED:
    @router.post("/api/messages")
    async def messages(req: Request):
        if "application/json" not in req.headers.get("Content-Type", ""):
            return Response(status_code=415)
        body = await req.json()
        activity = Activity().deserialize(body)
        auth_header = req.headers.get("Authorization", "")
        await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
        return Response(status_code=200)
else:
    pass

# --- Endpoint RAG para el Frontend ---
@router.post("/query")
async def query_rag(request: QueryRequest):
    """
    Endpoint to query the RAG pipeline.
    """
    if rag_chain is None:
        return {"response": "RAG pipeline is not initialized. Please call /indexar-documentos first.", "sources": []}

    user_query = request.query

    try:
        response = await rag_chain.ainvoke({"question": user_query, "chat_history": []})

        rag_response = response.get('answer', 'Could not retrieve a specific answer.')
        source_documents = response.get('source_documents', [])

        sources_list = []
        for doc in source_documents:
            sources_list.append({
                'page_content': getattr(doc, 'page_content', 'N/A'),
                'metadata': getattr(doc, 'metadata', {})
            })

        return {"response": rag_response, "sources": sources_list}
    except Exception as e:
        print(f"Error during RAG query: {e}")
        return {"response": f"An error occurred during the query: {e}", "sources": []}


# --- Healthcheck ---
@router.get("/healthz")
async def healthz():
    status = "OK" if rag_chain is not None else "Initializing RAG"
    return {"status": status, "rag_initialized": rag_chain is not None}


# --- Include router ---
app.include_router(router)

# --- Render entrypoint ---
if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Running Uvicorn locally on http://0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
