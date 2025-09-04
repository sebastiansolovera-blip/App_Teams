import os
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Request, Response # Keep Request and Response for Teams bot if needed
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain # Keep for conversational RAG
from langchain_huggingface import HuggingFaceEmbeddings # Keep for embeddings
from langchain_community.vectorstores import Chroma # Keep for Chroma vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI # Keep for LLM
# Remove document loading related imports as indexing is done locally
# from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from contextlib import asynccontextmanager
from langchain.memory import ConversationBufferMemory # Keep for conversational memory
# Remove prompt components if not explicitly needed by ConversationalRetrievalChain.from_llm
# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Define the path to the data folder - No longer needed for loading, but keep for reference if needed
# DATA_PATH = "data" # Not used for loading in this version

# Define the local directory for the Chroma index
# CORRECTED: Use relative path assuming the script is run from the backend directory
CHROMA_PERSIST_DIR = "chroma_db_local" # Adjusted path assuming it's relative to backend directory

# Global variable to hold the RAG chain and memory
rag_chain = None
memory = None # Global variable to hold the memory
vectorstore = None # Keep vectorstore as a global to access it later if needed

# Define the request body model - Keep as is
class QueryRequest(BaseModel):
    query: str

# --- Document Loading and Processing Function ---
# REMOVED: load_documents function is no longer needed in main.py

# --- RAG Pipeline Setup Function (Loads from Persistent Index) ---
def setup_rag_pipeline_from_local_index():
    """
    Sets up the RAG pipeline by loading the vector store from a persistent local index.

    Returns:
        A tuple containing the initialized Chroma vector database, LLM, and Memory.
        Returns None, None, None if the index cannot be loaded.
    """
    print(f"Attempting to load Chroma index from: {CHROMA_PERSIST_DIR}")
    # Ensure the Chroma index directory exists
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Error: Chroma index directory not found at {CHROMA_PERSIST_DIR}. Please run the local indexing script first.")
        return None, None, None, None # Return prompt template placeholder as well

    try:
        # 1. Initialize an embedding model - Must be the same as used for indexing
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. Load the Chroma vector database from the persistent directory
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        print("Chroma vector store loaded successfully from local index.")

        # 3. Initialize the chosen LLM - Must be the same as used for querying
        load_dotenv() # Ensure environment variables are loaded
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        print("LLM initialized.")

        # 4. Initialize ConversationBufferMemory - Keep for conversational RAG
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        print("Conversation memory initialized.")

         # Define a prompt template for the RAG chain - Can keep or remove based on need
        # Note: ConversationalRetrievalChain.from_llm doesn't directly take qa_prompt,
        # but the base chain might use a default or you can construct the chain manually.
        # For simplicity, we'll omit passing a custom prompt template here.
        # If needed, you'd construct the chain like:
        # chain = (
        #     {"context": retriever, "question": RunnablePassthrough()}
        #     | prompt
        #     | llm
        # )
        # For from_llm, the prompt handling is often internal or requires a different constructor.
        # Let's just return None for the prompt template as it's not used by from_llm
        qa_prompt = None


        return vectorstore, llm, memory, qa_prompt # Return None for prompt

    except Exception as e:
        print(f"Error loading RAG components from local index: {e}")
        return None, None, None, None # Return None for all if error occurs

# --- FastAPI Application and Endpoint ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Loads the RAG chain from the persistent local index on startup.
    """
    global rag_chain, memory # Access global variables
    print("Loading RAG pipeline from local index...")

    # Attempt to load the RAG components from the local index
    vectorstore, llm, memory, qa_prompt = setup_rag_pipeline_from_local_index()

    if vectorstore and llm and memory: # Check if all components were loaded successfully
        try:
            # Initialize the ConversationalRetrievalChain using the loaded components
            rag_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory, # Pass the memory object
                return_source_documents=True, # Explicitly request source documents
                output_key="answer", # Explicitly set the output key for memory
                # qa_prompt=qa_prompt # REMOVED: This parameter is not accepted by from_llm
            )
            print("RAG pipeline loaded from local index and initialized.")
        except Exception as e:
            print(f"Error initializing RAG chain from loaded components: {e}")
            rag_chain = None
            memory = None # Ensure memory is also None if chain initialization fails
    else:
        print("RAG components could not be loaded from local index. RAG chain not initialized.")
        rag_chain = None # Ensure rag_chain is None if components couldn't be loaded
        memory = None # Ensure memory is None

    yield # The application runs here

    print("Shutting down...")
    # No specific cleanup needed for in-memory components in this basic example
    pass

# Instantiate FastAPI app and router, passing the lifespan
app = FastAPI(lifespan=lifespan)

# Define the allowed origins for CORS.
# UPDATE THIS WITH THE EXACT URL OF YOUR SITE!
origins = [
    "https://sebastiansolovera-blip.github.io", # Added the base domain of GitHub Pages
    "https://sebastiansolovera-blip.github.io/App_Teams", # Added the specific URL of your site
    # If you need to test locally with the frontend:
    # "http://localhost",
    # "http://localhost:8080", # Ejemplo de puerto local, ajústalo si es necesario
    # "http://127.0.0.1",
    # "http://127.0.0.1:8080", # Ejemplo de puerto local
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)

# Note: We are keeping the router for potential future use or other endpoints
# but the main RAG logic will now be initialized in the lifespan.
router = APIRouter()

# --- Add a root endpoint ---
@router.get("/")
async def read_root():
    return {"message": "Knowledge Management Backend is running"}

# --- Removed the /indexar-documentos endpoint ---
# Indexing is now done locally using index_local.py
# If you still want an index endpoint for re-indexing, you would modify this
# to trigger the indexing process, but be mindful of memory.

# --- Endpoint Teams (Condicional) ---
# Assuming the Teams bot is still desired for Render deployment
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

if TEAMS_BOT_AVAILABLE:
    class RagBot(ActivityHandler):
        async def on_message_activity(self, turn_context: TurnContext):
            global rag_chain
            user_query = turn_context.activity.text

            if rag_chain is None:
                # Updated message since /indexar-documentos is removed
                await turn_context.send_activity(MessageFactory.text("RAG pipeline is not initialized. The knowledge base might not be available."))
                return

            print(f"Received query from Teams: '{user_query}'")
            try:
                # ConversationalRetrievalChain requires chat_history
                response = await rag_chain.ainvoke({"question": user_query, "chat_history": []}) # Use memory object directly if memory is part of chain
                rag_response = response.get('answer', 'Could not retrieve a specific answer.')
                await turn_context.send_activity(MessageFactory.text(rag_response))
                print("Sent response to Teams.")
            except Exception as e:
                print(f"Error during RAG query from Teams: {e}")
                await turn_context.send_activity(MessageFactory.text(f"An error occurred: {e}"))

    load_dotenv() # Ensure env vars are loaded for bot settings
    MICROSOFT_APP_ID = os.getenv("MicrosoftAppId")
    MICROSOFT_APP_PASSWORD = os.getenv("MicrosoftAppPassword")

    if MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD:
        print("Initializing Teams Bot Adapter...")
        SETTINGS = BotFrameworkAdapterSettings(
            app_id=MICROSOFT_APP_ID,
            app_password=MICROSOFT_APP_PASSWORD
        )
        ADAPTER = BotFrameworkAdapter(SETTINGS)
        # BOT = RagBot() # Instantiate BOT here if TEAMS_BOT_ENABLED is True
        TEAMS_BOT_ENABLED = True

        @router.post("/api/messages")
        async def messages(req: Request):
            if "application/json" not in req.headers.get("Content-Type", ""):
                return Response(status_code=415)
            body = await req.json()
            activity = Activity().deserialize(body)
            auth_header = req.headers.get("Authorization", "")

            # Check if the bot is initialized (BOT will be None if TEAMS_BOT_ENABLED is False)
            if BOT is None:
                 print("Teams Bot not initialized.")
                 return Response(status_code=503, content="Teams Bot is not initialized.")

            # Pass the conversation history from the global memory if memory is available
            # ConversationalRetrievalChain handles chat_history internally when initialized with memory=memory
            # So we just need to pass the activity to the bot's on_turn
            await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
            return Response(status_code=200)
    else:
        print("Microsoft App ID or Password not found. Teams bot endpoint will not be active.")
        ADAPTER = None
        # BOT = None
        TEAMS_BOT_ENABLED = False
else:
    print("Teams bot functionality is disabled.")
    ADAPTER = None
    # BOT = None
    TEAMS_BOT_ENABLED = False

# Instantiate BOT here if TEAMS_BOT_AVAILABLE and TEAMS_BOT_ENABLED are True
# Moved instantiation outside conditional endpoint definition
if TEAMS_BOT_AVAILABLE and TEAMS_BOT_ENABLED:
     BOT = RagBot()
     print("Teams Bot instantiated.")
else:
     BOT = None # Ensure BOT is explicitly None if not enabled/available


# --- Endpoint RAG for Frontend ---
@router.post("/query")
async def query_rag(request: QueryRequest):
    """
    Endpoint to query the RAG pipeline.
    """
    # Check if rag_chain and memory are initialized
    if rag_chain is None or memory is None:
        # Updated message since indexation is done on startup now
        return {"response": "RAG pipeline is not initialized. An error might have occurred during startup, or the knowledge base is not available.", "sources": []}


    user_query = request.query

    try:
        # The ConversationalRetrievalChain with memory automatically manages chat history.
        # We just need to pass the current question and the memory's chat history.
        # The chain expects {"question": ..., "chat_history": ...}
        # The memory object attached to the chain automatically populates "chat_history"
        # So we only need to pass the "question".
        response = await rag_chain.ainvoke({"question": user_query})


        rag_response = response.get('answer', 'Could not retrieve a specific answer.')
        # Ensure source_documents is a list before iterating
        source_documents = response.get('source_documents', []) if isinstance(response.get('source_documents'), list) else []


        sources_list = []
        # Ensure doc is a document object before accessing attributes
        for doc in source_documents:
             sources_list.append({
                'page_content': getattr(doc, 'page_content', 'N/A'),
                'metadata': getattr(doc, 'metadata', {})
            })

        return {"response": rag_response, "sources": sources_list} # Return sources to frontend
    except Exception as e:
        print(f"Error during RAG query: {e}")
        return {"response": f"An error occurred during the query: {e}", "sources": []}


# --- Healthcheck ---
@router.get("/healthz")
async def healthz():
    # Indicate health and RAG initialization status
    status = "OK" if rag_chain is not None else "Initializing RAG"
    return {"status": status, "rag_initialized": rag_chain is not None}


# --- Include router ---
app.include_router(router)

# --- Render entrypoint ---
# This block is typically for local execution or specific deployment setups
# On Render, the web server (like uvicorn) is configured separately
# if __name__ == "__main__":
#     import os
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     print(f"Running Uvicorn locally on http://0.0.0.0:{port}")
#     uvicorn.run("main:app", host="0.0.0.0", port=port)
