import uvicorn
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
# If you still want an index endpoint for re-index...port)

