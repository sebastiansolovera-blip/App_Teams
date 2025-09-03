import os
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Request, Response
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from contextlib import asynccontextmanager

from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext, ActivityHandler, MessageFactory
from botbuilder.schema import Activity, ActivityTypes

# Define the path to the data folder - ADJUST THIS BASED ON YOUR STRUCTURE IF NEEDED
# If your backend root in Render is /backend, DATA_PATH should be "./data"
# If your backend root in Render is /App_Teams/backend, DATA_PATH should be "./data"
# Let's keep it relative to the backend folder for simplicity in deployment root
DATA_PATH = "./data"

# Global variable to hold the RAG chain
rag_chain = None

# Define the request body model for the original /query endpoint (keeping for now)
class QueryRequest(BaseModel):
    query: str

# --- Document Loading and Processing Function ---
def load_documents(data_path: str):
    """
    Loads documents from a local folder, processes them, and splits them into chunks.

    Args:
        data_path: The path to the folder containing the documents.

    Returns:
        A list of document chunks.
    """
    documents = []
    print(f"Scanning directory: {data_path}")
    # Use the data_path relative to the script's execution location
    full_data_path = os.path.join(os.path.dirname(__file__), data_path)
    print(f"Scanning full path: {full_data_path}") # Added print for debugging

    if not os.path.exists(full_data_path):
        print(f"Error: Data directory not found at {full_data_path}")
        return []

    for root, _, files in os.walk(full_data_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Attempting to load file: {file_path}")
            try:
                if file.endswith(".txt"):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                    print(f"Successfully loaded .txt file: {file}")
                elif file.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
                    print(f"Successfully loaded .docx file: {file}")
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    print(f"Successfully loaded .pdf file: {file}")
                else:
                    print(f"Skipping unsupported file type: {file_path}")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    if not documents:
        print("No documents loaded from data folder.")
        return []

    print(f"Loaded {len(documents)} raw documents.")
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    return chunks

# --- RAG Pipeline Setup Function ---
def setup_rag_pipeline(document_chunks):
    """
    Sets up the RAG pipeline with embeddings, vector store, and LLM.

    Args:
        document_chunks: A list of document chunks.

    Returns:
        A tuple containing the initialized Chroma vector database and LLM.
    """
    # 1. Initialize an embedding model
    # Using a local Sentence-Transformer model for embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Initialize a Chroma vector database
    # Using document chunks and the embedding model to create the vector store
    # This will create an in-memory vector store for this PoC
    vectorstore = Chroma.from_documents(documents=document_chunks, embedding=embeddings)

    # 3. Initialize the chosen LLM
    # Ensure you have GOOGLE_API_KEY set in your .env file or environment variables
    # Replace with your preferred LLM if not using Google Generative AI

    # Load Google API Key from environment variables
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        print("GOOGLE_API_KEY loaded from environment variables.")
    else:
        print("WARNING: GOOGLE_API_KEY not found in environment variables.")
        print("Please ensure GOOGLE_API_KEY is set for the RAG pipeline to work.")


    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key) # Using gemini-1.5-flash and passing the key

    return vectorstore, llm

# --- Teams Bot Activity Handler ---
class RagBot(ActivityHandler):
    """
    Teams bot activity handler to process incoming messages and interact with the RAG chain.
    """
    async def on_message_activity(self, turn_context: TurnContext):
        """
        Process incoming message activities.
        """
        global rag_chain

        user_query = turn_context.activity.text

        if not user_query:
            await turn_context.send_activity(MessageFactory.text("Please enter a query."))
            return

        if rag_chain is None:
            await turn_context.send_activity(MessageFactory.text("RAG pipeline is not initialized. No documents loaded or an error occurred during setup."))
            return

        print(f"Received query from Teams: '{user_query}'")

        try:
            # Invoke the RAG chain
            # Pass an empty chat_history list since this PoC doesn't maintain conversation history
            response = rag_chain({"question": user_query, "chat_history": []})
            rag_response = response.get('answer', 'Could not retrieve a specific answer.')

            # Send the response back to Teams
            await turn_context.send_activity(MessageFactory.text(rag_response))
            print(f"Sent response to Teams.") # Log that response was sent

        except Exception as e:
            print(f"Error during RAG query from Teams: {e}")
            await turn_context.send_activity(MessageFactory.text(f"An error occurred during the query: {e}"))


# --- FastAPI Application and Endpoint ---

# Load environment variables for Bot Framework
load_dotenv()
MICROSOFT_APP_ID = os.getenv("MicrosoftAppId")
MICROSOFT_APP_PASSWORD = os.getenv("MicrosoftAppPassword")

# Add print statements to confirm env vars are loaded
if MICROSOFT_APP_ID:
    print("MicrosoftAppId loaded from environment variables.")
else:
    print("WARNING: MicrosoftAppId not found in environment variables.")

if MICROSOFT_APP_PASSWORD:
     print("MicrosoftAppPassword loaded from environment variables.")
else:
     print("WARNING: MicrosoftAppPassword not found in environment variables.")

# Create adapter settings with credentials
SETTINGS = BotFrameworkAdapterSettings(
    app_id=MICROSOFT_APP_ID,
    app_password=MICROSOFT_APP_PASSWORD
)

# Create the Bot Framework Adapter using the settings
# This adapter handles the authentication of incoming requests from Bot Framework
ADAPTER = BotFrameworkAdapter(SETTINGS)
print("BotFrameworkAdapter initialized with loaded settings.")

# Create the custom bot instance
BOT = RagBot()

# Note: For a production application, more granular authorization based on user identity
# or Teams channel/team would be required. This basic setup ensures the message comes
# from the registered bot channel service principal but doesn't authorize specific Teams users.


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Loads documents and sets up the RAG chain on startup.
    """
    global rag_chain
    print("Loading documents and setting up RAG pipeline...")
    # Ensure the data directory exists
    # Use the data_path relative to the script's execution location
    full_data_path = os.path.join(os.path.dirname(__file__), DATA_PATH)
    os.makedirs(full_data_path, exist_ok=True)

    # For the PoC, let's create a dummy file if the data folder is empty to avoid issues
    # Check if any supported files exist, not just any file
    supported_files_exist = any(
        f.endswith((".txt", ".docx", ".pdf"))
        for f in os.listdir(full_data_path)
        if os.path.isfile(os.path.join(full_data_path, f))
    )
    if not supported_files_exist:
         dummy_file_path = os.path.join(full_data_path, "example_document.txt")
         if not os.path.exists(dummy_file_path):
             with open(dummy_file_path, "w") as f:
                 f.write("This is an example document for the knowledge management PoC. It demonstrates that the system can load and process text files.")
             print(f"Created dummy document: {dummy_file_path} because no supported files were found.")
         else:
             print(f"Dummy document already exists: {dummy_file_path}")


    try:
        document_chunks = load_documents(DATA_PATH) # Pass the relative path
        if document_chunks:
            vectorstore, llm = setup_rag_pipeline(document_chunks)

            # Initialize the ConversationalRetrievalChain
            rag_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            print("RAG pipeline setup complete.")
        else:
            print("No document chunks loaded, RAG chain not initialized.")
            rag_chain = None
    except Exception as e:
        print(f"Error during RAG pipeline setup: {e}")
        rag_chain = None

    yield # The application runs here

    # Clean up resources on shutdown (optional for this basic PoC)
    print("Shutting down...")
    pass

# Instantiate FastAPI app and router, passing the lifespan
app = FastAPI(lifespan=lifespan)
router = APIRouter()

# Endpoint for Teams bot messages
@router.post("/api/messages")
async def messages(req: Request):
    """
    Endpoint to receive messages from the Bot Framework.
    This endpoint is authenticated by the BotFrameworkAdapter.
    """
    if "application/json" not in req.headers.get("Content-Type", ""):
        return Response(status_code=415)

    body = await req.json()
    activity = Activity().deserialize(body)

    # Process the activity with the bot's activity handler
    # The adapter's process_activity method handles authentication/validation
    auth_header = req.headers.get("Authorization", "")
    await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)

    return Response(status_code=200)


# Keep the original /query endpoint for potential direct testing or other uses
@router.post("/query")
async def query_rag_direct(request: QueryRequest):
    """
    Endpoint to query the RAG pipeline directly (not via Teams).
    This endpoint does NOT have Bot Framework authentication.

    Args:
        request: The query request body containing the user's query string.

    Returns:
        A dictionary containing the response from the RAG chain.
    """
    if rag_chain is None:
        return {"response": "RAG pipeline is not initialized. No documents loaded or an error occurred during setup."}

    user_query = request.query

    try:
        response = rag_chain({"question": user_query, "chat_history": []})
        rag_response = response.get('answer', 'Could not retrieve a specific answer.')
        return {"response": rag_response}
    except Exception as e:
        print(f"Error during direct RAG query: {e}")
        return {"response": f"An error occurred during the query: {e}"}


# Include the router in the app
app.include_router(router)

# To run this FastAPI app for Teams bot integration:
# 1. Save this code as main.py in your backend directory.
# 2. Ensure you have a .env file in the same directory with your GOOGLE_API_KEY, MicrosoftAppId, and MicrosoftAppPassword
# 3. Navigate to the backend directory in your terminal: cd your_backend_directory
# 4. Run the command: uvicorn main:app --reload
# 5. Configure your bot in the Azure Bot Service and point its messaging endpoint to the URL where this FastAPI app is hosted (e.g., your public ngrok URL + /api/messages)
