from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ChatMessageHistory
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_google_genai import GoogleGenerativeAIEmbeddings

TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()

store = {}

def get_session_history(session_id):
    '''
    ### Função que retorna o histórico de uma sessão específica
    - session_id: id da sessão
    '''
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=os.getenv('GEMINI_API_KEY'))