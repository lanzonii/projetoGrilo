from langchain_core.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSequence
)
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import (
    today,
    llm_fast,
    get_session_history,
    embeddings
)

from operator import itemgetter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

class FaqAgent():

    def __init__(self):
        PDF_PATH = 'faq_tools.pdf'
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, embeddings)
        self.chain = self.get_chain(db)

    def get_faq_context(self, question, db):
        results = db.similarity_search(question, k=6)
        return results

    def get_chain(self, db):
        system_prompt = ("system",
            """
                ### PAPEL
                - Você deve responder perguntas e dúvidas SOMENTE com base no documento normativo oficial (trechos fornecidos em CONTEXTO)
                - Se a informação solicitada não constar no documento, diga: "Essa informação não consta em nosso FAQ."


                ### REGRAS
                - Seja breve, educado e objetivo.
                - Se faltar um dado absolutamente essencial para decidir a rota, faça UMA pergunta mínima (CLARIFY). Caso contrário, deixe CLARIFY vazio.
                - Se a mensagem do usuário for uma dúvida geral sobre o sistema, funcionalidades, regras ou políticas > ROUTE=faq
                - Se for uma operação financeira, orçamento, transação > ROUTE=financeiro
                - Se for sobre compromissos, eventos, lembretes > ROUTE=agenda
                - Se fora_escopo: ofereça 1–2 sugestões práticas para voltar ao seu escopo (ex.: agendar algo, registrar/consultar um gasto).
                - Responda de forma textual.

                
                ### ENTRADA
                - ROUTE=faq
                - PERGUNTA ORIGINAL=...
                - PERSONA=... (use como diretriz de concisão/objetividade)
                - CLARIFY=... (se preenchido, responda primeiro)


                ### HISTÓRICO DA CONVERSA
                {chat_history}
                
            """
        )

        
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            ("human", (
             "Pergunta do usuário:\n{question}\n\n"
             "CONTEXTO (trechos do documento):\n{context}\n\n"
             "Responda com base APENAS no CONTEXTO."))
        ])

        return (RunnablePassthrough.assign( question=itemgetter("input"), context= lambda x: self.get_faq_context(x['input'], db), chat_history=lambda x: []) | prompt | llm_fast | StrOutputParser())