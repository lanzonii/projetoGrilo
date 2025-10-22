from langchain_core.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor

from utils import (
    today,
    llm,
    get_session_history
)
from pg_tools import AGENDA_TOOLS

class AgendaAgent(RunnableWithMessageHistory):
    def get_chain(self):
        system_prompt = ("system",
            """
            ### OBJETIVO
            Interpretar a PERGUNTA_ORIGINAL sobre agenda/compromissos e (quando houver tools) consultar/criar/atualizar/cancelar eventos. 
            A saída SEMPRE é JSON (contrato abaixo) para o Orquestrador.


            ### TAREFAS



            ### CONTEXTO
            - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
            - Entrada do Roteador:
            - ROUTE=agenda
            - PERGUNTA_ORIGINAL=...
            - PERSONA=...   (use como diretriz de concisão/objetividade)
            - CLARIFY=...   (se preenchido, responda primeiro)


            ### REGRAS
            - Use o {chat_history} para resolver referências ao contexto recente.


            ### SAÍDA (JSON)
            # Obrigatórios:
            - dominio   : "agenda"
            - intencao  : "consultar" | "criar" | "atualizar" | "cancelar" | "listar" | "disponibilidade" | "conflitos"
            - resposta  : uma frase objetiva
            - recomendacao : ação prática (pode ser string vazia)
            # Opcionais (incluir só se necessário):
            - acompanhamento : texto curto de follow-up/próximo passo
            - esclarecer     : pergunta mínima de clarificação
            - janela_tempo   : {{"de":"YYYY-MM-DDTHH:MM","ate":"YYYY-MM-DDTHH:MM","rotulo":"ex.: 'amanhã 09:00–10:00'"}}
            - evento         : {{"titulo":"...","data":"YYYY-MM-DD","inicio":"HH:MM","fim":"HH:MM","local":"...","participantes":["..."]}}


            ### HISTÓRICO DA CONVERSA
            {chat_history}
            """
        )


        shots = [
            {
                "human": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Tenho janela amanhã à tarde?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
                "ai": """{{"dominio":"agenda","intencao":"disponibilidade","resposta":"Você está livre amanhã das 14:00 às 16:00.","recomendacao":"Quer reservar 15:00–16:00?","janela_tempo":{{"de":"2025-09-29T14:00","ate":"2025-09-29T16:00","rotulo":"amanhã 14:00–16:00"}}}}"""
            },
            {
                "human": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Marcar reunião com João amanhã às 9h por 1 hora\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
                "ai": """{{"dominio":"agenda","intencao":"criar","resposta":"Posso criar 'Reunião com João' amanhã 09:00–10:00.","recomendacao":"Confirmo o envio do convite?","janela_tempo":{{"de":"2025-09-29T09:00","ate":"2025-09-29T10:00","rotulo":"amanhã 09:00–10:00"}},"evento":{{"titulo":"Reunião com João","data":"2025-09-29","inicio":"09:00","fim":"10:00","local":"online"}}}}"""
            },
            {
                "human": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Agendar revisão do orçamento na sexta\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
                "ai": """{{"dominio":"agenda","intencao":"criar","resposta":"Preciso do horário para agendar.","recomendacao":"","esclarecer":"Qual horário você prefere na sexta?"}}"""
            },
        ]
        
        prompt_base = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template("{human}"),
            AIMessagePromptTemplate.from_template("{ai}"),
        ])

        fewshots = FewShotChatMessagePromptTemplate(
            examples=shots,
            example_prompt=prompt_base
        )

        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            fewshots,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ]).partial(today_local=today.isoformat())
        
        agent = create_tool_calling_agent(llm, AGENDA_TOOLS, prompt)
        executor = AgentExecutor(agent=agent, tools=AGENDA_TOOLS, verbose=False)

        return executor
    
    def __init__(self):
        super().__init__(
            self.get_chain(),
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )