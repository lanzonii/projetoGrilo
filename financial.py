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
from pg_tools import TOOLS

class FinancialAgent(RunnableWithMessageHistory):
    def get_chain(self):
        system_prompt = ("system",
            """
                ### OBJETIVO
                Interpretar a PERGUNTA_ORIGINAL sobre finanças e operar as tools de `transactions` para responder. 
                A saída SEMPRE é JSON (contrato abaixo) para o Orquestrador.


                ### TAREFAS
            


                ### CONTEXTO
                - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
                - Entrada vem do Roteador via protocolo:
                - ROUTE=financeiro
                - PERGUNTA_ORIGINAL=...
                - PERSONA=...   (use como diretriz de concisão/objetividade)
                - CLARIFY=...   (se preenchido, priorize responder esta dúvida antes de prosseguir)


                ### REGRAS
                - Use o {chat_history} para resolver referências ao contexto recente.



                ### SAÍDA (JSON)
                Campos mínimos para enviar para o orquestrador:
                # Obrigatórios:
                - dominio   : "financeiro"
                - intencao  : "consultar" | "inserir" | "atualizar" | "deletar" | "resumo"
                - resposta  : uma frase objetiva
                - recomendacao : ação prática (pode ser string vazia se não houver)
                # Opcionais (incluir só se necessário):
                - acompanhamento : texto curto de follow-up/próximo passo
                - esclarecer     : pergunta mínima de clarificação (usar OU 'acompanhamento')
                - escrita        : {{"operacao":"adicionar|atualizar|deletar","id":123}}
                - janela_tempo   : {{"de":"YYYY-MM-DD","ate":"YYYY-MM-DD","rotulo":'mês passado'}}
                - indicadores    : {{chaves livres e numéricas úteis ao log}}


                ### HISTÓRICO DA CONVERSA
                {chat_history}
            """
        )


        shots = [
            {
                "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quanto gastei com mercado no mês passado?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
                "ai": """{{"dominio":"financeiro","intencao":"consultar","resposta":"Você gastou R$ 842,75 com 'comida' no mês passado.","recomendacao":"Quer detalhar por estabelecimento?","janela_tempo":{{"de":"2025-08-01","ate":"2025-08-31","rotulo":"mês passado (ago/2025)"}}}}"""
            },
            {
                "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Registrar almoço hoje R$ 45 no débito\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
                "ai": """{{"dominio":"financeiro","intencao":"inserir","resposta":"Lancei R$ 45,00 em 'comida' hoje (débito).","recomendacao":"Deseja adicionar uma observação?","escrita":{{"operacao":"adicionar","id":2045}}}}"""
            },
            {
                "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quero um resumo dos gastos\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
                "ai": """{{"dominio":"financeiro","intencao":"resumo","resposta":"Preciso do período para seguir.","recomendacao":"","esclarecer":"Qual período considerar (ex.: hoje, esta semana, mês passado)?"}}"""
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
        
        agent = create_tool_calling_agent(llm, TOOLS, prompt)
        executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)

        return executor
    
    def __init__(self):
        super().__init__(
            self.get_chain(),
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )