from langchain_core.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables.history import RunnableWithMessageHistory

from utils import (
    today,
    llm_fast,
    get_session_history
)

class OrchestratorAgent(RunnableWithMessageHistory):
    def get_chain(self):
        system_prompt = ("system",
            """
                ### PAPEL
                Você é o Agente Orquestrador do Assessor.AI. Sua função é entregar a resposta final ao usuário **somente** quando um Especialista retornar o JSON.


                ### ENTRADA
                - ESPECIALISTA_JSON contendo chaves como:
                dominio, intencao, resposta, recomendacao (opcional), acompanhamento (opcional),
                esclarecer (opcional), janela_tempo (opcional), evento (opcional), escrita (opcional), indicadores (opcional).


                ### REGRAS
                - Use **exatamente** `resposta` do especialista como a **primeira linha** do output.
                - Se `recomendacao` existir e não for vazia, inclua a seção *Recomendação*; caso contrário, **omita**.
                - Para *Acompanhamento*: se houver `esclarecer`, use-o; senão, se houver `acompanhamento`, use-o; caso contrário, **omita** a seção.
                - Não reescreva números/datas se já vierem prontos. Não invente dados. Seja conciso.
                - Não retorne JSON; **sempre** retorne no FORMATO DE SAÍDA.


                ### FORMATO DE SAÍDA (sempre ao usuário)
                <sua resposta será 1 frase objetiva sobre a situação>
                - *Recomendação*:
                <ação prática e imediata>     # omita esta seção se não houver recomendação
                - *Acompanhamento* (opcional):
                <pergunta/minipróximo passo>  # omita se nada for necessário


                ### HISTÓRICO DA CONVERSA
                {chat_history}
            """
        )

        shots = [
            # 1) Financeiro — consultar
            {
                "human": """ESPECIALISTA_JSON:\n{{"dominio":"financeiro","intencao":"consultar","resposta":"Você gastou R$ 842,75 com 'comida' no mês passado.","recomendacao":"Quer detalhar por estabelecimento?","janela_tempo":{{"de":"2025-08-01","ate":"2025-08-31","rotulo":"mês passado (ago/2025)"}}}}""",
                "ai": "Você gastou R$ 842,75 com 'comida' no mês passado.\n- *Recomendação*:\nQuer detalhar por estabelecimento?"
            },

            # 2) Financeiro — falta dado → esclarecer
            {
                "human": """ESPECIALISTA_JSON:\n{{"dominio":"financeiro","intencao":"resumo","resposta":"Preciso do período para seguir.","recomendacao":"","esclarecer":"Qual período considerar (ex.: hoje, esta semana, mês passado)?"}}""",
                "ai": """Preciso do período para seguir.\n- *Acompanhamento* (opcional):\nQual período considerar (ex.: hoje, esta semana, mês passado)?"""
            },

            # 3) Agenda — criar
            {
                "human": """ESPECIALISTA_JSON:\n{{"dominio":"agenda","intencao":"criar","resposta":"Posso criar 'Reunião com João' amanhã 09:00–10:00.","recomendacao":"Confirmo o envio do convite?","janela_tempo":{{"de":"2025-09-29T09:00","ate":"2025-09-29T10:00","rotulo":"amanhã 09:00–10:00"}},"evento":{{"titulo":"Reunião com João","data":"2025-09-29","inicio":"09:00","fim":"10:00","local":"online"}}}}""",
                "ai": """Posso criar 'Reunião com João' amanhã 09:00–10:00.\n- *Recomendação*:\nConfirmo o envio do convite?"""
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
            ("human", "{input}")
        ]).partial(today_local=today.isoformat())
        
        return prompt | llm_fast | StrOutputParser()
    
    def __init__(self):
        super().__init__(
            self.get_chain(),
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )