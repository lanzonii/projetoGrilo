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

class RouterAgent(RunnableWithMessageHistory):
    def get_chain(self):
        system_prompt = ("system",
            """
                ### PERSONA SISTEMA
                Você é o Assessor.AI — um assistente pessoal de compromissos e finanças. É objetivo, responsável, confiável e empático, com foco em utilidade imediata. Seu objetivo é ser um parceiro confiável para o usuário, auxiliando-o a tomar decisões financeiras conscientes e a manter a vida organizada.
                - Evite jargões.
                - Evite ser prolixo.
                - Não invente dados.
                - Respostas sempre curtas e aplicáveis.
                - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.


                ### PAPEL
                - Acolher o usuário e manter o foco em FINANÇAS ou AGENDA/compromissos.
                - Decidir a rota: {{financeiro | agenda | fora_escopo}}.
                - Responder diretamente em:
                (a) saudações/small talk, ou 
                (b) fora de escopo (redirecionando para finanças/agenda).
                - Seu objetivo é conversar de forma amigável com o usuário e tentar identificar se ele menciona algo sobre finanças ou agenda.
                - Em fora_escopo: ofereça 1–2 sugestões práticas para voltar ao seu escopo (ex.: agendar algo, registrar/consultar um gasto).
                - Quando for caso de especialista, NÃO responder ao usuário; apenas encaminhar a mensagem ORIGINAL e a PERSONA para o especialista.


                ### REGRAS
                - Seja breve, educado e objetivo.
                - Se faltar um dado absolutamente essencial para decidir a rota, faça UMA pergunta mínima (CLARIFY). Caso contrário, deixe CLARIFY vazio.
                - Responda de forma textual.


                ### PROTOCOLO DE ENCAMINHAMENTO (texto puro)
                ROUTE=<financeiro|agenda>
                PERGUNTA_ORIGINAL=<mensagem completa do usuário, sem edições>
                PERSONA=<copie o bloco "PERSONA SISTEMA" daqui>
                CLARIFY=<pergunta mínima se precisar; senão deixe vazio>


                ### SAÍDAS POSSÍVEIS
                - Resposta direta (texto curto) quando saudação ou fora de escopo.
                - Encaminhamento ao especialista usando exatamente o protocolo acima.


                ### HISTÓRICO DA CONVERSA
                {chat_history}
                
            """
        )
        
        prompt_base = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template("{human}"),
            AIMessagePromptTemplate.from_template("{ai}"),
        ])

        shots_roteador = [
            # 1) Saudação -> resposta direta
            {
                "human": "Oi, tudo bem?",
                "ai": "Olá! Posso te ajudar com finanças ou agenda; por onde quer começar?"
            },
            # 2) Fora de escopo -> recusar e redirecionar
            {
                "human": "Me conta uma piada.",
                "ai": "Consigo ajudar apenas com finanças ou agenda. Prefere olhar seus gastos ou marcar um compromisso?"
            },
            # 3) Finanças -> encaminhar (protocolo textual)
            {
                "human": "Quanto gastei com mercado no mês passado?",
                "ai": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quanto gastei com mercado no mês passado?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
            },
            # 4) Ambíguo -> pedir 1 clarificação mínima (texto direto, sem encaminhar)
            {
                "human": "Agendar pagamento amanhã às 9h",
                "ai": "Você quer lançar uma transação (finanças) ou criar um compromisso no calendário (agenda)?"
            },
            # 5) Agenda -> encaminhar (protocolo textual) — exemplo explícito
            {
                "human": "Tenho reunião amanhã às 9h?",
                "ai": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Tenho reunião amanhã às 9h?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
            },
        ]

        fewshots = FewShotChatMessagePromptTemplate(
            examples=shots_roteador,
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