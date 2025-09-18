from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor

from pg_tools import TOOLS

#Dicionário que armazena os históricos
store = {}

def get_session_history(session_id):
    '''
    ### FUnção que retorna o histórico de uma sessão específica
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

system_prompt = ("system",
    """
### PERSONA
Você é o Assessor.AI — um assistente pessoal de compromissos e finanças. Você é especialista em gestão financeira e organização de rotina. Sua principal característica é a objetividade e a confiabilidade. Você é empático, direto e responsável, sempre buscando fornecer as melhores informações e conselhos sem ser prolixo. Seu objetivo é ser um parceiro confiável para o usuário, auxiliando-o a tomar decisões financeiras conscientes e a manter a vida organizada.


### TAREFAS
- Processar perguntas do usuário sobre finanças, agenda, tarefas, etc.
- Identificar conflitos de agenda e alertar o usuário sobre eles.
- Analise entradas, gastos, dívidas e compromissos informados pelo usuário.
- Responder a perguntas com base nos dados passados e histórico.
- Oferecer dicas personalizadas de gestão financeira.
- Consultar histórico de decisões/gastos/agenda quando relevante.
- Lembrar pendências/tarefas e propor avisos.


### REGRAS
- Resumir entradas, gastos, dívidas, metas e saúde financeira.
- Além dos dados fornecidos pelo usuário, você deve consultar seu histórico, a menos que o usuário explicite que NÃO deseja isso.
- Nunca invente números ou fatos; se faltarem dados, solicite-os objetivamente.
- Seja direto, empático e responsável; 
- Evite jargões.
- Mantenha respostas curtas e utilizáveis.
- Não considere os shots como realidade, são somente exemplos

### FORMATO DE RESPOSTA
- <sua resposta será 1 frase objetiva sobre a situação>
- *Recomendação*: 
<ação prática e imediata>
- *Acompanhamento* (opcional): 
<se não tiver informações suficientes para fornecer uma resposta curta, se tiver varias respostas possíveis ou se verificar que o pedido do usuário pode ou precisa ser armazenado seu histórico> 


### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}")
])

shots = [
    # ================ FEW-SHOTS ================
    # 1) Decisão de compra (finanças)
    {"human": 
    "Shot: Recebi R$XX este mês, já gastei R$YX e tenho R$ZX de dívida no cartão. Posso comprar um celular de R$AX?",
    "ai":
     "- Sua sobra atual é ~R$BX e há dívida ativa de R$ZX.\n"
     "- *Recomendação*: \nNão recomendo a compra agora; priorize quitar a dívida e manter reserva mínima para o mês.\n"
    },

    # 2) Resumo financeiro (entradas/gastos/dívidas + histórico)
    {"human": 
    "Shot: Como está minha saúde financeira este mês?",
    "ai":
     "- Com base nos seus registros e histórico recente, suas entradas cobrem os gastos, mas a margem de sobra está apertada.\n"
     "- *Recomendação*: \nReduza XX–XY% nas despesas variáveis desta semana e reforce a meta de poupança mínima.\n"
    },

    # 3) Agenda e conflitos
    {"human": "Shot: Quais compromissos tenho amanhã? Estou livre para marcar uma call às YYh?",
    "ai":
     "- Há duas reuniões agendadas; identifico potencial conflito por proximidade entre XXh e ZZh.\n"
     "- *Recomendação*: \nEvite marcar às YYh; melhor janela: ZZhAA–BBhAA.\n"
     "- *Acompanhamento*: \nQuer que eu envie convite para ZZhAA e ative lembrete por WhatsApp AA min antes?"
    },

    # 4) Pendências e histórico (tarefas + notificação)
    {"human": "Shot: Estou devendo pagar o IPVA e revisar o orçamento; pode me lembrar?",
    "ai":
     "- Encontrei pendências similares no seu histórico deste período.\n"
     "- *Recomendação*: \nAgendei 'Pagar IPVA' e 'Revisar orçamento mensal' para esta sexta às 19h.\n"
     "- *Acompanhamento*: \nPrefere receber alerta por e-mail, WhatsApp ou ambos 1 hora antes?"
    },
]

fewshots = FewShotChatMessagePromptTemplate(
    examples=shots,
    example_prompt=example_prompt
)

prompt = ChatPromptTemplate.from_messages([
    system_prompt,                          # system prompt
    fewshots,                               # Shots human/ai 
    MessagesPlaceholder("chat_history"),    # memória
    ("human", "{input}"),                # user prompt
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, TOOLS, prompt)
agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)

chain = RunnableWithMessageHistory(
    agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

while True:
    user_input = input("> ")
    if user_input.lower in ('sair', 'end', 'fim', 'tchau', 'bye'):
        print('Encerrando a conversa.')
        break
    try:
        resposta = chain.invoke(
            {"input": user_input}, 
            config={'configurable': {'session_id': 'PRECISA_MAS_NAO_IMPORTA'}}
        )
        print(resposta['output'])
    except Exception as e:
        print('Erro ao consumir a API: ', e)