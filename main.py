from router import RouterAgent
from orchestrator import OrchestratorAgent
from financial import FinancialAgent
from agenda import AgendaAgent
from FAQ import FaqAgent

router_chain = RouterAgent()
orchestrator_chain = OrchestratorAgent()
financial_chain = FinancialAgent()
agenda_chain = AgendaAgent()
faq_chain = FaqAgent()

def executar_fluxo_acessor(user_input, session_id) -> str:
    global router_chain
    resposta_roteador = router_chain.invoke(
            {"input": user_input}, 
            config={'configurable': {'session_id': session_id}}
        )
    if 'ROUTE=' in resposta_roteador:

        resposta = dict([tuple(i.split('=')) for i in resposta_roteador.split('\n')])

        if resposta['ROUTE'] == 'faq':
            return faq_chain.invoke(resposta['PERGUNTA_ORIGINAL'])
        elif resposta['ROUTE'] == 'financeiro':
            return financial_chain.invoke(resposta['PERGUNTA_ORIGINAL'])
        elif resposta['ROUTE'] == 'agenda':
            return agenda_chain.invoke(resposta['PERGUNTA_ORIGINAL']) 
    else:
        return resposta_roteador

while True:
    user_input = input("> ")
    if user_input.lower in ('sair', 'end', 'fim', 'tchau', 'bye'):
        print('Encerrando a conversa.')
        break
    try:
        resposta = executar_fluxo_acessor(user_input, 'PRECISA_MAS_NAO_IMPORTA')
        print(resposta)
    except Exception as e:
        print('Erro ao consumir a API: ', e)