from router import RouterAgent
from orchestrator import OrchestratorAgent
from financial import FinancialAgent
from agenda import AgendaAgent
from FAQ import FaqAgent
from langgraph.graph import StateGraph, START, END

router_chain = RouterAgent()
orchestrator_chain = OrchestratorAgent()
financial_chain = FinancialAgent()
agenda_chain = AgendaAgent()
faq_chain = FaqAgent().chain

# Criação dos nós no langGraph
def router_node(state: dict) -> dict:
    global router_chain
    resposta_roteador = router_chain.invoke(
        {"input": state["input"]}, 
        config={"configurable": {"session_id": state["session_id"]}}
    )  
    
    if not resposta_roteador.startswith("ROUTE="):
        return {"resposta_usuario": resposta_roteador}
    
    rota = resposta_roteador.split("\n", 1)[0].split("=", 1)[1].strip().lower()
    if rota not in {"financeiro", "agenda", "faq"}:
        return {"erro": f"Rota inválida: {rota}"}

    return {"rota": rota, "roteador": resposta_roteador, 'input':state['input'], 'session_id': state['session_id']}

def faq_node(state: dict) -> dict:
    global faq_chain
    result = faq_chain.invoke(
        {"input": state['input']}, 
        config={"configurable": {"session_id": state["session_id"]}}
    )  
    return {"resposta_usuario": result, 'session_id': state['session_id']}

def financeiro_node(state: dict) -> dict:
    global financial_chain
    result = financial_chain.invoke(
        {"input": state['roteador']}, 
        config={"configurable": {"session_id": state["session_id"]}}
    )  
    return {"saida_especialista": result["output"], 'session_id': state['session_id']}

def agenda_node(state: dict) -> dict:
    global agenda_chain
    result = agenda_chain.invoke(
        {"input": state['roteador']}, 
        config={"configurable": {"session_id": state["session_id"]}}
    )  
    return {"saida_especialista": result["output"], 'session_id': state['session_id']}

def orchestrator_node(state: dict) -> dict:
    global orchestrator_chain
    resposta_final = orchestrator_chain.invoke(
        {"input": state['saida_especialista']},
        config={"configurable": {"session_id": state["session_id"]}}
    )  
    return {"resposta_usuario": resposta_final}
# ------------------- DECISOR ------------------------

def decide_after_router(state: dict) -> str:
    if state.get("erro") or state.get("resposta_usuario"):
        return "end"
    rota = state.get("rota")
    if rota == "financeiro":
        return "financeiro"
    if rota == "agenda":
        return "agenda"
    if rota == "faq":
        return "faq"
    return "end"

def decide_after_specialist(state: dict) -> str:
    if state.get("erro"):
        return "end"
    return "orquestrador"

# ------------------- CONSTRUÇÃO DO GRAFO ------------

graph = StateGraph(dict)

graph.add_node("roteador", router_node)
graph.add_node("financeiro", financeiro_node)
graph.add_node("agenda", agenda_node)
graph.add_node("faq", faq_node)
graph.add_node("orquestrador", orchestrator_node)

graph.add_edge(START, "roteador")

graph.add_conditional_edges(
    "roteador",
    decide_after_router,
    {
        "financeiro": "financeiro",
        "agenda": "agenda",
        "faq": "faq",
        "end": END,
    },
)

graph.add_conditional_edges(
    "financeiro",
    decide_after_specialist,
    {"orquestrador": "orquestrador", "end": END},
)
graph.add_conditional_edges(
    "agenda",
    decide_after_specialist,
    {"orquestrador": "orquestrador", "end": END},
)
graph.add_conditional_edges(
    "faq",
    decide_after_specialist,
    {"orquestrador": "orquestrador", "end": END},
)

graph.add_edge("orquestrador", END)

app = graph.compile()


# ------------------- FUNÇÃO DE EXECUÇÃO --------------

def executar_fluxo_assessor(pergunta_usuario: str, session_id: str) -> str:
    global app
    final_state = app.invoke({"input": pergunta_usuario, "session_id": session_id})
    if final_state.get("erro"):
        return f"Erro: {final_state['erro']}"
    return final_state.get("resposta_usuario", "Não foi possível responder.") # isso é um if não tiver resposta_usuario, mostre a "não foi possivel..."

while True:
    try:
        user_input = input("> ")
        if user_input.lower() in ('sair', 'end', 'fim', 'tchau', 'bye'):
            print("Encerrando a conversa.")
            break
        
        # Chama a função orquestradora que executa o fluxo completo (Roteador -> Especialista -> Orquestrador)
        resposta = executar_fluxo_assessor(
            pergunta_usuario=user_input, 
            session_id="PRECISA_MAS_NÃO_IMPORTA"
        )
        
        # Imprime a resposta formatada para o usuário (saída do Orquestrador/Roteador)
        print(resposta)
        
    except Exception as e:
            print("Erro ao consumir a API:", e)
            continue