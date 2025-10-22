# Olá, seja bem-vindo ao repositório do meu agente
## Esse repositório até agora contempla:
 - Agente roteador -> responde perguntas simples ou redireciona para os especialistas
 - Agente financeiro -> Responde perguntas e registra eventos relacionados a finanças com base no banco de dados
 - Agente de agenda -> Responde perguntas relacionadas à sua agenda e futuramente vai ter conexão com o google agenda
 - Agente de FAQ -> Responde perguntas operacionais do sistema somente com base no [documento pdf](faq_tools.pdf)
 - Agente orquestrador -> Atua como um "juiz de IA", verificando a resposta dos agentes especialistas e repassando ao usuário

## Organização do código:
#### A main do grilo tava gigantesca, todos os agentes estavam sendo criados lá desde o início e isso tava me dando muita agonia
#### Por isso, eu separei um arquivo pra cada agente e um "utils", que contempla as coisas em comum deles, como o llm, e etc.
### A estrutura dos arquivos de agente consiste na mesma coisa do grilo só que em uma classe, que geralmente herda do RunnableWithMessageHistory

# NAO TA RODANDO AINDA MAS VOU FAZER RODAR!!