# Checklist de claims ainda pendentes de revisão

Esta lista foi cruzada com o estado atual do `README.md`. Não inclui os
apontamentos já corrigidos sobre a descrição da grade de 328 runs, o oráculo
como referência de dados completos, as métricas de distância, a escolha do
exemplar do k-means e a separação entre o OPF publicado e as heurísticas do
projeto.

## Inferência estatística

- [ ] **Resolver a incompatibilidade entre uma única seleção OPF e o teste de sinais.**
  - Local: protocolo estatístico próximo de
    [`README.md:295`](README.md#treino-yolo-por-seleção--map-vs-oráculo).
  - Problema: com apenas uma seleção OPF, o teste bilateral de sinais produz
    `p = 1`; o critério atual nunca poderá declarar suporte estatístico para o
    OPF.
  - Decisão necessária: criar repetições independentes válidas, definir outra
    unidade inferencial/teste para o OPF ou declarar formalmente que sua análise
    será apenas descritiva.
  - Concluído quando: protocolo, código de análise e interpretação concordarem.

## TT100K

- [ ] **Validar a viabilidade do OPF no pool completo do TT100K.**
  - Local: [`README.md:324`](README.md#etapa-2--replicação-no-tt100k).
  - Problema: a implementação atual constrói vizinhanças todos-contra-todos, e
    Rocha alerta para custo impraticável em conjuntos grandes.
  - Pendente: executar piloto com medição de tempo e memória antes de congelar
    a afirmação de que o OPF operará sobre todo o pool.
  - Concluído quando: o piloto validar a execução ou o protocolo documentar
    subamostragem/restrição escalável.



## Dependências experimentais

- [ ] **Concluir a auditoria e o piloto TT100K antes de congelar claims de escala.**
  - Dependências: taxonomia, elegibilidade, split, tamanho real do pool e custo
    do OPF.
