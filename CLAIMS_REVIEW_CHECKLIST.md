# Checklist de claims ainda pendentes de revisão

Esta lista foi cruzada com o estado atual do `README.md`. Não inclui os
apontamentos já corrigidos sobre a descrição da grade de 328 runs, o oráculo
como referência de dados completos, as métricas de distância, a escolha do
exemplar do k-means e a separação entre o OPF publicado e as heurísticas do
projeto.

## Métodos de seleção

- [ ] **Documentar o TypiClust como variante da implementação publicada.**
  - Local: [`README.md:130`](README.md#os-métodos-escolhidos-e-por-quê).
  - Explicitar: uso de cosseno e ausência dos filtros/regras para clusters
    pequenos descritos no apêndice do artigo.
  - Concluído quando: o texto separar o núcleo do método das escolhas próprias
    do projeto.

- [ ] **Documentar as adaptações do ProbCover e limitar “estado da arte”.**
  - Local: [`README.md:131`](README.md#os-métodos-escolhidos-e-por-quê).
  - Explicitar: o artigo estima o raio com `k = número de classes`; o projeto
    usa `k = orçamento` e reinicia a cobertura quando ela se esgota.
  - Restringir “estado da arte” aos benchmarks de classificação avaliados no
    artigo de 2022.

- [ ] **Corrigir a atribuição e o nome da variante FreeSel.**
  - Local: [`README.md:132`](README.md#os-métodos-escolhidos-e-por-quê) e
    artefatos/código que usam `freesel_dino`.
  - Problema: o projeto implementa FreeSel-FDS determinístico e k-means local;
    o método principal usa amostragem proporcional à distância² e agrupamento
    espectral guiado por atenção.
  - Decidir se a técnica será renomeada para `freesel_fds_dino` em código,
    seleções, tabelas e figuras.
  - Concluído quando: nome e descrição identificarem inequivocamente a variante.

- [ ] **Reformular “uma placa pequena conta” como hipótese.**
  - Local: [`README.md:132`](README.md#os-métodos-escolhidos-e-por-quê).
  - Problema: FreeSel sustenta padrões locais, mas não demonstra especificamente
    seleção de placas pequenas.
  - Concluído quando: a frase estiver apresentada como hipótese a ser testada
    no estudo, não como efeito comprovado.

## Diagnósticos e medições

- [ ] **Trocar o Jaccard do OPF de `1,000` para `N/A`.**
  - Locais: linhas das quatro frações na seção
    [Diagnósticos das seleções](README.md#diagnósticos-das-seleções).
  - Problema: uma única execução não permite medir sobreposição entre
    repetições.
  - Concluído quando: tabelas, nota de rodapé, CSV gerador e figuras não
    tratarem determinismo como uma medição de Jaccard.

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

- [ ] **Adiar a claim de pool “~10× maior”.**
  - Local: [`README.md:317`](README.md#etapa-2--replicação-no-tt100k).
  - Problema: o artigo sustenta cerca de 10 mil imagens com sinais, mas o pool
    elegível só será conhecido após taxonomia, filtros e split.
  - Concluído quando: a frase for apresentada como estimativa pré-auditoria ou
    substituída pelo tamanho medido do pool.

- [ ] **Validar a viabilidade do OPF no pool completo do TT100K.**
  - Local: [`README.md:324`](README.md#etapa-2--replicação-no-tt100k).
  - Problema: a implementação atual constrói vizinhanças todos-contra-todos, e
    Rocha alerta para custo impraticável em conjuntos grandes.
  - Pendente: executar piloto com medição de tempo e memória antes de congelar
    a afirmação de que o OPF operará sobre todo o pool.
  - Concluído quando: o piloto validar a execução ou o protocolo documentar
    subamostragem/restrição escalável.

## Semi-supervisão

- [ ] **Distinguir FixMatch original da adaptação professor–aluno para detecção.**
  - Local: [`README.md:394`](README.md#etapa-3--trabalhos-futuros-semi-supervisão-na-dissertação).
  - Explicitar: FixMatch usa a mesma rede, não um professor EMA; aplicar sua
    regra dentro de um detector professor–aluno é adaptação do projeto.
  - Tratar `τ = 0,95` como ponto inicial oriundo de classificação, não como
    limiar validado para detecção.

- [ ] **Corrigir a motivação dos limiares FreeMatch.**
  - Local: [`README.md:399`](README.md#etapa-3--trabalhos-futuros-semi-supervisão-na-dissertação).
  - Explicitar: o limiar global é modulado por estatísticas de confiança por
    classe; ele não é estimado diretamente da frequência das classes.
  - Concluído quando: eventual relação com desbalanceamento estiver formulada
    como motivação/hipótese, não como mecanismo do artigo.

- [ ] **Descrever corretamente o peso do SoftMatch e marcar a adaptação por box.**
  - Local: [`README.md:405`](README.md#etapa-3--trabalhos-futuros-semi-supervisão-na-dissertação).
  - Explicitar: a gaussiana é truncada e dinâmica; abaixo da média o peso
    decai, e na média ou acima dela recebe peso máximo.
  - Registrar que aplicar o peso individualmente por box em detecção é uma
    adaptação ainda não validada pelo artigo.

## Dependências experimentais

- [ ] **Executar a grade BVTSLD antes de transformar riscos em resultados.**
  - Estado atual: 0/328 runs.
  - Depois da execução, revisar todas as claims sobre ranking, variância,
    composição por classe e ganho contra `random` usando os artefatos gerados.

- [ ] **Concluir a auditoria e o piloto TT100K antes de congelar claims de escala.**
  - Dependências: taxonomia, elegibilidade, split, tamanho real do pool e custo
    do OPF.
