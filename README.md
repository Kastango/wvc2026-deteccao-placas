# Seleção de amostras sem rótulos para detecção de placas de trânsito

## O problema

Treinar um detector de placas exige rotular *bounding boxes* manualmente. O
custo cresce com o tamanho do conjunto de dados. Antes de existir qualquer
rótulo, é preciso decidir quais imagens rotular primeiro. Esse é o problema de
*cold-start sample selection*: escolher as primeiras imagens sem nenhum rótulo
disponível.

A pergunta de pesquisa é:

> É possível escolher, **sem usar rótulos do dataset-alvo**, um pequeno
> conjunto de imagens que preserve o desempenho de um detector de sinais de
> trânsito?

O experimento compara dois treinos do mesmo YOLOv8n. O **oráculo** usa 100%
do pool rotulado e define o teto de desempenho. Cada método de seleção escolhe
5, 10, 20 ou 50% do pool, e o YOLOv8n é treinado só com essas imagens. A grade
completa produz o mAP de cada método, fração de rótulos, repetição e semente
de treino.

## As etapas do estudo

**Etapa 1 — BVTSLD (teste preliminar)** — pool de 693 imagens. Valida o
*pipeline* e compara os métodos em escala pequena.

- [x] Auditoria do dataset, partições fixas e oráculo
- [x] 164 seleções (6 métodos × 4 frações × 8 repetições; OPF: 1) e diagnósticos
- [ ] Grade completa de treino YOLO (0/328 runs)

**Etapa 2 — TT100K (replicação em escala)** — pool ~10× maior. Verifica se o
ranking dos métodos se mantém.

- [ ] Auditoria do dataset e mapeamento de taxonomia
- [ ] Partições fixas pool/validação/teste (semente 42)
- [ ] Embeddings DINOv2 e padrões locais DINO do pool
- [ ] Oráculo YOLOv8n com 100% do pool
- [ ] 164 seleções (OPF determinístico: 1 repetição) e diagnósticos
- [ ] Grade completa de treino YOLO (328 runs) e análise estatística

**Etapa 3 — Semi-supervisão (dissertação)** — a melhor seleção vira o conjunto
rotulado inicial; o restante do pool entra com *pseudo-labels*.

- [ ] Montar o pipeline professor–aluno de detecção semi-supervisionada
- [ ] Comparar as estratégias de filtragem de *pseudo-labels* (estilo FixMatch, FreeMatch e SoftMatch) com base nas Etapas 1 e 2

---

## Etapa 1 — Teste preliminar no BVTSLD

### Conjunto de dados e partições fixas

O BVTSLD (Brazilian Vertical Traffic Signs and Lights Dataset) foi auditado e
mapeado para três classes-alvo: `regulatory`, `warning` e `information`.
Imagens com semáforos fora dessa taxonomia ficam em quarentena.

| Resultado | Valor |
|---|---:|
| Imagens originais elegíveis | 990 |
| *Bounding boxes* | 1.279 |
| *Bounding boxes* `regulatory` | 1.084 |
| *Bounding boxes* `warning` | 98 |
| *Bounding boxes* `information` | 97 |
| Imagens em quarentena (semáforos fora da taxonomia) | 373 |
| Pool de treino | 693 imagens |
| Partição de validação | 148 imagens |
| Partição de teste | 149 imagens |
| Vazamento entre partições | 0 imagens |

As partições são fixas (semente 42). A partição de teste permanece fechada.
Ela será aberta uma única vez, no final, para a avaliação definitiva. Todas as
comparações intermediárias usam apenas a validação.

Fontes: [`records.json`](outputs/bvtsld/records.json),
[`split.json`](outputs/bvtsld/split.json),
[`quarantine.json`](outputs/bvtsld/quarantine.json) e
[`taxonomy_report.json`](outputs/bvtsld/taxonomy_report.json).

### Oráculo YOLOv8n — o teto de referência

O oráculo foi treinado com 100% do pool no protocolo fixo (ver
[apêndice](#protocolo-fixo-de-treino-yolo)). Apenas a validação foi avaliada.

| Partição | mAP@0.5 | mAP@0.5:0.95 | AP@0.5 `regulatory` | AP@0.5 `warning` | AP@0.5 `information` |
|---|---:|---:|---:|---:|---:|
| Validação | 0,9483 | 0,6270 | 0,9645 | 0,9721 | 0,9082 |

- Tempo de treino: 2.318,2 s (~38,6 min) em Apple M2 Pro/MPS.
- Checkpoint local: `outputs/bvtsld/runs/oracle/weights/best.pt` (fora do Git).
- Protocolo e métricas: [`oracle_results.json`](outputs/bvtsld/oracle_results.json).

![Curvas de treino do oráculo](figs/oracle_training_curves.png)

*Perdas de treino e validação, precisão, revocação e mAP ao longo das 40
épocas.*

![Curva precisão-revocação do oráculo na validação](figs/oracle_validation_pr_curve.png)

*Curvas de precisão–revocação na validação. O valor agregado é 0,948 mAP@0.5.*

![Matriz de confusão normalizada do oráculo](figs/oracle_validation_confusion_matrix_normalized.png)

*Matriz de confusão normalizada na validação. A coluna `background` mostra
falsos positivos; a linha `background`, falsos negativos.*

![Predições do oráculo em imagens de validação](figs/oracle_validation_predictions.jpg)

*Exemplos de predições do oráculo, com a classe-alvo e a confiança do
detector.*

### Os métodos escolhidos e por quê

Nenhum método usa rótulos do dataset-alvo. Eles usam apenas *embeddings* de
redes auto-supervisionadas prontas (vetores L2-normalizados, distância de
cosseno) ou nenhuma representação. O conjunto cobre as principais famílias de
seleção *cold-start* da literatura, com um representante de cada família:

| Método | Representação | Como seleciona | Por que está no estudo | Referência |
|---|---|---|---|---|
| `random` | Nenhuma | Sorteia as imagens do orçamento de forma uniforme. | *Baseline* de controle. O protocolo estatístico mede o ganho pareado de cada método contra ele. | — |
| `kmeans_dinov2` | DINOv2, 384 dim, imagem inteira | Forma `k = orçamento` grupos e escolhe o medoide de cada um. | Representante simples da família de **representatividade global**: uma imagem real por grupo de cenas. | k-means: [Lloyd (1982)](https://doi.org/10.1109/TIT.1982.1056489); DINOv2: [Oquab et al. (2024)](https://arxiv.org/abs/2304.07193) |
| `opf_dinov2` | DINOv2, 384 dim, imagem inteira | Usa as raízes da floresta de caminhos ótimos (OPF) como picos de densidade e completa o orçamento com cotas proporcionais por grupo. | Descobre o número de grupos **de forma adaptativa**, sem impor `k = orçamento`. Hipótese complementar ao k-means. | [Rocha, Cappabianco & Falcão (2009)](https://doi.org/10.1002/ima.20191) |
| `typiclust_dinov2` | DINOv2, 384 dim, imagem inteira | Forma `k = orçamento` grupos e escolhe a imagem de maior densidade local em cada um. | Método de referência para **rotulagem com orçamento baixo**: amostras típicas superam estratégias de incerteza quando há poucos rótulos. | [Hacohen, Dekel & Weinshall (2022)](https://arxiv.org/abs/2202.02794) |
| `probcover_dinov2` | DINOv2, 384 dim, imagem inteira | Escolhe a imagem que cobre mais vizinhos ainda não cobertos, dentro de um raio estimado sem rótulos. | Formula a seleção como **cobertura** do pool. Estado da arte em *cold-start* junto com o TypiClust. | [Yehuda et al. (2022)](https://arxiv.org/abs/2205.11320) |
| `freesel_dino` | DINO v1, 384 dim, **padrões locais** (5 por imagem) | Busca o padrão local ainda não coberto mais distante; a imagem dona do padrão entra na seleção. | Único método que enxerga **regiões locais** — uma placa pequena conta, mesmo quando a cena inteira já parece representada. Usa DINO v1 por fidelidade ao artigo original, cujos mapas de atenção guiam a extração dos padrões. | FreeSel: [Xie et al. (2023)](https://arxiv.org/abs/2309.17342); DINO: [Caron et al. (2021)](https://arxiv.org/abs/2104.14294) |

Duas observações de protocolo:

- O `opf_dinov2` roda sempre sobre o pool inteiro e é determinístico; repetições
  adicionais produziriam a mesma seleção. Por isso ele usa 1 repetição tanto no
  BVTSLD quanto no TT100K, enquanto os demais métodos usam 8.
- Os orçamentos são de 35 (5%), 69 (10%), 139 (20%) e 346 (50%) imagens por
  seleção.

### Diagnósticos das seleções

Foram geradas e auditadas 164 seleções: 6 métodos × 4 frações de rótulos × 8
repetições (OPF: 1). As tabelas abaixo são diagnósticos calculados **antes do
treino YOLO**. Eles medem a representação do pool, a estabilidade, a
quantidade de *bounding boxes* recuperadas e o tempo de seleção. Não definem
o ranking.

Como ler as colunas:

- **Cobertura DINOv2**: distância média de cada imagem do pool à imagem
  selecionada mais parecida no espaço DINOv2 global. Menor é melhor.
- **Δ cobertura**: diferença relativa para o `random` na mesma fração.
  Negativo é melhor.
- **Δ pior caso**: mesma comparação usando a maior distância encontrada.
  Negativo é melhor.
- **Jaccard**: sobreposição entre as repetições do método. Mede estabilidade.
- **Tempo (s)**: tempo para gerar todas as repetições da fração, em um Apple
  M2 Pro. Os valores atuais foram medidos com a máquina compartilhada com
  outros processos e serão remedidos de forma sequencial em um servidor com
  GPU dedicado e ocioso. A GPU acelera a extração das representações; as
  rotinas de seleção baseadas no scikit-learn continuam majoritariamente na
  CPU. Até essa remedição, os valores servem apenas como ordem de grandeza.

#### Fração de 5% — 35 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `kmeans_dinov2` | 0,1996 | −18,1% | −18,2% | 0,305 | 48,4 | 101,6 |
| `typiclust_dinov2` | 0,2006 | −17,7% | −17,3% | 0,307 | 49,5 | 45,5 |
| `probcover_dinov2` | 0,2051 | −15,8% | −18,0% | 0,490 | 48,8 | 42,4 |
| `random` | 0,2436 | 0,0% | 0,0% | 0,028 | 44,0 | <0,1 |
| `opf_dinov2` | 0,2486 | +2,0% | +14,4% | 1,000¹ | 50,0 | 6,6 |
| `freesel_dino` | 0,2575 | +5,7% | −16,1% | 0,330 | 44,2 | 4,5 |

#### Fração de 10% — 69 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `kmeans_dinov2` | 0,1655 | −19,6% | −25,5% | 0,293 | 92,4 | 181,9 |
| `typiclust_dinov2` | 0,1666 | −19,1% | −23,9% | 0,274 | 93,2 | 81,7 |
| `probcover_dinov2` | 0,1708 | −17,0% | −20,5% | 0,414 | 93,2 | 71,4 |
| `freesel_dino` | 0,2048 | −0,5% | −19,5% | 0,494 | 83,2 | 8,7 |
| `random` | 0,2058 | 0,0% | 0,0% | 0,056 | 88,4 | <0,1 |
| `opf_dinov2` | 0,2211 | +7,4% | +15,1% | 1,000¹ | 99,0 | 3,7 |

#### Fração de 20% — 139 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `kmeans_dinov2` | 0,1234 | −21,6% | −35,2% | 0,391 | 185,6 | 380,5 |
| `typiclust_dinov2` | 0,1237 | −21,4% | −31,0% | 0,387 | 183,6 | 166,6 |
| `probcover_dinov2` | 0,1307 | −17,0% | −15,7% | 0,344 | 181,9 | 141,3 |
| `random` | 0,1574 | 0,0% | 0,0% | 0,113 | 179,6 | <0,1 |
| `freesel_dino` | 0,1576 | +0,1% | −28,3% | 0,669 | 171,5 | 16,3 |
| `opf_dinov2` | 0,1862 | +18,3% | −4,3% | 1,000¹ | 187,0 | 3,7 |

#### Fração de 50% — 346 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `kmeans_dinov2` | 0,0550 | −30,5% | −63,2% | 0,682 | 441,5 | 917,9 |
| `typiclust_dinov2` | 0,0550 | −30,4% | −63,1% | 0,670 | 441,9 | 567,1 |
| `probcover_dinov2` | 0,0668 | −15,5% | −63,9% | 0,762 | 450,5 | 508,4 |
| `freesel_dino` | 0,0716 | −9,5% | −24,7% | 0,861 | 434,4 | 51,5 |
| `random` | 0,0791 | 0,0% | 0,0% | 0,337 | 450,1 | <0,1 |
| `opf_dinov2` | 0,1126 | +42,3% | +7,1% | 1,000¹ | 460,0 | 4,2 |

¹ O OPF é determinístico neste pool (o pool inteiro cabe em um único ajuste do
algoritmo). O protocolo usa uma única repetição para o método no BVTSLD.

Leitura preliminar: `kmeans_dinov2` e `typiclust_dinov2` têm a melhor
cobertura média nas quatro frações, com `probcover_dinov2` próximo. O
`opf_dinov2` tem cobertura pior que o `random` em todas as frações, mas
recupera mais *bounding boxes*. Esse resultado **não sustenta um ranking entre
métodos**: k-means, TypiClust e ProbCover selecionam no mesmo espaço DINOv2 em
que a cobertura é medida, enquanto FreeSel opera em padrões DINO locais e
`random` não otimiza representação alguma. Portanto, a cobertura DINOv2 é um
diagnóstico interno de comportamento, não evidência comparativa de qualidade.
Uma eventual análise de cobertura cruzada deve usar uma representação externa,
congelada antes da análise e não utilizada por nenhum seletor. O ranking real
virá exclusivamente do mAP da grade de treino.

Fonte completa: [`selections_summary.csv`](outputs/bvtsld/selections_summary.csv).
As seleções individuais estão em
[`outputs/bvtsld/selections/`](outputs/bvtsld/selections/).

### Como cada método enxerga o pool

![Comparação dos espaços de representação dos seis métodos de seleção](figs/methods_selection_spaces_bvtsld_tsne_frac10_rep1.png)

*Uma seleção real de cada método, com fração de 10%. Os pontos cinza são as
imagens do pool no espaço de representação usado pelo método; os pontos azuis
são as 69 imagens selecionadas. No `FreeSel`, cada imagem tem cinco padrões
locais, mas apenas o padrão que motivou a escolha é destacado, para manter a
comparação em 69 pontos azuis. O `random` não usa embedding e aparece em uma
grade arbitrária de índices. Cada t-SNE é independente e serve apenas para
visualização; a seleção opera no espaço original de 384 dimensões.*

### Treino YOLO por seleção — mAP vs. oráculo

A grade contém **328 runs**: 164 seleções × 2 sementes de treino (41 e 42),
40 épocas cada, no mesmo protocolo do oráculo. Cada run registra em
[`triage_results.csv`](outputs/bvtsld/triage_results.csv):

- **Qualidade**: precisão, revocação, F1, mAP@0.5, mAP@0.75, mAP@0.5:0.95 e
  AP@0.5 por classe na validação. O mAP@0.5:0.95 já varre limiares de IoU de
  0,50 a 0,95; o mAP@0.75 dá a leitura em IoU estrito. A AP por classe é
  necessária porque `warning` e `information` são raras (98 e 97 *bounding
  boxes* no pool): na fração de 5%, uma seleção pode conter zero exemplos de
  uma dessas classes, a AP dela desaba e domina a variância do mAP — reportar
  por classe separa esse efeito da qualidade geral da seleção.
- **Tempo**: tempo de treino, tempo de validação, inferência média por imagem
  (ms) e tempo de CPU (usuário + sistema) do run.
- **Consumo computacional**: pico de RAM do processo, memória média e de pico
  da GPU durante o run. A utilização média da GPU (%) é registrada apenas em
  device CUDA; o macOS não expõe essa leitura sem privilégios de
  administrador.

A fase de seleção tem o próprio registro: o
[`selections_summary.csv`](outputs/bvtsld/selections_summary.csv) guarda, por
técnica × fração, tempo de seleção, RAM e as métricas de cobertura. O script
[`summarize_metrics.py`](scripts/summarize_metrics.py) cruza os dois arquivos
e gera o [`metrics_summary.csv`](outputs/bvtsld/metrics_summary.csv): uma
linha por técnica × fração com média e desvio de todas as métricas — a fonte
direta das tabelas deste README.

A tabela abaixo será preenchida com a média sobre repetições e sementes. Cada
célula reporta mAP@0.5 / mAP@0.5:0.95 na validação.

Referência: o oráculo, treinado com 100% do pool, atinge **0,9483 / 0,6270**.

| Método | 5% | 10% | 20% | 50% |
|---|---:|---:|---:|---:|
| `random` | — / — | — / — | — / — | — / — |
| `kmeans_dinov2` | — / — | — / — | — / — | — / — |
| `opf_dinov2` | — / — | — / — | — / — | — / — |
| `typiclust_dinov2` | — / — | — / — | — / — | — / — |
| `probcover_dinov2` | — / — | — / — | — / — | — / — |
| `freesel_dino` | — / — | — / — | — / — | — / — |

Estado atual da etapa de treino:

| Item | Estado |
|---|---:|
| Seleções salvas | 164/164 |
| Configurações de treino materializadas | 164/164 |
| Treino de verificação (*smoke*) | aprovado |
| Grade completa de treino | **0/328 runs** |

O treino de verificação tem duas épocas. Ele confirma que dataset, rótulos,
seleção, treino, validação e gravação dos artefatos funcionam de ponta a
ponta. Seu mAP não é resultado experimental. Após a grade completa, a análise
estatística (ganho médio pareado contra o `random`, intervalo de confiança de
95% por *bootstrap* hierárquico, teste exato de aleatorização de sinais e
correção de Holm) é gerada por
[`analyze_triage.py`](scripts/analyze_triage.py). O estado auditável está em
[`project_status.json`](outputs/bvtsld/project_status.json).

Depois de escolher a fração na validação, a abertura única do teste avaliará
**todos os métodos nessa fração**, e não apenas o vencedor da validação. A
escolha não será refeita com base no teste. Isso permite verificar se o ganho
do vencedor se sustenta fora dos dados usados para escolhê-lo e expõe o
otimismo de seleção (*winner's curse*): entre muitos candidatos ruidosos, o
maior resultado de validação tende a superestimar o desempenho verdadeiro.

---

## Etapa 2 — Replicação no TT100K

*Etapa não iniciada. As tabelas serão preenchidas quando a etapa começar.*

O [TT100K (Tsinghua-Tencent 100K)](https://cg.cs.tsinghua.edu.cn/traffic-sign/)
([Zhu et al., 2016](https://doi.org/10.1109/CVPR.2016.232)) contém cerca de
100 mil imagens de *street view* em alta resolução (2048 × 2048). Cerca de 10
mil têm placas anotadas — um pool ~10× maior que o do BVTSLD, com placas
pequenas em cenas complexas. O objetivo é verificar se o ranking dos métodos
do teste preliminar se mantém em escala.

O protocolo é o mesmo da Etapa 1: auditoria e taxonomia, partições fixas,
oráculo, 6 métodos × 4 frações, 8 repetições para métodos estocásticos e 1 para
o OPF, além de 2 sementes de treino, com mAP registrado para todos os métodos e
frações. O OPF opera sobre o pool completo também nesta escala; não há amostra
aleatória intermediária. São 164 seleções e 328 runs.

Escalar para o TT100K reduz o risco de apenas ~120 atualizações nas menores
seleções do BVTSLD, porque 5% do pool já contém centenas de imagens. Isso não
elimina completamente o confundidor: com 40 épocas fixas, o número de passos
ainda cresce linearmente com a fração. Por isso, antes de iniciar a grade do
TT100K, a política congelada deve ser uma destas duas: manter 40 épocas e
interpretar a curva como desempenho sob orçamento computacional crescente, ou
fixar o número de atualizações por run para isolar melhor o efeito da seleção.
Essa decisão será tomada antes do primeiro treino da Etapa 2 e não será
alterada depois de observar resultados.

A grade completa não deve começar no M2 Pro. O pré-requisito operacional da
Etapa 2 é reservar uma GPU CUDA dedicada, espaço para checkpoints e uma janela
de execução retomável. Antes de reservar a grade inteira, serão cronometrados
o oráculo e um run de 5% e 50% no hardware definitivo; esses três tempos
extrapolam o custo total de 328 runs. O piloto estima custo e capacidade, mas
não elimina método, fração ou repetição da grade congelada.

### Conjunto de dados e partições fixas — TT100K

| Resultado | Valor |
|---|---:|
| Imagens elegíveis | — |
| *Bounding boxes* | — |
| Classes-alvo | — |
| Pool de treino | — |
| Partição de validação | — |
| Partição de teste | — |

### Oráculo YOLOv8n — TT100K

| Partição | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| Validação | — | — |

### Treino YOLO por seleção — mAP vs. oráculo — TT100K

Cada célula reporta mAP@0.5 / mAP@0.5:0.95 na validação. Referência: oráculo
com 100% do pool: — / —.

| Método | 5% | 10% | 20% | 50% |
|---|---:|---:|---:|---:|
| `random` | — / — | — / — | — / — | — / — |
| `kmeans_dinov2` | — / — | — / — | — / — | — / — |
| `opf_dinov2` | — / — | — / — | — / — | — / — |
| `typiclust_dinov2` | — / — | — / — | — / — | — / — |
| `probcover_dinov2` | — / — | — / — | — / — | — / — |
| `freesel_dino` | — / — | — / — | — / — | — / — |

---

## Etapa 3 — Trabalhos futuros: semi-supervisão na dissertação

Com os resultados das Etapas 1 e 2, a seleção vencedora define o conjunto
rotulado inicial. O restante do pool entra sem rótulos, por meio de
*pseudo-labels*.

Uma distinção importante: FixMatch, FreeMatch e SoftMatch foram propostos para
**classificação**. Em detecção, o *pseudo-label* é um conjunto de *bounding
boxes* filtradas por confiança e NMS, e a linha de pesquisa própria da área —
Unbiased Teacher ([Liu et al., 2021](https://arxiv.org/abs/2102.09480)), Soft
Teacher ([Xu et al., 2021](https://arxiv.org/abs/2106.09018)) e, para a
família YOLO, Efficient Teacher ([Xu et al., 2023](https://arxiv.org/abs/2302.07577))
— usa um par professor–aluno com EMA para gerar e consumir esses rótulos. O
plano da dissertação é adotar essa estrutura professor–aluno e comparar, dentro
dela, três estratégias de filtragem dos *pseudo-labels*, derivadas dos métodos
de classificação:

- **Limiar fixo** (estilo FixMatch, [Sohn et al., 2020](https://arxiv.org/abs/2001.07685)):
  a predição do professor na visão com *weak augmentation* vira *pseudo-label*
  quando a confiança supera um limiar fixo (`τ = 0,95`) e supervisiona o aluno
  na visão com *strong augmentation*.

- **Limiares adaptativos por classe** (estilo FreeMatch,
  [Wang et al., 2023](https://arxiv.org/abs/2205.07246)): limiares globais e
  por classe estimados a partir da confiança do próprio modelo. Relevante
  quando as classes têm frequências muito diferentes, como `information` no
  BVTSLD.

- **Pesos contínuos de confiança** (estilo SoftMatch,
  [Chen et al., 2023](https://arxiv.org/abs/2301.10921)): substitui o corte
  binário por um peso gaussiano centrado na confiança média, para equilibrar
  quantidade e qualidade dos *pseudo-labels*.

---

## Apêndice

### Termos

| Termo | O que é |
|---|---|
| **pool** | as imagens de treino disponíveis; fingimos que nenhuma tem rótulo |
| **fração de rótulos** | quanto do pool ganha rótulo manual: 5, 10, 20 ou 50% |
| **seleção** | o subconjunto de imagens escolhido para receber rótulo manual |
| **embedding** | um vetor de números que resume o conteúdo de uma imagem, gerado por uma rede pronta (não precisa de rótulo para calcular) |
| **agrupamento (clustering)** | juntar imagens de embedding parecido em grupos ("cenas de rodovia", "ruas à noite"...) |
| **cobertura DINOv2** | distância média de cada imagem do pool à imagem selecionada mais parecida no DINOv2 global — diagnóstico interno desse espaço, não critério de ranking entre métodos |
| **oráculo** | YOLO treinado com 100% dos rótulos — o teto de referência |
| **instância de seleção** | uma execução independente da técnica, identificada por uma semente de seleção; é a unidade de comparação com o sorteio |
| **semente de treino** | inicialização e aleatoriedade do YOLO; cada instância de seleção é treinada com as mesmas 2 sementes em todas as técnicas |

### Protocolo fixo de treino YOLO

| Item | Configuração fixa |
|---|---|
| Modelo e treino | YOLOv8n pré-treinado no COCO; 640 px; 40 épocas; SGD; *batch* 16; `patience=0`; determinístico; sementes 41 e 42 |
| Aumentações | HSV `(0.015, 0.7, 0.4)`; translação `0.1`; escala `0.5`; espelhamento `0.5`; *mosaic* `1.0` desligado nas 10 épocas finais; *erasing* `0.4` |
| Decisão | mAP@0.5:0.95 de validação; comparação pareada com `random`; ganho mínimo relevante de `0.02` |
| Inferência estatística | Ganho médio pareado, IC 95% por *bootstrap* hierárquico, teste exato de sinais e correção de Holm |

### Reprodução

```bash
git clone <URL_DO_REPOSITORIO>
cd wvc2026-deteccao-placas
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

O BVTSLD não é versionado. Coloque o dataset em:

```text
datasets/bvtsld/Brazilian Vertical Traffic Signs and Lights Dataset/
```

Os resultados compactos e as 164 seleções estão no Git. Imagens brutas,
embeddings, datasets YOLO materializados, checkpoints e execuções de treino
permanecem locais. Para compartilhar checkpoints, use um release ou um
armazenamento de artefatos, não o histórico do repositório.

O ambiente deve usar as versões de `requirements.txt`, incluindo
`ultralytics==8.3.0`:

```bash
.venv/bin/python scripts/generate_embeddings.py --verify --sample 32
.venv/bin/python scripts/validate_bvtsld.py
.venv/bin/python scripts/run_local_triage.py --dry-run
.venv/bin/python scripts/run_local_triage.py --smoke
.venv/bin/python scripts/run_local_triage.py
```

O treinador é retomável: cada run concluído entra em
`outputs/bvtsld/triage_results.csv` e não é repetido. Para executar só uma
parte da grade:

```bash
.venv/bin/python scripts/run_local_triage.py \
  --technique typiclust_dinov2 --fraction 0.10 --repeat 1 --train-seed 42
```

Em um clone que ainda não tenha os artefatos locais, gere as duas
representações antes das seleções:

```bash
.venv/bin/python scripts/generate_embeddings.py --dataset bvtsld
.venv/bin/python scripts/generate_bvtsld_local_selections.py
```

Após completar os 328 runs:

```bash
.venv/bin/python scripts/analyze_triage.py outputs/bvtsld/triage_results.csv \
  --output outputs/bvtsld/triage_analysis.csv
.venv/bin/python scripts/summarize_metrics.py
```

O primeiro comando gera a análise estatística pareada. O segundo agrega as
médias por técnica × fração (qualidade, tempo e consumo computacional) em
`outputs/bvtsld/metrics_summary.csv`.

### Layout dos artefatos

```text
README.md                                visão geral, protocolo e resultados
requirements.txt                         dependências Python fixadas
scripts/                                 auditoria, seleção, treino e análise
figs/                                    figuras de publicação
outputs/bvtsld/records.json              anotações limpas
outputs/bvtsld/split.json                partições fixas pool/validação/teste
outputs/bvtsld/selections/*.json         164 seleções
outputs/bvtsld/selections_summary.csv    cobertura, estabilidade e tempos
outputs/bvtsld/oracle_results.json       protocolo e métricas do oráculo
outputs/bvtsld/triage_results.csv        métricas de treino por run (quando gerado)
outputs/bvtsld/metrics_summary.csv       médias por técnica x fração (quando gerado)
datasets/bvtsld/                         dataset bruto (fora do Git)
outputs/bvtsld/embeddings_*.npy          embeddings congelados (fora do Git)
outputs/bvtsld/yolo_bvtsld/              dataset YOLO materializado (fora do Git)
outputs/bvtsld/runs/                     checkpoints e execuções de treino (fora do Git)
```

### Referências

- Caron, M. et al. (2021). *Emerging Properties in Self-Supervised Vision
  Transformers* (DINO). ICCV. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- Chen, H. et al. (2023). *SoftMatch: Addressing the Quantity-Quality
  Trade-off in Semi-supervised Learning*. ICLR.
  [arXiv:2301.10921](https://arxiv.org/abs/2301.10921)
- Hacohen, G., Dekel, A. & Weinshall, D. (2022). *Active Learning on a Budget:
  Opposite Strategies Suit High and Low Budgets* (TypiClust). ICML.
  [arXiv:2202.02794](https://arxiv.org/abs/2202.02794)
- Liu, Y.-C. et al. (2021). *Unbiased Teacher for Semi-Supervised Object
  Detection*. ICLR. [arXiv:2102.09480](https://arxiv.org/abs/2102.09480)
- Lloyd, S. (1982). *Least Squares Quantization in PCM* (k-means). IEEE
  Transactions on Information Theory.
  [DOI:10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489)
- Oquab, M. et al. (2024). *DINOv2: Learning Robust Visual Features without
  Supervision*. TMLR. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Rocha, L. M., Cappabianco, F. A. M. & Falcão, A. X. (2009). *Data Clustering
  as an Optimum-Path Forest Problem with Applications in Image Analysis*.
  International Journal of Imaging Systems and Technology.
  [DOI:10.1002/ima.20191](https://doi.org/10.1002/ima.20191)
- Sohn, K. et al. (2020). *FixMatch: Simplifying Semi-Supervised Learning with
  Consistency and Confidence*. NeurIPS.
  [arXiv:2001.07685](https://arxiv.org/abs/2001.07685)
- Wang, Y. et al. (2023). *FreeMatch: Self-adaptive Thresholding for
  Semi-supervised Learning*. ICLR.
  [arXiv:2205.07246](https://arxiv.org/abs/2205.07246)
- Xie, Y. et al. (2023). *Towards Free Data Selection with General-Purpose
  Models* (FreeSel). NeurIPS.
  [arXiv:2309.17342](https://arxiv.org/abs/2309.17342)
- Xu, B. et al. (2023). *Efficient Teacher: Semi-Supervised Object Detection
  for YOLOv5*. [arXiv:2302.07577](https://arxiv.org/abs/2302.07577)
- Xu, M. et al. (2021). *End-to-End Semi-Supervised Object Detection with Soft
  Teacher*. ICCV. [arXiv:2106.09018](https://arxiv.org/abs/2106.09018)
- Yehuda, O., Dekel, A., Hacohen, G. & Weinshall, D. (2022). *Active Learning
  Through a Covering Lens* (ProbCover). NeurIPS.
  [arXiv:2205.11320](https://arxiv.org/abs/2205.11320)
- Zhu, Z. et al. (2016). *Traffic-Sign Detection and Classification in the
  Wild* (TT100K). CVPR.
  [DOI:10.1109/CVPR.2016.232](https://doi.org/10.1109/CVPR.2016.232)
