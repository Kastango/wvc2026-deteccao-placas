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

O experimento compara o **oráculo**, treinado com 100% do pool rotulado e usado
como referência de dados completos, com os modelos YOLOv8n treinados sobre
cada subconjunto selecionado. Cada método de seleção escolhe 5, 10, 20 ou 50%
do pool, e cada seleção é treinada com duas sementes, 41 e 42. A grade completa
totaliza 328 runs e produz o mAP de cada método, fração de rótulos, repetição e
semente de treino.

## As etapas do estudo

**Etapa 1 — BVTSLD (teste preliminar)** — pool de 889 imagens, duas
classes-alvo (`regulatory` e `traffic_light`). Valida o *pipeline* e compara
os métodos em escala pequena.

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

O BVTSLD (Brazilian Vertical Traffic Signs and Lights Dataset) foi auditado
automaticamente e mapeado para **duas classes-alvo**: `regulatory` (doze
códigos de placas de regulamentação R-*) e `traffic_light` (três focos de
semáforo). Com três classes, poucas dezenas de boxes ficariam fora de
`regulatory` e o mAP macro seria instável nas partições pequenas; com duas, a
razão entre classes é 2,6:1 e todas as partições têm suporte suficiente. A
taxonomia de três classes fica reservada ao TT100K. Único código excluído:
`000025` (A-18, advertência) — as imagens que o contêm ficam em quarentena,
fora de todas as partições.

| Resultado | Valor |
|---|---:|
| Imagens originais elegíveis | 1.271 |
| *Bounding boxes* | 2.007 |
| *Bounding boxes* `regulatory` | 1.442 |
| *Bounding boxes* `traffic_light` | 565 |
| Imagens em quarentena (código `000025`) | 92 |
| Pool de treino | 889 imagens (1.024 / 376 boxes) |
| Partição de validação | 191 imagens (202 / 95 boxes) |
| Partição de teste | 191 imagens (216 / 94 boxes) |
| Vazamento entre partições | 0 imagens |

As partições são fixas e reproduzíveis:
[`generate_split.py`](scripts/generate_split.py) embaralha os IDs elegíveis
com a semente 42 e reserva 15% para validação e 15% para teste. O teste será
aberto uma única vez, na avaliação final; todas as comparações intermediárias
usam apenas a validação.

Fontes: [`records.json`](outputs/bvtsld/records.json),
[`split.json`](outputs/bvtsld/split.json),
[`quarantine.json`](outputs/bvtsld/quarantine.json) e
[`taxonomy_report.json`](outputs/bvtsld/taxonomy_report.json).

### Oráculo YOLOv8n — referência de dados completos

O oráculo foi treinado com 100% do pool no protocolo fixo (ver
[apêndice](#protocolo-fixo-de-treino-yolo)). Apenas a validação foi avaliada.

| Partição | mAP@0.5 | mAP@0.5:0.95 | AP@0.5 `regulatory` | AP@0.5 `traffic_light` |
|---|---:|---:|---:|---:|
| Validação | 0,9365 | 0,6035 | 0,9502 | 0,9228 |

- Tempo de treino: 1.698,7 s (~28,3 min) em Apple M2 Pro/MPS, em processo
  dedicado.
- Checkpoint local: `outputs/bvtsld/runs/oracle/weights/best.pt` (fora do Git).
- Protocolo e métricas: [`oracle_results.json`](outputs/bvtsld/oracle_results.json).

![Curvas de treino do oráculo](figs/oracle_training_curves.png)

*Perdas de treino e validação, precisão, revocação e mAP ao longo das 40
épocas.*

![Curva precisão-revocação do oráculo na validação](figs/oracle_validation_pr_curve.png)

*Curvas de precisão–revocação na validação. O valor agregado é 0,937 mAP@0.5.*

![Matriz de confusão normalizada do oráculo](figs/oracle_validation_confusion_matrix_normalized.png)

*Matriz de confusão normalizada na validação. A coluna `background` mostra
falsos positivos; a linha `background`, falsos negativos.*

![Predições do oráculo em imagens de validação](figs/oracle_validation_predictions.jpg)

*Exemplos de predições do oráculo, com a classe-alvo e a confiança do
detector.*

### Os métodos escolhidos e por quê

Os seis métodos contrastam vieses distintos de seleção — sorteio, agrupamento
global, densidade e conectividade por OPF, densidade local, cobertura e
padrões locais — sem pretender uma taxonomia exaustiva. Regras comuns a todos:

- Nenhum seletor consulta os rótulos do dataset-alvo.
- Salvo indicação em contrário, a representação é o *embedding* DINOv2 da
  imagem inteira (384 dimensões, L2-normalizado).
- Orçamentos: 44 (5%), 89 (10%), 178 (20%) e 445 (50%) imagens por seleção.
- Métodos estocásticos rodam 8 repetições por fração; o `opf_dinov2`,
  determinístico sobre o pool inteiro, roda 1 — repetições produziriam a
  mesma seleção.

**`random`** — sorteia uniformemente as imagens do orçamento, sem
representação. É o *baseline* de controle: o protocolo estatístico mede o
ganho pareado de cada método contra ele.

**`kmeans_dinov2`** ([Lloyd, 1982](https://doi.org/10.1109/TIT.1982.1056489);
DINOv2: [Oquab et al., 2024](https://arxiv.org/abs/2304.07193)) — *baseline*
de **representatividade global**: forma `k = orçamento` grupos com distância
euclidiana e retorna a imagem real mais próxima de cada centroide. Lloyd
fundamenta a alternância entre médias e atribuição ao centro mais próximo; a
escolha de um exemplar real é etapa adicional deste estudo.

**`opf_dinov2`** ([Rocha, Cappabianco & Falcão, 2009](https://doi.org/10.1002/ima.20191))
— agrupa o pool em árvores enraizadas nos máximos de densidade de um grafo
kNN (dissimilaridade `log_squared_euclidean`, padrão do OPFython). Não fixa o
número de grupos igual ao orçamento: com `max_k = 20`, o número de vizinhos é
escolhido pelo menor corte normalizado e o número de árvores resulta do
agrupamento — hipótese complementar ao k-means. **Extensão deste estudo:**
grupos com cota positiva fornecem primeiro a raiz, e o orçamento é completado
com amostras próximas às raízes, em cotas proporcionais ao tamanho dos
grupos. No BVTSLD, o pool inteiro cabe em um único ajuste do algoritmo, o que
torna a seleção determinística.

**`typiclust_dinov2`** ([Hacohen, Dekel & Weinshall, 2022](https://arxiv.org/abs/2202.02794))
— método de referência para **rotulagem com orçamento baixo**: amostras
típicas superam estratégias de incerteza quando há poucos rótulos. Forma
`k = orçamento` grupos e escolhe a imagem de maior densidade local em cada
um. A implementação deste estudo usa distância de cosseno e omite os filtros
de clusters pequenos do apêndice do artigo.

**`probcover_dinov2`** ([Yehuda et al., 2022](https://arxiv.org/abs/2205.11320))
— formula a seleção como **cobertura**: escolhe a imagem que cobre mais
vizinhos ainda não cobertos, dentro de um raio estimado sem rótulos (cosseno).
Estado da arte em *cold-start*, junto com o TypiClust, nos *benchmarks* de
classificação do artigo. Adaptações deste estudo: o raio é estimado com
`k = orçamento` (o artigo usa `k = número de classes`) e a cobertura é
reiniciada quando se esgota.

**`freesel_dino`** ([Xie et al., 2023](https://arxiv.org/abs/2309.17342);
DINO: [Caron et al., 2021](https://arxiv.org/abs/2104.14294)) — único método
que enxerga **regiões locais**: cada imagem gera cinco padrões via k-means
sobre as *features* DINO v1 (384 dimensões), guiado pelos mapas de atenção, e
a seleção busca o padrão não coberto mais distante (cosseno); a imagem dona
do padrão entra. Hipótese associada, a ser testada na grade: placas pequenas
contam mesmo quando a cena inteira já parece representada. A implementação é
a variante determinística FDS; o método principal do artigo usa agrupamento
espectral guiado por atenção e amostragem proporcional à distância².

### Diagnósticos das seleções

São 164 seleções: 6 métodos × 4 frações de rótulos × 8 repetições (OPF: 1).
As tabelas abaixo são diagnósticos **anteriores ao treino YOLO** — cobertura
do pool, estabilidade, *bounding boxes* recuperadas e tempo de seleção. Elas
não definem o ranking.

Como ler as colunas:

- **Cobertura DINOv2**: distância média de cada imagem do pool à imagem
  selecionada mais parecida no espaço DINOv2 global. Menor é melhor.
- **Δ cobertura**: diferença relativa para o `random` na mesma fração.
  Negativo é melhor.
- **Δ pior caso**: mesma comparação usando a maior distância encontrada.
  Negativo é melhor.
- **Jaccard**: sobreposição entre as repetições do método. Mede estabilidade.
- **Tempo (s)**: tempo para gerar todas as repetições da fração (Apple M2
  Pro). Cada técnica × fração roda em um processo dedicado e sequencial, sem
  interferência das demais em tempo, CPU e pico de RSS; CPU e RSS por técnica
  ficam no CSV.

#### Fração de 5% — 44 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `kmeans_dinov2` | 0,2019 | −16,7% | −14,3% | 0,252 | 71,8 | 198,7 |
| `typiclust_dinov2` | 0,2021 | −16,6% | −15,1% | 0,245 | 72,8 | 87,7 |
| `probcover_dinov2` | 0,2081 | −14,1% | −10,3% | 0,435 | 69,2 | 81,2 |
| `random` | 0,2422 | 0,0% | 0,0% | 0,028 | 71,8 | <0,1 |
| `opf_dinov2` | 0,2470 | +2,0% | +8,5% | 1,000¹ | 80,0 | 5,7 |
| `freesel_dino` | 0,2552 | +5,4% | −19,6% | 0,288 | 59,5 | 9,4 |

#### Fração de 10% — 89 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `typiclust_dinov2` | 0,1661 | −17,2% | −15,8% | 0,263 | 143,8 | 149,8 |
| `kmeans_dinov2` | 0,1662 | −17,2% | −17,0% | 0,279 | 142,2 | 347,8 |
| `probcover_dinov2` | 0,1752 | −12,7% | −5,0% | 0,378 | 146,2 | 136,9 |
| `random` | 0,2006 | 0,0% | 0,0% | 0,058 | 142,5 | <0,1 |
| `freesel_dino` | 0,2107 | +5,0% | −20,9% | 0,428 | 124,9 | 14,1 |
| `opf_dinov2` | 0,2181 | +8,7% | +6,1% | 1,000¹ | 167,0 | 5,3 |

#### Fração de 20% — 178 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `kmeans_dinov2` | 0,1220 | −21,6% | −44,4% | 0,404 | 290,9 | 624,0 |
| `typiclust_dinov2` | 0,1224 | −21,3% | −43,6% | 0,382 | 285,1 | 268,0 |
| `probcover_dinov2` | 0,1338 | −14,0% | −17,3% | 0,380 | 289,6 | 317,9 |
| `freesel_dino` | 0,1534 | −1,3% | −32,7% | 0,614 | 261,2 | 35,1 |
| `random` | 0,1555 | 0,0% | 0,0% | 0,114 | 282,8 | <0,1 |
| `opf_dinov2` | 0,1798 | +15,7% | −3,2% | 1,000¹ | 314,0 | 5,7 |

#### Fração de 50% — 445 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `kmeans_dinov2` | 0,0524 | −33,3% | −65,6% | 0,697 | 710,9 | 1.753,9 |
| `typiclust_dinov2` | 0,0524 | −33,2% | −65,1% | 0,697 | 707,1 | 823,8 |
| `probcover_dinov2` | 0,0675 | −14,0% | −65,1% | 0,755 | 721,1 | 797,7 |
| `freesel_dino` | 0,0700 | −10,8% | −32,1% | 0,853 | 665,2 | 78,8 |
| `random` | 0,0785 | 0,0% | 0,0% | 0,336 | 698,9 | <0,1 |
| `opf_dinov2` | 0,1103 | +40,6% | +17,0% | 1,000¹ | 754,0 | 6,7 |

¹ Repetição única: o OPF é determinístico neste pool (ver
[métodos](#os-métodos-escolhidos-e-por-quê)).

Leitura preliminar: `kmeans_dinov2` e `typiclust_dinov2` têm a melhor
cobertura média nas quatro frações; `probcover_dinov2` vem próximo. O
`opf_dinov2` cobre pior que o `random`, mas recupera mais *bounding boxes*.
Isso **não sustenta um ranking**: k-means, TypiClust e ProbCover otimizam o
mesmo espaço DINOv2 em que a cobertura é medida, então a métrica é um
diagnóstico interno, não evidência comparativa de qualidade. O ranking virá
exclusivamente do mAP da grade de treino.

Fonte completa: [`selections_summary.csv`](outputs/bvtsld/selections_summary.csv).
As seleções individuais estão em
[`outputs/bvtsld/selections/`](outputs/bvtsld/selections/).

### Como cada método enxerga o pool

![Comparação dos espaços de representação dos seis métodos de seleção](figs/methods_selection_spaces_bvtsld_tsne_frac10_rep1.png)

*Uma seleção real por método (fração de 10%): pontos cinza são o pool no
espaço de representação do método; pontos azuis, as 89 imagens selecionadas.
No `FreeSel`, destaca-se apenas o padrão local que motivou cada escolha; no
`random`, sem embedding, a grade de índices é arbitrária. O t-SNE serve só
para visualização — a seleção opera nas 384 dimensões originais.*

### Treino YOLO por seleção — mAP vs. oráculo

O protocolo usa 8 repetições por método: é o mínimo que permite significância
no teste exato com correção de Holm ($p$ mínimo corrigido 0,039; com 7 seria
0,078). Precedentes da área usam menos — 5 em
[Munjal et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Munjal_Towards_Robust_and_Reproducible_Active_Learning_Using_Neural_Networks_CVPR_2022_paper.html)
e 3 no [FreeSel](https://openreview.net/forum?id=KBXcDAaZE7) —, mas só para
análise descritiva, sem teste de hipótese
([Colas et al., 2018](https://doi.org/10.48550/arXiv.1806.08295)).

A grade contém **328 runs**: 164 seleções × 2 sementes de treino (41 e 42),
40 épocas cada, no mesmo protocolo do oráculo. Cada run registra em
[`triage_results.csv`](outputs/bvtsld/triage_results.csv):

- **Qualidade**: precisão, revocação, F1, mAP@0.5, mAP@0.75, mAP@0.5:0.95 e
  AP@0.5 por classe na validação. A AP por classe separa o efeito da
  composição da seleção (quantas imagens com `traffic_light` entraram) da
  qualidade geral do detector.
- **Tempo**: tempo de treino, tempo de validação, inferência média por imagem
  (ms) e tempo de CPU (usuário + sistema) do run.
- **Consumo computacional**: pico de RAM do processo e memória média/de pico
  da GPU. A utilização da GPU (%) só é registrada em CUDA (o macOS não a expõe
  sem privilégios de administrador). Na execução definitiva, `--isolate` roda
  um run por vez, cada um em um processo novo: tempo de CPU e pico de RSS
  pertencem só àquele run.

A fase de seleção tem registro próprio: o
[`selections_summary.csv`](outputs/bvtsld/selections_summary.csv) guarda, por
técnica × fração, tempo de seleção, RAM e cobertura.
[`summarize_metrics.py`](scripts/summarize_metrics.py) cruza os dois CSVs e
gera o [`metrics_summary.csv`](outputs/bvtsld/metrics_summary.csv) — uma linha
por técnica × fração com média e desvio, fonte das tabelas deste README.

A tabela abaixo será preenchida com a média sobre repetições e sementes. Cada
célula reporta mAP@0.5 / mAP@0.5:0.95 na validação.

Referência: o oráculo, treinado com 100% do pool, atinge **0,9365 / 0,6035**.

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

O treino de verificação tem duas épocas e confirma o pipeline de ponta a
ponta; seu mAP não é resultado experimental. Após a grade completa,
[`analyze_triage.py`](scripts/analyze_triage.py) calcula o ganho médio pareado
contra o `random`, o IC 95% por *bootstrap* hierárquico, o teste exato de
sinais e a correção de Holm. O estado auditável está em
[`project_status.json`](outputs/bvtsld/project_status.json).

Depois de escolher a fração na validação, a abertura única do teste avaliará
**todos os métodos nessa fração**, não só o vencedor, e a escolha não será
refeita. Isso expõe o otimismo de seleção (*winner's curse*): entre candidatos
ruidosos, o melhor resultado de validação tende a superestimar o desempenho
verdadeiro.

---

## Etapa 2 — Replicação no TT100K

*Etapa não iniciada. As tabelas serão preenchidas quando a etapa começar.*

O [TT100K (Tsinghua-Tencent 100K)](https://cg.cs.tsinghua.edu.cn/traffic-sign/)
([Zhu et al., 2016](https://doi.org/10.1109/CVPR.2016.232)) contém cerca de
100 mil imagens de *street view* em alta resolução (2048 × 2048). Cerca de 10
mil têm placas anotadas — um pool ~10× maior que o do BVTSLD, com placas
pequenas em cenas complexas. O objetivo é verificar se o ranking dos métodos
do teste preliminar se mantém em escala.

O protocolo repete a Etapa 1 com **três classes-alvo** — regulamentação
(`p*`), advertência (`w*`) e indicação (`i*`), agregadas dos códigos originais
—, viáveis nessa escala porque cada classe tem centenas a milhares de boxes
por partição. São as mesmas 164 seleções e 328 runs; o OPF continua operando
sobre o pool completo, sem amostra intermediária.

Com 40 épocas fixas, o número de atualizações cresce com a fração — nas
menores seleções do BVTSLD são só ~120 atualizações. O TT100K atenua o
confundidor (5% do pool já são centenas de imagens), mas não o elimina. Antes
do primeiro treino da Etapa 2 será congelada uma de duas políticas: manter 40
épocas (curva sob orçamento computacional crescente) ou fixar o número de
atualizações por run (isola melhor o efeito da seleção).

A grade exige GPU CUDA dedicada, espaço para checkpoints e execução
retomável — não o M2 Pro. Um piloto no hardware definitivo (oráculo + um run
de 5% e um de 50%) extrapolará o custo dos 328 runs, sem eliminar método,
fração ou repetição da grade congelada.

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

FixMatch, FreeMatch e SoftMatch foram propostos para **classificação**. Em
detecção, o *pseudo-label* é um conjunto de *bounding boxes* filtradas por
confiança e NMS, e a linha própria da área — Unbiased Teacher
([Liu et al., 2021](https://arxiv.org/abs/2102.09480)), Soft Teacher
([Xu et al., 2021](https://arxiv.org/abs/2106.09018)) e, para YOLO, Efficient
Teacher ([Xu et al., 2023](https://arxiv.org/abs/2302.07577)) — usa um par
professor–aluno com EMA. O plano é adotar essa estrutura e comparar, dentro
dela, três estratégias de filtragem de *pseudo-labels*:

- **Limiar fixo** (estilo FixMatch, [Sohn et al., 2020](https://arxiv.org/abs/2001.07685)):
  a predição do professor na visão com *weak augmentation* vira *pseudo-label*
  quando a confiança supera um limiar fixo (`τ = 0,95`) e supervisiona o aluno
  na visão com *strong augmentation*.

- **Limiares adaptativos por classe** (estilo FreeMatch,
  [Wang et al., 2023](https://arxiv.org/abs/2205.07246)): limiares globais e
  por classe estimados a partir da confiança do próprio modelo. Relevante
  quando as classes têm frequências muito diferentes, como nas classes raras
  do TT100K.

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
| **oráculo** | YOLO treinado com 100% dos rótulos, usado como referência de dados completos |
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

#### 1. Ambiente e dados

```bash
git clone https://github.com/Kastango/wvc2026-deteccao-placas.git
cd wvc2026-deteccao-placas
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Os datasets não são versionados. O script abaixo baixa das fontes oficiais,
retoma downloads interrompidos, confere o tamanho congelado e extrai os ZIPs
com segurança. Revise as licenças antes de confirmar:

```bash
# BVTSLD v2 (~3,86 GiB; CC BY 4.0)
.venv/bin/python scripts/download_datasets.py \
  --dataset bvtsld --accept-license

# TT100K 2016 (~17,84 GiB; CC BY-NC; requer ao menos 100 GiB livres)
.venv/bin/python scripts/download_datasets.py \
  --dataset tt100k --accept-license
```

Os destinos são `datasets/bvtsld/Brazilian Vertical Traffic Signs and Lights
Dataset/` e `datasets/tt100k/data/`. Para apenas verificar as URLs e os tamanhos
sem baixar, use `--dataset all --check`.

Na primeira execução, é necessário acesso à internet para baixar os pesos
DINO/DINOv2 e YOLOv8n. Imagens brutas, embeddings, dataset YOLO, checkpoints e
runs permanecem locais; os resultados compactos e as 164 seleções estão no Git.

#### 2. Reprodução rápida

O fluxo rápido materializa o dataset YOLO, verifica as representações,
reutiliza as seleções versionadas, roda um treino de duas épocas e audita os
artefatos — verificação de ponta a ponta, sem resultado experimental:

```bash
.venv/bin/python scripts/reproduce.py \
  --stage quick --accept-dataset-licenses
```

Para apenas conferir os comandos, sem executar nada:

```bash
.venv/bin/python scripts/reproduce.py --stage quick --dry-run
```

#### 3. Reprodução completa

O comando abaixo inclui o fluxo rápido, o retreino do oráculo, os 328 treinos
comparativos e a análise estatística. É uma execução longa, indicada para GPU
CUDA dedicada:

```bash
.venv/bin/python scripts/reproduce.py \
  --stage all --device cuda --accept-dataset-licenses
```

Todos os comandos aceitam `--dataset bvtsld` (padrão) ou `--dataset tt100k`.
As classes-alvo, caminhos e o mapa de códigos de cada dataset ficam
centralizados em [`dataset_config.py`](scripts/dataset_config.py): o BVTSLD usa
duas classes e o TT100K usará as três classes agregadas por prefixo.

O treinador é retomável: cada run concluído entra em `triage_results.csv` e
não é repetido. As etapas também podem ser chamadas separadamente:

| Etapa | Comando | Saída principal |
|---|---|---|
| Download BVTSLD | `--stage download` | `datasets/bvtsld/` |
| Partições fixas | `--stage split` | `outputs/bvtsld/split.json` |
| Preparar YOLO | `--stage prepare` | `outputs/bvtsld/yolo_bvtsld/` |
| Representações | `--stage embeddings` | embeddings DINOv2 e padrões FreeSel |
| Seleções | `--stage selections` | 164 JSONs e `selections_summary.csv` |
| Oráculo | `--stage oracle` | `oracle_results.json` e checkpoint local |
| Verificação curta | `--stage smoke` | `triage_smoke.csv` e checkpoint local |
| Auditoria | `--stage audit` | `local_pretrain_audit.json` |
| Validar artefatos | `--stage verify` | `project_status.json` |
| Grade completa | `--stage train` | `triage_results.csv` e checkpoints |
| Análise | `--stage analyze` | `triage_analysis.csv` e `metrics_summary.csv` |

Para a coleta definitiva de tempo de CPU e pico de RAM, execute a grade com
isolamento por processo:

```bash
.venv/bin/python scripts/run_local_triage.py --isolate --device cuda
```

O comando percorre somente as células ainda pendentes e executa uma por vez.
Cada célula roda em um processo Python novo; se uma delas falhar, a grade para
sem marcar esse run como concluído e pode ser retomada pelo mesmo comando.

Exemplo para executar somente uma célula da grade:

```bash
.venv/bin/python scripts/run_local_triage.py \
  --technique typiclust_dinov2 --fraction 0.10 --repeat 1 --train-seed 42
```

Use `--force` no orquestrador somente para regenerar artefatos locais já
existentes. A geração de seleções arquiva a versão anterior antes de escrever.

#### 4. Notebook de resultados

O notebook [01_results_bvtsld.ipynb](notebooks/01_results_bvtsld.ipynb) só lê
os CSVs e gera tabelas e gráficos; não contém lógica de seleção nem controla
treinos. Para abri-lo:

```bash
.venv/bin/pip install jupyterlab
.venv/bin/jupyter lab notebooks/01_results_bvtsld.ipynb
```

### Layout dos artefatos

```text
README.md                                visão geral, protocolo e resultados
requirements.txt                         dependências Python fixadas
scripts/                                 auditoria, seleção, treino e análise
notebooks/01_results_bvtsld.ipynb        exploração dos resultados gerados
figs/                                    figuras de publicação
outputs/bvtsld/records.json              anotações limpas (2 classes)
outputs/bvtsld/split.json                partições fixas pool/validação/teste
outputs/bvtsld/taxonomy_report.json      auditoria automática e mapa congelado
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
- Colas, C., Sigaud, O. & Oudeyer, P.-Y. (2018). *How Many Random Seeds?
  Statistical Power Analysis in Deep Reinforcement Learning Experiments*.
  [arXiv:1806.08295](https://arxiv.org/abs/1806.08295)
- Hacohen, G., Dekel, A. & Weinshall, D. (2022). *Active Learning on a Budget:
  Opposite Strategies Suit High and Low Budgets* (TypiClust). ICML.
  [arXiv:2202.02794](https://arxiv.org/abs/2202.02794)
- Liu, Y.-C. et al. (2021). *Unbiased Teacher for Semi-Supervised Object
  Detection*. ICLR. [arXiv:2102.09480](https://arxiv.org/abs/2102.09480)
- Lloyd, S. (1982). *Least Squares Quantization in PCM* (k-means). IEEE
  Transactions on Information Theory.
  [DOI:10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489)
- Munjal, P., Hayat, N., Hayat, M., Sourati, J. & Khan, S. (2022). *Towards
  Robust and Reproducible Active Learning Using Neural Networks*. CVPR.
  [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Munjal_Towards_Robust_and_Reproducible_Active_Learning_Using_Neural_Networks_CVPR_2022_paper.html)
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
