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

**Etapa 2 — TT100K (replicação em escala)** — verifica se o ranking dos
métodos se mantém.

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
mapeado para duas classes-alvo: `regulatory` (doze códigos R-*) e
`traffic_light` (três focos de semáforo). O código `000025` (placa de
advertência A-18), fora dessa taxonomia, foi excluído; suas 92 imagens
permanecem em quarentena.

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

As partições são fixas (semente 42; 70/15/15%). O teste será usado uma única
vez, na avaliação final; até lá, as comparações usam apenas a validação.

Fontes: [`records.json`](outputs/bvtsld/records.json),
[`split.json`](outputs/bvtsld/split.json),
[`quarantine.json`](outputs/bvtsld/quarantine.json) e
[`taxonomy_report.json`](outputs/bvtsld/taxonomy_report.json).

### Oráculo YOLOv8n — referência de dados completos

O oráculo foi treinado com 100% do pool no
[protocolo fixo](#protocolo-fixo-de-treino-yolo) e avaliado na validação.

| Partição | mAP@0.5 | mAP@0.5:0.95 | AP@0.5 `regulatory` | AP@0.5 `traffic_light` |
|---|---:|---:|---:|---:|
| Validação | 0,9365 | 0,6035 | 0,9502 | 0,9228 |

O treino levou 1.698,7 s (~28,3 min) no Apple M2 Pro/MPS. Protocolo e métricas
estão em [`oracle_results.json`](outputs/bvtsld/oracle_results.json);

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

Os seis métodos representam estratégias distintas: sorteio, agrupamento
global, densidade, cobertura e padrões locais. Todos seguem as mesmas regras:

- Nenhum seletor consulta os rótulos do dataset-alvo.
- Salvo indicação em contrário, usam o modelo pequeno do DINOv2 para
  representar a imagem inteira (384 dimensões, L2-normalizado).
- Orçamentos: 44 (5%), 89 (10%), 178 (20%) e 445 (50%) imagens por seleção.
- Métodos estocásticos rodam 8 repetições por fração; o OPF, determinístico,
  roda uma vez.

**`random`** — sorteio uniforme usado como controle.

**`kmeans_dinov2`** ([Lloyd, 1982](https://doi.org/10.1109/TIT.1982.1056489);
DINOv2: [Oquab et al., 2024](https://arxiv.org/abs/2304.07193)) — *baseline*
de **representatividade global**: forma `k = orçamento` grupos e escolhe a
imagem mais próxima de cada centroide.

**`opf_dinov2`** ([Rocha, Cappabianco & Falcão, 2009](https://doi.org/10.1002/ima.20191);
implementação: [de Rosa & Papa, 2021](https://doi.org/10.1016/j.simpa.2021.100113))
— cria árvores a partir dos máximos de densidade de um grafo kNN. O
OPFython 1.0.12 usa distância `log_squared_euclidean` e busca, até `k = 20`,
o menor *normalized cut*. O orçamento é distribuído entre os grupos: cada um
fornece sua raiz e, depois, amostras próximas. A seleção é determinística.

**`typiclust_dinov2`** ([Hacohen, Dekel & Weinshall, 2022](https://arxiv.org/abs/2202.02794))
— forma `k = orçamento` grupos e escolhe a imagem de maior densidade local em
cada um. A implementação usa distância de cosseno e não aplica os filtros de
grupos pequenos do artigo.

**`probcover_dinov2`** ([Yehuda et al., 2022](https://arxiv.org/abs/2205.11320))
— escolhe iterativamente a imagem que cobre mais vizinhos ainda não cobertos.
O raio é estimado sem rótulos com `k = orçamento`, em vez do número de classes
usado no artigo, e a cobertura reinicia quando necessário.

**`freesel_fds_dino`** ([Xie et al., 2023](https://arxiv.org/abs/2309.17342);
DINO: [Caron et al., 2021](https://arxiv.org/abs/2104.14294)) — único método
comparado que usa **regiões locais**. Cada imagem gera cinco padrões a partir das
*features* DINO guiadas por atenção; a variante FDS seleciona sucessivamente o
padrão mais distante e inclui sua imagem. Essa escolha testa se regiões
pequenas podem ser informativas mesmo em cenas globalmente semelhantes.

### Diagnósticos das seleções

São 164 seleções: cinco métodos estocásticos × 4 frações × 8 repetições, mais
o OPF × 4 frações × 1 execução. As tabelas descrevem as seleções **antes do
treino YOLO** e não definem o ranking.

Como ler as colunas:

- **Cobertura DINOv2**: distância média de cada imagem do pool à selecionada
  mais próxima no espaço DINOv2 global. Menor é melhor.
- **Δ cobertura**: diferença relativa para o `random` na mesma fração;
  valores negativos são melhores.
- **Δ pior caso**: mesma comparação para a maior distância encontrada;
  valores negativos são melhores.
- **Jaccard**: sobreposição média entre repetições. Sem duas seleções, é `N/A`.
- ***Bounding boxes***: média de caixas presentes nas imagens selecionadas;
  é apenas um diagnóstico, pois os métodos não consultam os rótulos.
- **Tempo (s)**: tempo total das repetições daquela fração no Apple M2 Pro.
  Cada técnica e fração roda isoladamente; CPU e pico de RSS estão no CSV.

#### Fração de 5% — 44 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `kmeans_dinov2` | 0,2019 | −16,7% | −14,3% | 0,252 | 71,8 | 198,7 |
| `typiclust_dinov2` | 0,2021 | −16,6% | −15,1% | 0,245 | 72,8 | 87,7 |
| `probcover_dinov2` | 0,2081 | −14,1% | −10,3% | 0,435 | 69,2 | 81,2 |
| `random` | 0,2422 | 0,0% | 0,0% | 0,028 | 71,8 | <0,1 |
| `opf_dinov2` | 0,2470 | +2,0% | +8,5% | N/A¹ | 80,0 | 5,7 |
| `freesel_fds_dino` | 0,2552 | +5,4% | −19,6% | 0,288 | 59,5 | 9,4 |

#### Fração de 10% — 89 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `typiclust_dinov2` | 0,1661 | −17,2% | −15,8% | 0,263 | 143,8 | 149,8 |
| `kmeans_dinov2` | 0,1662 | −17,2% | −17,0% | 0,279 | 142,2 | 347,8 |
| `probcover_dinov2` | 0,1752 | −12,7% | −5,0% | 0,378 | 146,2 | 136,9 |
| `random` | 0,2006 | 0,0% | 0,0% | 0,058 | 142,5 | <0,1 |
| `freesel_fds_dino` | 0,2107 | +5,0% | −20,9% | 0,428 | 124,9 | 14,1 |
| `opf_dinov2` | 0,2181 | +8,7% | +6,1% | N/A¹ | 167,0 | 5,3 |

#### Fração de 20% — 178 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `kmeans_dinov2` | 0,1220 | −21,6% | −44,4% | 0,404 | 290,9 | 624,0 |
| `typiclust_dinov2` | 0,1224 | −21,3% | −43,6% | 0,382 | 285,1 | 268,0 |
| `probcover_dinov2` | 0,1338 | −14,0% | −17,3% | 0,380 | 289,6 | 317,9 |
| `freesel_fds_dino` | 0,1534 | −1,3% | −32,7% | 0,614 | 261,2 | 35,1 |
| `random` | 0,1555 | 0,0% | 0,0% | 0,114 | 282,8 | <0,1 |
| `opf_dinov2` | 0,1798 | +15,7% | −3,2% | N/A¹ | 314,0 | 5,7 |

#### Fração de 50% — 445 imagens por seleção

| Método | Cobertura DINOv2 | Δ cobertura | Δ pior caso | Jaccard | *Bounding boxes* | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| `kmeans_dinov2` | 0,0524 | −33,3% | −65,6% | 0,697 | 710,9 | 1.753,9 |
| `typiclust_dinov2` | 0,0524 | −33,2% | −65,1% | 0,697 | 707,1 | 823,8 |
| `probcover_dinov2` | 0,0675 | −14,0% | −65,1% | 0,755 | 721,1 | 797,7 |
| `freesel_fds_dino` | 0,0700 | −10,8% | −32,1% | 0,853 | 665,2 | 78,8 |
| `random` | 0,0785 | 0,0% | 0,0% | 0,336 | 698,9 | <0,1 |
| `opf_dinov2` | 0,1103 | +40,6% | +17,0% | N/A¹ | 754,0 | 6,7 |

¹ Jaccard não se aplica ao OPF: há uma única seleção por fração e, portanto,
nenhum par de repetições cuja sobreposição possa ser medida.

K-means e TypiClust apresentam a melhor cobertura média nas quatro frações,
seguidos por ProbCover. O OPF cobre pior que o `random`, mas recupera mais
*bounding boxes*. Como parte dos métodos otimiza o próprio espaço DINOv2 usado
nessa medida, o ranking será definido apenas pelo mAP após o treino.

Fonte completa: [`selections_summary.csv`](outputs/bvtsld/selections_summary.csv).
As seleções individuais estão em
[`outputs/bvtsld/selections/`](outputs/bvtsld/selections/).

### Como cada método enxerga o pool

![Comparação dos espaços de representação dos seis métodos de seleção](figs/methods_selection_spaces_bvtsld_tsne_frac10_rep1.png)

*Repetição 1 da fração de 10% (89 imagens). Cada painel projeta o espaço usado
pelo método; cinza representa o pool e azul, a seleção. O t-SNE é apenas uma
visualização: a seleção ocorre nos espaços originais.*

### Treino YOLO por seleção — mAP vs. oráculo

Cada método que pode gerar seleções diferentes entre execuções é avaliado em
8 repetições por fração. O uso de múltiplas execuções também aparece em
avaliações anteriores: Munjal et al. usam 5 inicializações e o FreeSel, 3
seleções independentes
([Munjal et al., 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Munjal_Towards_Robust_and_Reproducible_Active_Learning_Using_Neural_Networks_CVPR_2022_paper.html);
[Xie et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/047682108c3b053c61ad2da5a6057b4e-Abstract-Conference.html)).

K-means, TypiClust, ProbCover e FreeSel são comparados ao `random` por
repetição. Como o OPF produz uma única seleção por fração, sua comparação é
apenas descritiva, sem intervalo de confiança ou valor-p.

A grade contém **328 runs**: 164 seleções × 2 sementes de treino, com 40
épocas cada. O [`triage_results.csv`](outputs/bvtsld/triage_results.csv)
registra métricas de detecção, tempo e uso de memória. O
[`metrics_summary.csv`](outputs/bvtsld/metrics_summary.csv) reunirá as médias
por método e fração.

As métricas registradas são:

- **Qualidade**: precisão, revocação, F1, mAP@0.5, mAP@0.75, mAP@0.5:0.95 e
  AP@0.5 por classe. A AP por classe ajuda a distinguir a composição da
  seleção do desempenho geral do detector.
- **Tempo**: duração do treino e da validação, inferência média por imagem e
  tempo de CPU.
- **Consumo computacional**: pico de RAM e memória média e máxima da GPU.
  Cada run é executado em um processo isolado para que as medidas pertençam
  somente àquela execução.

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
| `freesel_fds_dino` | — / — | — / — | — / — | — / — |

Estado atual da etapa de treino:

| Item | Estado |
|---|---:|
| Seleções salvas | 164/164 |
| Configurações de treino materializadas | 164/164 |
| Treino de verificação (*smoke*) | aprovado |
| Grade completa de treino | **0/328 runs** |

O treino de verificação confirma o pipeline, mas seu mAP não é resultado
experimental. Após a grade completa,
[`analyze_triage.py`](scripts/analyze_triage.py) calcula, para os quatro métodos
inferenciais, o ganho médio pareado contra o `random`, o IC 95% por *bootstrap*
hierárquico, o teste exato de sinais e a correção de Holm; para o OPF, calcula
somente a comparação descritiva definida acima. O estado auditável está em
[`project_status.json`](outputs/bvtsld/project_status.json).

Após escolher a fração na validação, todos os métodos serão avaliados uma única
vez no teste, sem refazer a escolha.

---

## Etapa 2 — Replicação no TT100K

*Etapa ainda não iniciada.*

O [TT100K (Tsinghua-Tencent 100K)](https://cg.cs.tsinghua.edu.cn/traffic-sign/)
([Zhu et al., 2016](https://doi.org/10.1109/CVPR.2016.232)) contém cerca de
100 mil imagens de *street view* em alta resolução (2048 × 2048), com placas
pequenas em cenas complexas. O objetivo é verificar se o ranking dos métodos
do teste preliminar se mantém em escala.

O protocolo seguirá a Etapa 1, agora com três classes agregadas:
regulamentação (`p*`), advertência (`w*`) e indicação (`i*`). A política de
treino será definida antes da primeira execução, e a grade será rodada em uma
GPU CUDA dedicada.

As tabelas abaixo registram o que será preenchido quando a etapa começar.
Os traços indicam que ainda não há resultados.

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

Cada célula reportará mAP@0.5 / mAP@0.5:0.95 na validação.

| Método | 5% | 10% | 20% | 50% |
|---|---:|---:|---:|---:|
| `random` | — / — | — / — | — / — | — / — |
| `kmeans_dinov2` | — / — | — / — | — / — | — / — |
| `opf_dinov2` | — / — | — / — | — / — | — / — |
| `typiclust_dinov2` | — / — | — / — | — / — | — / — |
| `probcover_dinov2` | — / — | — / — | — / — | — / — |
| `freesel_fds_dino` | — / — | — / — | — / — | — / — |

---

## Etapa 3 — Trabalhos futuros: semi-supervisão na dissertação

A melhor estratégia das Etapas 1 e 2 definirá o conjunto rotulado inicial de
um modelo professor–aluno. O restante do pool será aproveitado por
*pseudo-labels*, seguindo trabalhos de detecção como Unbiased Teacher
([Liu et al., 2021](https://arxiv.org/abs/2102.09480)), Soft Teacher
([Xu et al., 2021](https://arxiv.org/abs/2106.09018)) e Efficient Teacher
([Xu et al., 2023](https://arxiv.org/abs/2302.07577)).

Serão comparadas três estratégias de confiança: limiar fixo
([FixMatch](https://arxiv.org/abs/2001.07685)), limiares adaptativos por classe
([FreeMatch](https://arxiv.org/abs/2205.07246)) e pesos contínuos
([SoftMatch](https://arxiv.org/abs/2301.10921)).

---

## Apêndice

### Termos

| Termo | O que é |
|---|---|
| **pool** | imagens de treino tratadas como não rotuladas durante a seleção |
| **fração de rótulos** | quanto do pool ganha rótulo manual: 5, 10, 20 ou 50% |
| **seleção** | o subconjunto de imagens escolhido para receber rótulo manual |
| **embedding** | vetor que resume o conteúdo de uma imagem, calculado sem usar seus rótulos |
| **agrupamento (clustering)** | juntar imagens de embedding parecido em grupos ("cenas de rodovia", "ruas à noite"...) |
| **cobertura DINOv2** | distância média do pool à seleção nesse espaço; é um diagnóstico, não o critério de ranking |
| **oráculo** | YOLO treinado com 100% dos rótulos, usado como referência de dados completos |
| **instância de seleção** | execução independente de uma técnica, usada como unidade de comparação |
| **semente de treino** | controla a inicialização e a aleatoriedade do YOLO |

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

Os datasets não são versionados. O script baixa os arquivos das fontes
oficiais, verifica o tamanho e extrai os ZIPs.

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

Na primeira execução também são baixados os pesos DINO/DINOv2 e YOLOv8n.

#### 2. Reprodução rápida

O fluxo rápido verifica dados, representações, seleções e treino de ponta a
ponta. Seu resultado não faz parte do experimento:

```bash
.venv/bin/python scripts/reproduce.py \
  --stage quick --accept-dataset-licenses
```

Para apenas conferir os comandos, sem executar nada:

```bash
.venv/bin/python scripts/reproduce.py --stage quick --dry-run
```

#### 3. Reprodução completa

O fluxo completo inclui o oráculo, os 328 treinos e a análise estatística.

```bash
.venv/bin/python scripts/reproduce.py \
  --stage all --device cuda --accept-dataset-licenses
```

Todos os comandos aceitam `--dataset bvtsld` (padrão) ou `--dataset tt100k`.
As classes-alvo, caminhos e o mapa de códigos de cada dataset ficam
centralizados em [`dataset_config.py`](scripts/dataset_config.py): o BVTSLD usa
duas classes e o TT100K usará as três classes agregadas por prefixo.

O treinador é retomável e não repete execuções concluídas. As etapas também
podem ser chamadas separadamente:

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

O comando executa uma configuração por processo e pode ser retomado após uma
falha.

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
- Pierre, M. M. (2023). *Brazilian Vertical Traffic Signs and Lights
  Dataset*. Mendeley Data, version 2.
  [DOI:10.17632/jbpsr4fvg9.2](https://doi.org/10.17632/jbpsr4fvg9.2)
- Rosa, G. H. de & Papa, J. P. (2021). *OPFython: A Python Implementation
  for Optimum-Path Forest*. Software Impacts.
  [DOI:10.1016/j.simpa.2021.100113](https://doi.org/10.1016/j.simpa.2021.100113)
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
