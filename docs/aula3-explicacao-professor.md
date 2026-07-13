# Aula 3 — Guia do Professor: o modelo explicado linha a linha

> Documento auxiliar **para o professor**. Explica o que a Fase 3 acrescenta: o
> treino do modelo, a previsão e a nova aba. Pressupõe os guias das Aulas 1 e 2.

---

## 0. A grande ideia da aula (diga primeiro)

Nas Fases 1 e 2 **descrevemos** o passado e fizemos um simulador **estatístico**
(Poisson, regras fixas). Agora a máquina vai **aprender** sozinha os padrões, a
partir das features que construímos. É a diferença entre "eu programei a regra" e
"o computador descobriu a regra olhando os dados".

O input já está pronto: o `dataset_modelagem.csv` (uma linha por partida, 18
features, sem data leak). A saída é um modelo que, dado um confronto, devolve
**P(Casa) / P(Empate) / P(Fora)**.

---

## 1. O conceito que não pode faltar: SPLIT TEMPORAL

Em ML normal a gente embaralha os dados e separa treino/teste aleatoriamente.
**Em previsão esportiva isso é proibido.**

> Diga assim: "Prever futebol é prever o **futuro**. Se eu treinar com jogos da
> rodada 15 para prever a rodada 10, estou deixando o modelo espiar o futuro —
> ele vai parecer genial no teste e falhar na vida real. É o mesmo pecado do data
> leak da Aula 2, agora na hora de validar."

A regra: **treina no passado, testa no futuro.**
```
Rodadas antigas ──────► corte │ ──────► Rodadas recentes
      TREINO                  │        TESTE
```

---

## 2. `src/features/build_features.py` — 1 função nova

`montar_features_atual(partidas, casa, fora)` — a ponte treino↔previsão.
- No treino, cada linha usa o histórico **antes** daquele jogo.
- Para prever um jogo **agora**, usamos o histórico **completo** de cada time (não
  há jogo futuro para vazar) e montamos **as mesmas 18 colunas**.
- **Por que isto importa:** o modelo aprendeu a ler 18 números numa ordem. Se na
  previsão a gente entregar colunas diferentes, ele erra ou quebra. Essa função
  garante que treino e previsão "falam a mesma língua". Há um teste que confere
  (`test_features_treino_e_previsao_batem`).
- A constante `FEATURES` (lista das 18 colunas) vira a **fonte única da verdade**,
  usada pelo treino e pela previsão.

---

## 3. `src/models/treinar.py` — o coração da aula

### 3.1. `split_temporal(dataset, fracao_teste=0.30)`
```python
rodadas = sorted(ds["rodada"].unique())
corte = rodadas[int(len(rodadas) * (1 - fracao_teste))]
treino = ds[ds["rodada"] < corte]
teste  = ds[ds["rodada"] >= corte]
```
- Pega o conjunto de rodadas, acha o ponto de corte (~70% do caminho) e separa.
  **Tudo antes do corte treina; tudo a partir dele testa.** Simples e correto.

### 3.2. Baseline ingênuo
```python
baseline_acc = (y_teste == "Casa").mean()
```
- A "nota de corte": e se a gente **sempre** chutasse "mandante vence"? No
  futebol isso acerta ~45-50%. O modelo só tem valor se **superar** isso.

### 3.3. Os dois modelos
```python
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
rf     = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample")
```
- **Regressão Logística:** modelo **linear**. O `StandardScaler` coloca as
  features na mesma escala (senão "saldo de gols" domina "forma"). É o "primeiro
  ML", fácil de explicar.
- **Random Forest:** um **comitê de árvores de decisão** que votam. Captura
  relações não-lineares e costuma ir melhor com pouco esforço.
- `class_weight="balanced"`: como há menos empates/derrotas que vitórias do
  mandante, isso evita o modelo simplesmente "chutar sempre casa". (`random_state`
  fixa a aleatoriedade → resultado reprodutível.)

### 3.4. Avaliação — `_avaliar`
```python
accuracy_score(y_te, pred)                    # % de acertos
log_loss(y_te, proba, labels=modelo.classes_) # qualidade das PROBABILIDADES
```
- **Acurácia** conta acertos. **Log loss** olha a **confiança**: prever "70% casa"
  e o time perder dói mais que prever "40% casa". Log loss menor = melhor
  calibrado. Ensina que "acertar o palpite" e "acertar a probabilidade" são coisas
  diferentes.

### 3.5. Escolha do melhor + importância
```python
nome_melhor = max(metricas, key=lambda m: (acuracia, -log_loss))
importancias = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(...)
```
- Salvamos o melhor por acurácia (desempate: log loss). A **importância das
  features** (do Random Forest) fecha o ciclo com a EDA da Fase 2: será que
  `diff_forma5` e `diff_saldo` realmente pesam? Normalmente sim.

### 3.6. Persistência — `treinar_e_salvar`
```python
joblib.dump(pacote, MODELO_V1)   # models/modelo_v1.pkl
```
- Guardamos o modelo **treinado** num arquivo. Assim não precisa retreinar toda
  vez — carrega pronto. O `.pkl` é **ignorado no Git** (como os dados): ele é
  gerado, não versionado. Quem clonar, treina de novo com um comando.
- Guardamos no "pacote" também as métricas, a matriz e as importâncias — para o
  app mostrar tudo **sem retreinar**.

---

## 4. `src/models/predizer.py` — usar o modelo

```python
def carregar_modelo():
    if not MODELO_V1.exists():
        return treinar.treinar_e_salvar()   # treina na 1ª vez
    return joblib.load(MODELO_V1)

def prever_partida(casa, fora, ...):
    feats = montar_features_atual(partidas, casa, fora)   # estado atual
    X = pd.DataFrame([feats])[pacote["features"]]         # mesma ordem do treino
    proba = pacote["modelo"].predict_proba(X)[0]
    ...
```
- `carregar_modelo` é "esperto": se o `.pkl` não existe, treina e salva na hora.
  O app nunca fica sem modelo.
- `prever_partida` monta as features do confronto **hoje** e pede as
  probabilidades. Repare no `[pacote["features"]]`: reordena as colunas
  exatamente como no treino — a tal "mesma língua".

---

## 5. `app/streamlit_app.py` — a aba "🧠 Previsão (ML)"

Quinta aba, com:
1. **Cards de probabilidade** + palpite do modelo.
2. **Modelo (ML) × Poisson (Fase 2)** lado a lado — o momento mais rico: dois
   métodos diferentes para a mesma pergunta. Quando concordam, mais confiança;
   quando divergem, ótimo debate.
3. **Desempenho:** acurácia (vs baseline), log loss, e a tabela comparando os dois
   modelos.
4. **Matriz de confusão** (heatmap) — mostra *onde* erra.
5. **Importância das features** (barras) — o que o modelo mais usou.

> Detalhe técnico: o modelo é carregado com `@st.cache_resource` (guarda o objeto
> na memória entre interações, sem recriar).

---

## 6. Lendo a matriz de confusão (mostre na tela)

```
            PREVISTO
            Casa  Empate  Fora
REAL Casa  [ 18     3      2 ]   <- acertou 18 vitórias do mandante
     Empate[  6     4      3 ]   <- empates: os mais difíceis
     Fora  [  3     2      7 ]
```
- A **diagonal** são os acertos. Fora dela, os erros. Quase sempre a coluna/linha
  do **empate** é a mais bagunçada — empate é o resultado mais imprevisível do
  futebol. Excelente gancho de discussão.

---

## 7. Erros comuns na aula

| Sintoma | Causa | Solução |
|---------|-------|---------|
| Acurácia ~ baseline (ou abaixo) | poucos dados; futebol é difícil | **é esperado** — o foco é a metodologia |
| `ValueError` de colunas na previsão | features fora de ordem | `montar_features_atual` + reindex por `FEATURES` já resolvem |
| Modelo "chuta sempre casa" | classes desbalanceadas | `class_weight="balanced"` |
| `.pkl` não existe após clonar | é ignorado no Git | rodar `python -m src.models.treinar` |
| App não atualiza o modelo | `cache_resource` antigo | reiniciar o `streamlit run` |

---

## 8. Honestidade intelectual (diga isto!)

- Com ~100 jogos de treino, a acurácia realista fica em **50-55%**. Não é pouco:
  prever futebol é difícil **de propósito** (se fosse fácil, as casas de aposta
  quebrariam).
- O modelo **não** conhece lesão, cartão, viagem, motivação — isso é a **Fase 5**.
- **Não use para apostar.** É um projeto didático.

> Gancho da Fase 4: "Agora o modelo só roda no nosso script. Como fazer outras
> pessoas/aplicativos usarem essa previsão? Próxima aula: empacotamos tudo numa
> **API**."

---

## 9. Onde está cada coisa

| Arquivo | Papel |
|---------|-------|
| `src/features/build_features.py` | `montar_features_atual` (ponte treino↔previsão) + `FEATURES` |
| `src/models/treinar.py` | split temporal, baseline, LogReg/RF, métricas, salva o `.pkl` |
| `src/models/predizer.py` | `carregar_modelo`, `prever_partida` |
| `app/streamlit_app.py` | aba "🧠 Previsão (ML)" |
| `tests/test_modelo.py` | split sem vazamento, paridade de features, acurácia > acaso |
