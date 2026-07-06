# Aula 2 — Guia do Professor: features explicadas linha a linha

> Documento auxiliar **para o professor**. Foca no novo da Fase 2: a engenharia
> de features e a nova aba do dashboard. Pressupõe que o guia da Aula 1 já foi
> visto.

---

## 0. A grande ideia da aula (diga isto primeiro)

Na Aula 1 mostramos **o que aconteceu** (tabela, forma). Um modelo não aprende
com "o Flamengo é bom" — ele aprende com **números**. Feature engineering é
transformar a história em **colunas numéricas** que descrevem o estado dos times
**antes** de cada jogo.

O produto da aula é o **dataset de modelagem**: uma tabela onde
**cada linha é uma partida** e as colunas respondem: "como os dois times
chegaram para este jogo?" — mais uma coluna `alvo` com quem venceu.

```
| time_casa | time_fora | casa_forma5 | fora_forma5 | diff_saldo | ... | alvo   |
| Palmeiras | Santos    | 13          | 4           | +12        | ... | Casa   |
```

---

## 1. O conceito que não pode faltar: DATA LEAK

**Data leak (vazamento de dados)** = usar, ao montar uma feature, informação que
**ainda não existia** no momento da partida. É o erro nº 1 em projetos de ML
esportivo.

Exemplo para a turma:
> "Se para prever o jogo da rodada 10 eu usar o *saldo de gols final da
> temporada*, estou trapaceando: na rodada 10 esse número ainda não existe. O
> modelo ficaria 'genial' no teste e **inútil** na vida real."

Como evitamos no código: montamos o dataset **em ordem cronológica** e calculamos
as features de cada jogo usando **só o histórico anterior** — o resultado do jogo
atual entra no histórico **depois** de a linha já ter sido gerada.

> 💡 Tem um **teste automático** (`test_sem_data_leak`) que prova isso. Vale
> mostrar rodando ao vivo: "o computador confere que não trapaceamos".

---

## 2. `src/data/paths.py` — 1 linha nova

```python
DATASET_MODELAGEM = PROCESSED / "dataset_modelagem.csv"
```
- Só registramos o endereço do novo arquivo. Padrão da Aula 1 mantido: caminhos
  ficam todos num lugar só.

---

## 3. `src/features/build_features.py` — o núcleo da aula

### 3.1. Constantes (topo)
```python
PONTOS = {"V": 3, "E": 1, "D": 0}   # regra de pontos do futebol
JANELA = 5                          # "forma recente" = últimos 5 jogos
MIN_JOGOS = 3                       # só gera linha se ambos já jogaram >= 3
```
- `MIN_JOGOS` existe para não criar features sem sentido nas primeiras rodadas
  (um time com 1 jogo não tem "forma"). É uma decisão de qualidade de dados.

### 3.2. `_features_time(hist, mando)` — features de UM time
Recebe o **histórico de jogos anteriores** de um time (lista de dicionários) e
devolve as features. Cada jogo no histórico tem: `res` (V/E/D), `pontos`, `gp`
(gols pró), `gc` (gols contra), `casa` (jogou em casa? True/False).

```python
ultimos = hist[-JANELA:]                        # os 5 mais recentes (rolling window)
forma = sum(j["pontos"] for j in ultimos)       # pontos nos últimos 5
media_gp = sum(j["gp"] for j in ultimos) / len(ultimos)   # média de gols feitos
media_gc = sum(j["gc"] for j in ultimos) / len(ultimos)   # média de gols sofridos
saldo = sum(j["gp"] - j["gc"] for j in hist)    # saldo na temporada (tudo)
aprov_geral = sum(j["pontos"] for j in hist) / (3 * n)    # % de pontos ganhos
```
- **Janela móvel (rolling window):** `hist[-5:]` é o pandas/Python jeito simples
  de dizer "os 5 últimos". Conceito que aparece muito em séries temporais.
- **Aproveitamento** = pontos ganhos ÷ pontos possíveis (3 por jogo). Vai de 0 a 1.

```python
quer_casa = mando == "casa"
mesmos = [j for j in hist if j["casa"] == quer_casa]
aprov_mando = (sum pontos / (3*len)) if mesmos else aprov_geral
```
- **Aproveitamento no mando:** o mandante entra pela porta "casa", então olhamos
  só os jogos dele **em casa**; o visitante, só os jogos **fora**. Se ainda não
  tem histórico naquele mando, usamos o aproveitamento geral como aproximação.

```python
sem_derrota = 0
for j in reversed(hist):        # do jogo mais recente para trás
    if j["res"] == "D": break
    sem_derrota += 1
```
- **Sequência invicta:** conta quantos jogos seguidos, a partir do último, o time
  passou sem perder. `sem_vitoria` é análogo (sem vencer). Capturam "momento".

### 3.3. `_registrar(...)` — atualiza o histórico
```python
def _registrar(hist, time, gp, gc, casa):
    res = "V" if gp > gc else ("D" if gp < gc else "E")
    hist.setdefault(time, []).append({...})
```
- Acrescenta um jogo ao histórico do time. `setdefault(time, [])` = "se o time
  ainda não tem lista, cria uma vazia". **É chamado DEPOIS de gerar a linha** —
  essa ordem é o que evita o data leak.

### 3.4. `construir_dataset(...)` — o laço principal
```python
df = df.sort_values(["rodada", "data"])     # ORDEM CRONOLÓGICA (essencial!)
hist = {}                                    # histórico por time
h2h = {}                                     # confrontos diretos

for _, jogo in df.iterrows():
    h_casa = hist.get(casa, []); h_fora = hist.get(fora, [])
    if len(h_casa) >= MIN_JOGOS and len(h_fora) >= MIN_JOGOS:
        fc = _features_time(h_casa, "casa")   # <- só histórico ANTERIOR
        ff = _features_time(h_fora, "fora")
        ...
        linhas.append({... features ..., "alvo": alvo})
    # SÓ AGORA registra o jogo atual:
    _registrar(hist, casa, gc, gf, casa=True)
    _registrar(hist, fora, gf, gc, casa=False)
    h2h[...].append(vencedor)
```
- **Leia em voz alta a ordem:** (1) pega histórico anterior → (2) gera as features
  e a linha → (3) só então adiciona o jogo atual ao histórico. Inverter isso =
  data leak.
- **Confronto direto (h2h):** `h2h_saldo_casa = vitórias do mandante − vitórias
  do visitante` nos encontros **anteriores** entre os dois. `frozenset((a,b))`
  serve de chave que ignora a ordem (A×B é o mesmo par que B×A).
- **Features comparativas** (`diff_*`): a diferença mandante − visitante costuma
  ser mais informativa para o modelo do que os dois valores separados.

### 3.5. `resumo_times(...)` — para o dashboard
- Versão **descritiva** (sem alvo, sem preocupação com leak): calcula o estado
  **atual** de cada time usando **todos** os jogos. Alimenta a tabela e os
  gráficos da nova aba. Separa aproveitamento em casa e fora — que vira o scatter.

### 3.6. `main()`
- Lê `partidas.csv`, chama `construir_dataset`, salva o CSV e imprime a
  **distribuição do alvo** (quantos Casa/Empate/Fora). Serve para discutir
  **desbalanceamento**: mandante vence bem mais — o modelo precisa "bater" isso.

---

## 4. `src/data/preparar_dados.py` — a bola de neve

```python
coletar_tabela.main(); limpar.main(); forma.main()
from ..features import build_features
build_features.main()          # <- passo novo da Fase 2
```
- O pipeline da Aula 1 ganhou **mais um passo** no fim. Nada foi jogado fora — é
  o princípio do curso: cada fase soma à anterior.

---

## 5. `app/streamlit_app.py` — a nova aba

O app foi reorganizado em **abas** (`st.tabs`):
```python
aba_tabela, aba_analise = st.tabs(["📊 Classificação & Forma", "🔬 Análise de Times"])
with aba_tabela:   # tudo da Aula 1 (intacto)
with aba_analise:  # novidade da Aula 2
```
A aba "Análise de Times" tem 4 blocos:
1. **Tabela de features** (`st.dataframe(resumo)`) — o estado atual de cada time.
2. **Comparar dois times** — dois `selectbox` e um gráfico de barras agrupadas
   (`px.bar(..., barmode="group")`).
3. **Scatter casa × fora** — cada ponto é um time; a linha tracejada é a
   diagonal. Acima dela = time rende melhor fora do que em casa. Ótimo para
   discutir "fator casa".
4. **Heatmap de correlação** — `px.imshow` da matriz de correlação. Convertermos
   o alvo em número (`Casa=1, Empate=0, Fora=-1`) para ver quais features se
   correlacionam com o resultado. **É o gancho da Aula 3**: as features com
   correlação mais forte são candidatas naturais para o modelo.

> Nota técnica: trocamos `use_container_width=True` por `width="stretch"` (API
> nova do Streamlit) — só um detalhe de versão.

---

## 6. `notebooks/02-features.ipynb` — a EDA

Reproduz, passo a passo, as 3 visualizações exigidas:
1. **Heatmap** de correlação entre features.
2. **Boxplot** da forma do mandante por resultado (`px.box`): times em melhor
   forma vencem mais em casa? Espera-se que a caixa do "Casa" fique mais alta.
3. **Scatter** aproveitamento casa × fora por time.

Mensagem final da EDA: **"quais features mais se relacionam com o resultado?"** —
a resposta guia a escolha de features no modelo da próxima aula.

---

## 7. `tests/test_features.py` — a prova

| Teste | O que garante |
|-------|---------------|
| `test_dataset_tem_pelo_menos_8_features` | o dataset tem features suficientes (critério de aceitação) |
| `test_alvo_tem_tres_classes` | o alvo é sempre Casa/Empate/Fora |
| **`test_sem_data_leak`** | **construir o dataset só até a partida k dá a MESMA linha da partida k no dataset completo** → nenhum jogo futuro influencia o passado |
| `test_resumo_times_tem_20` | o resumo cobre os 20 times e aproveitamentos ficam entre 0 e 1 |

O teste de data leak é o mais importante da aula — vale explicá-lo devagar:
> "Se eu montar as features usando só os jogos até a partida 120, e depois montar
> usando o campeonato inteiro, a linha da partida 120 tem que ser **idêntica**.
> Se mudar, é porque um jogo do futuro vazou para o passado."

---

## 8. Erros comuns na aula

| Sintoma | Causa | Solução |
|---------|-------|---------|
| Dataset com poucas linhas | poucos jogos disputados (início de temporada) | normal; `MIN_JOGOS` corta as primeiras rodadas |
| `KeyError` num time | nome não padronizado | conferir `src/data/times.py` (mapa de apelidos) |
| Heatmap "todo azul/vermelho" | poucas partidas → correlações instáveis | mais dados (temporada mais avançada) suaviza |
| Aba nova não aparece | app antigo em cache | recarregar a página / reiniciar o `streamlit run` |

---

## 9. Roteiro de fala (resumo)

1. "Modelo não entende 'time bom', entende **número**." → motivação.
2. "Cada linha é **uma partida**; as colunas dizem como os times chegaram."
3. "A regra sagrada: **só o passado**. Nada do futuro pode vazar." → data leak.
4. "Forma, aproveitamento, saldo, sequência, confronto direto." → as features.
5. "Vamos **olhar** os dados." → heatmap, boxplot, scatter.
6. "Quais features falam mais alto sobre o resultado?" → gancho da Aula 3.
