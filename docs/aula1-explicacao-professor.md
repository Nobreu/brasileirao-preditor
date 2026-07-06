# Aula 1 — Guia do Professor: o projeto explicado linha a linha

> Documento auxiliar **para o professor**. Explica cada arquivo da Fase 1, o
> porquê de cada decisão e os conceitos a ensinar. Use como cola durante a aula.

---

## 0. Como tudo se conecta (o fluxo de dados)

A Fase 1 é uma **esteira de dados** (pipeline) em 3 etapas, terminando num painel:

```
  [ API football-data.org ]  ── ou ──  [ simulação determinística ]
                       │
                       ▼
   (1) coletar_tabela.py  ──►  data/raw/partidas.csv  +  tabela_atual.csv
                       │
                       ▼
   (2) limpar.py          ──►  data/processed/partidas.csv  +  tabela.csv
                       │
                       ▼
   (3) forma.py           ──►  data/processed/forma_recente.csv
                       │
                       ▼
   app/streamlit_app.py   ──►  Dashboard no navegador
```

`preparar_dados.py` é só o "botão liga" que executa 1 → 2 → 3 em ordem.

**Mensagem-chave para os alunos:** dados crus nunca vão direto para a tela. Eles
passam por coleta → limpeza → transformação. Cada etapa tem um arquivo e uma
responsabilidade única.

---

## 1. Conceitos que o professor precisa dominar antes

| Conceito | Onde aparece | Como explicar em 1 frase |
|----------|--------------|--------------------------|
| **DataFrame** | todo o projeto | "Uma tabela do Excel dentro do Python." |
| **Pipeline** | `preparar_dados.py` | "Uma receita: cada passo usa o resultado do anterior." |
| **Fallback** | `coletar_tabela.py` | "Plano B automático quando o plano A falha." |
| **Determinismo** | `simular_temporada` | "Mesma semente → mesmos números sempre." |
| **Variável de ambiente / `.env`** | `coletar_api` | "Senha guardada fora do código." |
| **Cache** | `streamlit_app.py` | "Guarda o resultado pra não recalcular toda hora." |
| **Padronização** | `times.py` | "'Inter', 'SC Internacional' e 'Internacional' viram o mesmo nome." |

---

## 2. `requirements.txt` — as dependências

```
pandas>=2.0        # tabelas de dados (o coração do projeto)
numpy              # números e sorteio aleatório (simulação)
requests           # baixar dados da internet (API)
beautifulsoup4     # raspagem de HTML (reserva, caso a API falhe)
streamlit          # transforma script Python em site/dashboard
plotly             # gráficos interativos
scikit-learn       # machine learning (entra de verdade na Fase 3)
joblib             # salvar modelos em arquivo (Fase 3)
fastapi / uvicorn  # API web (Fase 4)
python-dotenv      # ler o arquivo .env com a chave da API
```

> Já incluímos as libs das fases futuras para o `requirements.txt` não mudar a
> cada aula. Na Fase 1 só usamos pandas, numpy, requests, streamlit, plotly e
> python-dotenv.

---

## 3. `.gitignore` — o que NÃO vai para o GitHub

```
__pycache__/  *.pyc        → lixo que o Python gera ao rodar
.venv/  venv/              → o ambiente virtual (cada um cria o seu)
.env                       → A CHAVE DA API (segredo! nunca subir)
data/raw/*.csv             → dados crus (são gerados, não versionados)
data/processed/*.csv       → dados tratados (idem)
models/*.pkl               → modelos treinados (Fase 3)
PROJETO_BRASILEIRAO_PREDITOR.md  → o roteiro do curso (material do professor)
.claude/                   → configurações locais de ferramenta
```

**Ponto pedagógico de ouro:** dados e segredos **não** vão para o Git. O
repositório guarda **código** (a receita), não os **dados** (o resultado). Quem
clonar o projeto roda o pipeline e gera os próprios dados.

---

## 4. `src/data/paths.py` — endereços centrais

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
```
- `Path(__file__)` = o caminho deste próprio arquivo (`src/data/paths.py`).
- `.resolve()` = transforma em caminho absoluto completo.
- `.parents[2]` = sobe 2 níveis: `paths.py` → `data/` → `src/` → **raiz**.
- **Por quê:** assim o projeto acha as pastas estando em qualquer lugar. Sem
  isso, rodar de dentro de `notebooks/` quebraria os caminhos relativos.

```python
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
```
- O operador `/` do `pathlib` junta caminhos de forma que funciona no Windows e
  no Linux igual (`\` vs `/`). **Nunca** escrevemos `"data\raw"` na mão.

```python
PARTIDAS_RAW = RAW / "partidas.csv"
TABELA_RAW = RAW / "tabela_atual.csv"
PARTIDAS_PROC = PROCESSED / "partidas.csv"
TABELA_PROC = PROCESSED / "tabela.csv"
FORMA_PROC = PROCESSED / "forma_recente.csv"
```
- Damos um **nome** para cada arquivo. Se um dia mudar o nome do CSV, muda aqui
  num lugar só, e todo o projeto acompanha.

```python
def garantir_pastas() -> None:
    for pasta in (RAW, PROCESSED, MODELS):
        pasta.mkdir(parents=True, exist_ok=True)
```
- `mkdir` cria a pasta. `parents=True` cria as pastas intermediárias se faltarem.
  `exist_ok=True` = "não dê erro se já existir". Chamada no início de cada script
  para garantir que as pastas existem antes de salvar arquivos.

---

## 5. `src/data/times.py` — os times e a padronização de nomes

### A lista canônica (linhas 9-30)
```python
TIMES_SERIE_A = ["Athletico-PR", "Atlético-GO", ... ]  # 20 times
```
- É a **fonte da verdade** dos nomes "oficiais" que usamos. Serve de base para a
  simulação. (Obs.: é a lista de 2024; com dados reais da API, valem os nomes
  que a API mandar — ver `normalizar_nome`.)

### A força dos times (linhas 35-56)
```python
FORCA = {"Botafogo": 1.40, "Cuiabá": 0.80, ...}
```
- Número que representa "quão forte" cada time é. **Só serve para a simulação**
  (quando não há internet) — não é estatística real. Valor maior = marca mais
  gols e sofre menos.

### O mapa de apelidos (linhas 59-121)
```python
APELIDOS = {
    "inter": "Internacional",
    "rb bragantino": "Bragantino",
    "ca mineiro": "Atlético-MG",     # nome que a API usa
    ...
}
```
- **O problema que resolve:** a mesma equipe aparece com nomes diferentes
  conforme a fonte. A API chama de "CA Mineiro", o torcedor chama de "Galo", o
  site chama de "Atlético-MG". Sem padronizar, o Atlético-MG viraria **3 times
  diferentes** na tabela.
- A chave (esquerda) é o nome "sujo" em minúsculas e sem acento; o valor
  (direita) é o nome bonito final.

### A função `normalizar_nome` (linhas 124-151)
```python
def normalizar_nome(nome: str) -> str:
    if not isinstance(nome, str):
        return nome                       # segurança: se vier nulo, devolve igual
    chave = nome.strip().lower()          # tira espaços e põe em minúsculas
    import unicodedata
    chave_sem_acento = (                  # "São Paulo" -> "sao paulo"
        unicodedata.normalize("NFKD", chave)
        .encode("ascii", "ignore").decode("ascii")
    )
    if chave_sem_acento in APELIDOS:      # achou no dicionário? devolve o oficial
        return APELIDOS[chave_sem_acento]
    for canon in TIMES_SERIE_A:           # já é um nome canônico (com/sem acento)?
        if chave_sem_acento == (... canon sem acento ...):
            return canon
    return nome.strip()                   # não achou: devolve limpo, sem perder
```
- **Estratégia em camadas:** (1) tenta o dicionário de apelidos; (2) tenta casar
  com a lista canônica ignorando acento; (3) se nada bater, devolve o nome
  original limpo — **nunca apaga um time só porque não reconheceu**.
- `unicodedata.normalize("NFKD", ...).encode("ascii", "ignore")` é o truque
  clássico de **remover acentos** para comparar textos.

> 💡 **Possível pergunta de aluno:** "Por que não usar só o nome da API direto?"
> Resposta: porque misturamos fontes (API hoje, raspagem amanhã) e queremos um
> nome único e bonito independente de quem mandou o dado.

---

## 6. `src/data/coletar_tabela.py` — a coleta (com plano B)

Cabeçalho (linhas 25-26):
```python
SEMENTE = 42                         # semente do sorteio (determinismo)
DATA_INICIO = date(2024, 4, 13)      # data fictícia do 1º jogo simulado
```

### 6.1. `_gerar_calendario` (linhas 32-58) — quem joga contra quem
Monta a tabela de jogos usando o **método do círculo** (algoritmo clássico de
torneio "todos contra todos"):
```python
n = len(times)                  # 20
rodada_meia = n - 1             # 19 rodadas no 1º turno
fixos = times[0]                # 1 time fica parado
rotativos = times[1:]           # os outros 19 giram
```
- A cada rodada, emparelha o primeiro com o último, o segundo com o penúltimo
  etc. (`ordem[i]` contra `ordem[n-1-i]`).
- `if r % 2 == 0` alterna o **mando de campo** entre rodadas para ninguém jogar
  só em casa.
- `rotativos = [rotativos[-1]] + rotativos[:-1]` → **gira a lista** (pega o
  último e põe na frente). É o "círculo" do algoritmo.
- O **returno** (linha 57) repete os confrontos invertendo casa/fora: 38 rodadas
  no total, 380 jogos.

> O `_` no início do nome (`_gerar_calendario`) é convenção Python para
> "função interna, uso privado deste arquivo".

### 6.2. `simular_temporada` (linhas 64-91) — inventa os placares
```python
rng = np.random.default_rng(semente)         # gerador aleatório COM semente fixa
...
lambda_casa = 1.35 * (f_casa / f_fora) ** 0.9
lambda_fora = 1.05 * (f_fora / f_casa) ** 0.9
gols_casa = int(rng.poisson(lambda_casa))
gols_fora = int(rng.poisson(lambda_fora))
```
- **Distribuição de Poisson:** o modelo estatístico clássico para contar gols num
  jogo de futebol. `lambda` é a "média esperada de gols".
- A média do mandante começa maior (1.35 vs 1.05) → representa a **vantagem de
  jogar em casa**. E é ajustada pela razão de forças dos times.
- Como a semente é fixa, **a simulação dá sempre o mesmo campeonato** — essencial
  para a aula ser reproduzível (todo aluno vê os mesmos números).
- Cada jogo vira uma linha (dicionário) com rodada, data, times e gols.

### 6.3. `coletar_api` (linhas 97-136) — os dados reais
```python
from dotenv import load_dotenv
load_dotenv(override=False)                   # lê o .env sem sobrescrever o ambiente
chave = os.environ.get("FOOTBALL_DATA_API_KEY")
if not chave:
    return None                               # sem chave -> avisa "não consegui"
```
```python
url = "https://api.football-data.org/v4/competitions/BSA/matches"
resp = requests.get(url, headers={"X-Auth-Token": chave}, timeout=15)
resp.raise_for_status()                       # erro HTTP -> levanta exceção
dados = resp.json()                           # resposta vem em JSON
```
- `BSA` é o código do Brasileirão Série A na API. A chave vai no **cabeçalho**
  `X-Auth-Token` (autenticação).
- `timeout=15` = desiste depois de 15s (não trava a aula se a internet cair).
```python
for m in dados.get("matches", []):
    if m.get("status") != "FINISHED":
        continue                              # ignora jogos que ainda não aconteceram
    placar = m["score"]["fullTime"]
    linhas.append({... "gols_casa": placar["home"] ...})
```
- Só pega jogos **finalizados** — não dá para calcular tabela com jogo que não
  rolou.
```python
except Exception as exc:
    print(f"[coletar_api] Falhou, usando simulação. Motivo: {exc}")
    return None
```
- **O coração do plano B:** se *qualquer coisa* der errado (internet, limite de
  API, formato inesperado), a função devolve `None` em vez de quebrar. Quem
  decide o plano B é o `main`.

### 6.4. `derivar_tabela` (linhas 142-178) — monta a classificação
```python
estat = {t: {"P":0,"J":0,"V":0,"E":0,"D":0,"GP":0,"GC":0} for t in todos_os_times}
```
- Um "boletim" zerado para cada time (Pontos, Jogos, Vitórias, Empates,
  Derrotas, Gols Pró, Gols Contra).
```python
for _, jogo in partidas.iterrows():           # percorre jogo por jogo
    ...
    if gc > gf:    # mandante fez mais gols
        estat[casa]["V"] += 1; estat[casa]["P"] += 3; estat[fora]["D"] += 1
    elif gc < gf:  # visitante venceu
        estat[fora]["V"] += 1; estat[fora]["P"] += 3; estat[casa]["D"] += 1
    else:          # empate
        estat[casa]["E"] += 1; estat[fora]["E"] += 1
        estat[casa]["P"] += 1; estat[fora]["P"] += 1
```
- Implementa **a regra do futebol**: vitória = 3 pontos, empate = 1, derrota = 0.
```python
tabela["SG"] = tabela["GP"] - tabela["GC"]                  # saldo de gols
tabela = tabela.sort_values(["P","V","SG","GP"], ascending=False)
tabela.insert(0, "posicao", tabela.index + 1)
```
- Ordena pelos **critérios de desempate oficiais**: pontos → vitórias → saldo →
  gols pró. Depois numera as posições de 1 a 20.

### 6.5. `main` (linhas 184-202) — orquestra a coleta
```python
partidas = coletar_api()                       # tenta o real
origem = "API football-data.org"
if partidas is None:                           # falhou? plano B
    partidas = simular_temporada()
    origem = "simulação determinística"
partidas["time_casa"] = partidas["time_casa"].map(normalizar_nome)   # padroniza
...
partidas.to_csv(PARTIDAS_RAW, index=False, encoding="utf-8")
```
- `.map(normalizar_nome)` aplica a padronização **em cada célula** da coluna.
- `index=False` = não salva o número de linha do pandas no CSV.
- `encoding="utf-8"` = preserva acentos (São Paulo, Grêmio).

---

## 7. `src/data/limpar.py` — o tratamento

```python
df["time_casa"] = df["time_casa"].map(normalizar_nome)   # reforça padronização
df["data"] = pd.to_datetime(df["data"], errors="coerce") # texto -> data de verdade
for col in ("rodada","gols_casa","gols_fora"):
    df[col] = pd.to_numeric(df[col], errors="coerce")    # texto -> número
```
- `errors="coerce"` = "se não conseguir converter, vira vazio (NaN)" em vez de
  quebrar. É a forma defensiva de converter tipos.
```python
df = df.dropna(subset=["gols_casa","gols_fora","time_casa","time_fora"])
df[["gols_casa","gols_fora"]] = df[["gols_casa","gols_fora"]].astype(int)
```
- `dropna` remove linhas com buraco nos campos essenciais (jogo sem placar não
  serve). Depois converte gols para inteiro.
```python
df["resultado"] = df.apply(
    lambda r: "Casa" if r.gols_casa > r.gols_fora
    else ("Fora" if r.gols_casa < r.gols_fora else "Empate"), axis=1)
```
- Cria uma coluna **legível** com quem venceu. `axis=1` = aplica linha a linha.
  Útil para a EDA e para o dashboard.
- No `main`, salva os dados limpos em `data/processed/` e **recalcula a tabela**
  a partir das partidas já limpas (reaproveitando `derivar_tabela`).

> **Por que limpar de novo se a coleta já padronizou?** Porque a limpeza é uma
> etapa independente: ela deve funcionar mesmo que os dados venham de um CSV
> pronto, sem ter passado pela coleta. Cada etapa não confia cegamente na
> anterior.

---

## 8. `src/data/forma.py` — a forma recente (últimos N jogos)

### `_jogos_do_time` (a parte mais "esperta")
```python
em_casa = partidas[partidas["time_casa"] == time].copy()
em_casa["gols_pro"] = em_casa["gols_casa"]      # quando joga em casa, "pró" = gols_casa
em_casa["gols_contra"] = em_casa["gols_fora"]

fora = partidas[partidas["time_fora"] == time].copy()
fora["gols_pro"] = fora["gols_fora"]            # quando joga fora, "pró" = gols_fora
fora["gols_contra"] = fora["gols_casa"]

jogos = pd.concat([em_casa, fora]).sort_values(["rodada","data"])
```
- **A sacada:** um time aparece ora na coluna `time_casa`, ora em `time_fora`.
  Para olhar "do ponto de vista do time", juntamos os dois casos e criamos
  colunas neutras `gols_pro` / `gols_contra`. Aí tanto faz se foi em casa ou fora.
```python
jogos["res"] = jogos.apply(lambda r: "V" if r.gols_pro > r.gols_contra
                           else ("D" if r.gols_pro < r.gols_contra else "E"), axis=1)
```
- Marca cada jogo como V/E/D **na perspectiva daquele time**.

### `calcular_forma`
```python
ultimos = _jogos_do_time(partidas, time).tail(n)     # os N MAIS RECENTES
seq = list(ultimos["res"])                           # ex: ["V","E","D","V","V"]
"pontos_ultimos_n": sum(pontos_por_res[r] for r in seq)   # V=3, E=1, D=0
"sequencia": "-".join(seq)                           # "V-E-D-V-V"
```
- `.tail(n)` pega as **últimas n linhas** depois de ordenar por rodada → ou seja,
  os jogos mais recentes. É isso que garante "forma = últimos 5 jogos literais".
- A sequência é guardada como texto `"V-E-D-V-V"` (antigo → recente). O dashboard
  depois quebra isso e pinta cada letra.

> **Por que a lógica está em um módulo e não só no notebook?** Para o dashboard
> **reaproveitar a mesma função**. Regra de ouro: lógica que importa mora em
> `src/`, não dentro de um notebook.

---

## 9. `src/data/preparar_dados.py` — o "botão liga"

```python
from . import coletar_tabela, forma, limpar

def executar(forcar: bool = False) -> None:
    processados = (PARTIDAS_PROC, TABELA_PROC, FORMA_PROC)
    if not forcar and all(p.exists() for p in processados):
        print("Dados já existem...")
        return                              # evita rebaixar a API à toa
    coletar_tabela.main()                   # 1) coleta
    limpar.main()                           # 2) limpa
    forma.main()                            # 3) forma recente
```
- Executa o pipeline inteiro **na ordem certa**.
- `forcar=False`: se os dados já existem, não faz nada (rápido). `forcar=True`
  (usado por `python -m src.data.preparar_dados`): refaz tudo, buscando dados
  novos da API.
- É exatamente o que o dashboard chama quando abre e não encontra os dados.

---

## 10. `app/streamlit_app.py` — o dashboard

### Truque do caminho (linhas iniciais)
```python
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
```
- O Streamlit roda o arquivo de dentro de `app/`, então o Python não acharia o
  pacote `src`. Essas linhas **adicionam a raiz do projeto** ao caminho de
  importação. (Detalhe técnico; não precisa gastar muito tempo com alunos.)

### Carregar dados com cache e auto-preparação
```python
@st.cache_data
def carregar_dados():
    if not all(p.exists() for p in (TABELA_PROC, FORMA_PROC, PARTIDAS_PROC)):
        preparar_dados.executar()           # gera os dados se faltarem
    return pd.read_csv(TABELA_PROC), pd.read_csv(FORMA_PROC), pd.read_csv(PARTIDAS_PROC)
```
- `@st.cache_data` = **memoriza** o resultado. Sem isso, o Streamlit releria os
  CSVs a cada clique do usuário. Com isso, lê uma vez só.
- Se os CSVs não existem, **chama o pipeline sozinho** → o app "se vira" mesmo se
  o aluno esquecer de rodar `preparar_dados`.

### A "forminha" colorida
```python
def badge_sequencia(seq: str) -> str:
    for r in str(seq).split("-"):           # "V-E-D" -> ["V","E","D"]
        cor = CORES_RES.get(r, "#cccccc")   # V=verde, E=cinza, D=vermelho
        bolinhas.append("<span style='...background:{cor}...'>{r}</span>")
```
- Gera **HTML** com um quadradinho colorido por resultado. O Streamlit renderiza
  com `unsafe_allow_html=True`. É o que dá o efeito visual 🟩⬜🟥.

### O layout (de cima para baixo)
1. `st.set_page_config(...)` → título e ícone da aba.
2. **Sidebar** com `st.selectbox` → filtro por time.
3. `st.columns(4)` + `st.metric` → 4 cartões de resumo (times, partidas, rodadas,
   gols).
4. **Classificação**: montada como tabela HTML para conseguir colorir a coluna
   "Forma". Cada linha é construída no loop `for _, r in destaque.iterrows()`.
5. **Forma recente**: se um time está filtrado, mostra detalhe; senão, uma grade
   de cartões (`st.columns(4)` em loop).
6. **Gráfico**: `px.bar(...)` (Plotly) com a pontuação de cada time, horizontal.

> **Mensagem para os alunos:** um app Streamlit é só um **script Python lido de
> cima para baixo**. Cada `st.algo()` desenha um pedaço da tela na ordem em que
> aparece no código. Não tem "frente e verso" como num site tradicional.

---

## 11. `tests/test_dados.py` — a rede de segurança

```python
def test_simulacao_tem_todos_os_jogos():
    p = simular_temporada()
    assert len(p) == 380                    # 20 times, turno e returno
    assert set(p["time_casa"]) == set(TIMES_SERIE_A)
```
- Um **teste** é código que confere se outro código está certo. `assert` =
  "garanta que isto é verdade, senão acuse erro".
- Testamos: nº de jogos, tabela com 20 times, regra de pontos (`P == 3*V + E`),
  padronização de nomes, e forma com no máximo 5 jogos.
- Rodar com `python tests/test_dados.py` → imprime "OK" se tudo passa.

> **Por que ensinar testes já na aula 1?** Para criar o hábito: "como eu *sei*
> que meu cálculo de pontos está certo?" Em vez de conferir no olho, o teste
> confere sozinho — e continua conferindo nas próximas fases.

---

## 12. `notebooks/01-eda.ipynb` — a exploração

- O notebook **importa as mesmas funções** de `src/` e mostra os resultados:
  tabela, forma recente e a distribuição de resultados (`value_counts`).
- A célula `partidas['resultado'].value_counts(normalize=True)` responde uma
  pergunta clássica: **"mando de campo importa?"** (qual % de vitórias do
  mandante). É o gancho perfeito para a Fase 2 (feature engineering).
- **Regra que vale a pena repetir:** o notebook é para *explorar e visualizar*; a
  lógica que o app usa fica nos módulos `.py`, nunca presa no notebook.

---

## 13. Erros comuns na aula (e como resolver na hora)

| Sintoma | Causa provável | Solução |
|---------|----------------|---------|
| `ModuleNotFoundError: src` | rodou de dentro de `app/` ou `notebooks/` | rodar da **raiz** do projeto, ou conferir o truque de `sys.path` |
| App abre vazio / sem dados | CSVs não gerados | `python -m src.data.preparar_dados` |
| Acentos quebrados no terminal | console do Windows | é só o terminal; o **CSV está correto** em UTF-8 |
| API retorna erro 403/429 | chave inválida ou limite de requisições | esperar 1 min; o projeto cai na simulação sozinho |
| `Activate.ps1` bloqueado | política do PowerShell | `Set-ExecutionPolicy -Scope Process -Bypass` |

---

## 14. Roteiro de fala sugerido (resumo de 1 minuto por arquivo)

1. **paths.py** — "Aqui ficam os endereços. Mexeu o arquivo de lugar? Muda só aqui."
2. **times.py** — "Aqui a gente garante que cada time tem UM nome só."
3. **coletar_tabela.py** — "Aqui buscamos os dados. Se a internet falhar, temos um plano B."
4. **limpar.py** — "Dado cru é sujo. Aqui ele vira dado confiável."
5. **forma.py** — "Aqui respondemos: como esse time vem jogando nos últimos 5?"
6. **preparar_dados.py** — "O botão que aperta tudo na ordem."
7. **streamlit_app.py** — "A tela. Um script lido de cima pra baixo."
8. **test_dados.py** — "A prova de que a conta está certa, automática."

---

## 15. Gancho para a Fase 2

> "Hoje a gente respondeu **o que aconteceu** (tabela e forma). Repare na última
> célula do notebook: o mandante vence muito mais que o visitante. Isso é uma
> **pista** — uma variável que ajuda a prever o futuro. Na próxima aula, vamos
> transformar cada um desses padrões em **features** para um modelo aprender."
