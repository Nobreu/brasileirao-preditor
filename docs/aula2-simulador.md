# Simulador de Partida + Projeção de Temporada (Fase 2+)

> Documento auxiliar **para o professor**. Explica o simulador estatístico
> adicionado à Fase 2: modelo de Poisson, Monte Carlo e projeção de temporada.

---

## 0. A ideia em uma frase

Com os gols que cada time **realmente** fez e sofreu, dá para estimar quantos
gols ele tende a fazer no próximo jogo — e daí calcular a chance de vitória,
empate ou derrota. Isso é o **modelo de Poisson**, o método clássico do futebol.
Não é machine learning; é probabilidade. Por isso cabe na Fase 2 (e prepara a
intuição para o modelo da Fase 3).

---

## 1. Por que Poisson?

A distribuição de **Poisson** descreve "quantas vezes um evento raro acontece num
intervalo" — exatamente o caso de gols numa partida. Ela tem um único parâmetro,
`lambda` (λ), que é a **média esperada de gols**. Sabendo λ, sabemos a
probabilidade de 0, 1, 2, 3... gols.

> Analogia para a turma: "λ é o 'termômetro de gols' do time naquele jogo. Um λ
> de 2,0 quer dizer que, em média, esperamos 2 gols dele — mas às vezes sai 0,
> às vezes sai 4."

---

## 2. Como calculamos a força de cada time — `forcas_times`

A partir dos jogos finalizados, e **separando casa e fora**:
```
ataque_casa[time]  = (gols que o time FAZ em casa)   / média da liga em casa
defesa_casa[time]  = (gols que o time SOFRE em casa)  / média da liga fora
ataque_fora[time]  = (gols que o time FAZ fora)       / média da liga fora
defesa_fora[time]  = (gols que o time SOFRE fora)     / média da liga em casa
```
- Um `ataque_casa = 1.3` significa "esse time faz 30% mais gols em casa que a
  média". `defesa < 1` é bom (sofre menos que a média).
- **Shrinkage** (`SHRINKAGE = 4`): no começo da temporada, com poucos jogos, as
  médias são instáveis (um 4×0 distorce tudo). Puxamos os ratings suavemente para
  1.0 (a média) usando "4 jogos fantasma". Conforme o time joga mais, o dado real
  domina. É uma forma simples de **regularização**.

---

## 3. Gols esperados e placar — `gols_esperados` e `matriz_placares`

Para um confronto **mandante × visitante**:
```
λ_casa = média_liga_casa × ataque_casa[mandante] × defesa_fora[visitante]
λ_fora = média_liga_fora × ataque_fora[visitante] × defesa_casa[mandante]
```
- Lê-se: "o quanto o mandante ataca em casa **contra** o quanto o visitante
  defende fora". O produto capta o **duelo** ataque × defesa.

Depois montamos a **matriz de placares** (0..6 × 0..6):
```
P(placar a×b) = poisson.pmf(a, λ_casa) × poisson.pmf(b, λ_fora)
```
- Cada célula é a chance daquele placar exato. Assumimos os dois lados
  **independentes** (simplificação — ver limitação no fim).
- Normalizamos a matriz para somar 1 (compensa cortar em 6 gols).

De uma matriz saem todas as informações — `probabilidades`:
- **P(vitória casa)** = soma abaixo da diagonal (mandante fez mais).
- **P(empate)** = diagonal. **P(vitória fora)** = acima da diagonal.
- **Ambos marcam (BTTS)** = soma das células com a≥1 e b≥1.
- **Over 2,5** = soma das células com a+b ≥ 3.
- **Placar mais provável** = a célula de maior valor.

> Mostre ao vivo: as três probabilidades **sempre somam 100%**. Há um teste que
> garante isso (`test_matriz_e_probabilidades_somam_um`).

---

## 4. Monte Carlo — `monte_carlo_partida`

O cálculo acima é **analítico** (fórmula fechada). O **Monte Carlo** faz o mesmo
por "força bruta": sorteia 10.000 jogos com aquele λ e conta os resultados.
```python
gc = rng.poisson(λ_casa, 10000)   # 10 mil placares do mandante
gf = rng.poisson(λ_fora, 10000)
freq_casa = média(gc > gf)         # em quantos % o mandante venceu
```
- **Por que ensinar os dois?** Porque o Monte Carlo é intuitivo ("simula muitas
  vezes e conta") e **confirma** o resultado da fórmula. Quando os dois batem
  (diferença < 1 ponto percentual), o aluno confia no modelo. É a ponte perfeita
  para a projeção da temporada, que **só** dá para fazer por simulação.

---

## 5. Narração automática — `narrar`

Transforma os números num parágrafo em português ("O Palmeiras entra como
favorito (68%)... placar mais provável 1×0..."). É o "gerar informação" pedido:
o app não mostra só números, ele **conta a história** do jogo. É `if/else` puro
sobre as probabilidades — bom para mostrar que "IA que escreve texto" pode
começar simples, com regras.

---

## 6. Projeção da temporada — `projetar_temporada`

A pergunta mais legal: **"e como termina o campeonato?"** Respondemos simulando
**todos os jogos que faltam**, milhares de vezes:

1. Partimos da **tabela atual** (pontos e saldo de cada time).
2. Para cada jogo futuro, calculamos λ_casa/λ_fora (mesma força de antes) e
   **sorteamos** um placar — repetido para as 10.000 simulações de uma vez
   (vetorizado em numpy, por isso roda em ~0,4s).
3. Somamos os pontos e, ao fim de cada simulação, **classificamos** os 20 times.
4. Contamos: em quantas simulações cada time foi **campeão**, ficou no **G4** ou
   caiu para a **zona de rebaixamento**.

Saída por time: `prob_titulo`, `prob_g4`, `prob_rebaixamento`, pontos e posição
médios projetados.

> **Invariantes que valem mostrar** (e que os testes verificam):
> - A soma das probabilidades de título de todos os times = **100%** (há
>   exatamente 1 campeão por simulação).
> - A soma das probabilidades de rebaixamento = **400%** (4 rebaixados).
> Se esses números não fecharem, tem bug — é um ótimo "cheque de sanidade".

Se a base não tem jogos futuros (temporada encerrada ou modo offline), a aba
avisa e não quebra.

---

## 7. As duas novas abas do dashboard

- **🎲 Simulador**: escolhe mandante e visitante → cards de probabilidade, gols
  esperados, métricas de aposta (BTTS/Over 2.5), **heatmap de placares**,
  narração e um expander com o **Monte Carlo de 10 mil jogos**.
- **🏆 Projeção**: escolhe o número de simulações → tabela com chances de
  título/G4/rebaixamento e gráficos de barras dos favoritos ao título e dos
  ameaçados de queda.

---

## 8. Limitações (honestidade intelectual — diga isto!)

- **Independência dos gols:** assumimos que os gols do mandante e do visitante
  não se influenciam. Na vida real, placares como 0×0 e 1×1 são um pouco mais
  comuns do que Poisson prevê — a **correção de Dixon-Coles** resolve isso e fica
  como evolução futura.
- **Sem contexto:** o modelo não sabe de lesões, cartões, viagem, motivação. Isso
  é justamente o tema da **Fase 5** (camada de contexto).
- **Depende da amostra:** no início da temporada, com poucos jogos, as
  estimativas são mais incertas (por isso o shrinkage).

> Gancho: "Esse simulador é **estatístico** — olha só a média de gols. Na Fase 3
> vamos deixar um **modelo aprender** padrões mais finos a partir das features
> que construímos hoje."

---

## 9. Onde está cada coisa

| Arquivo | Papel |
|---------|-------|
| `src/features/simulador.py` | modelo de Poisson, Monte Carlo, projeção |
| `src/data/coletar_tabela.py` | agora também baixa os **jogos futuros** (SCHEDULED) |
| `src/data/limpar.py` | limpa os jogos futuros |
| `app/streamlit_app.py` | abas 🎲 Simulador e 🏆 Projeção |
| `tests/test_simulador.py` | testes (somas de probabilidade, Monte Carlo, invariantes) |
