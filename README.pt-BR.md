# Cortex Analytics Platform

[English](README.md) · Português

![status](https://img.shields.io/badge/status-concluído%20(desafio%20técnico%2C%202026--06)-lightgrey)
![backend](https://img.shields.io/badge/backend-FastAPI-009688)
![frontend](https://img.shields.io/badge/frontend-React-61DAFB)
![ml](https://img.shields.io/badge/ML-scikit--learn-F7931E)
![license](https://img.shields.io/badge/license-MIT-green)

Plataforma analítica full-stack feita como desafio técnico de uma vaga de desenvolvedor.
Ingere telemetria de ativos industriais (temperatura, vibração, pressão, horas de operação),
roda um ETL em Pandas, dois modelos de scikit-learn (anomalia e risco de falha) e mostra tudo
num dashboard React.

## Status

**Concluído (desafio técnico, 2026-06).** Não continuou depois da entrega. Em 2026-11 o
modelo de risco de falha ganhou avaliação em holdout, um script de avaliação e testes.

## O problema

Telemetria de máquina costuma ficar espalhada, sem monitoramento central e sem nenhuma
camada preditiva. A falha só aparece quando o equipamento já parou. O desafio pedia uma
camada única que ingere a telemetria, calcula métricas operacionais, gera alertas e estima
o risco de falha por ativo.

## Avaliação do modelo de risco de falha

**Os dados são sintéticos.** O `backend/seed.py` gera 5 ativos com 60 leituras cada
(semente 42). Três ativos são saudáveis e dois pioram com o tempo. Uma leitura é marcada
`down` por uma regra fixa: temperatura acima de 92 ou vibração acima de 9. São 300 leituras,
23 com falha (7,7%), todas em dois ativos.

A primeira versão treinava e media nas mesmas linhas. Agora o modelo é avaliado num holdout
estratificado 75/25 com semente fixa, e só depois é retreinado com tudo para pontuar os
ativos. A resposta de `POST /api/ml/failures/run` traz a avaliação junto.

Resultado de `python avaliar.py`:

| Avaliação | n teste (falhas) | ROC-AUC | Precisão | Recall | Matriz (TN, FP, FN, TP) |
|---|---|---|---|---|---|
| Método antigo: treino = teste (não usar) | 300 (23) | 1,000 | 1,000 | 1,000 | 277, 0, 0, 23 |
| **Holdout estratificado 75/25** | **75 (6)** | **1,000** | **1,000** | **0,833** | **69, 0, 1, 5** |
| A própria regra do seed, no mesmo teste | 75 (6) | 1,000 | 1,000 | 1,000 | 69, 0, 0, 6 |
| Sem temperatura e vibração | 75 (6) | 0,949 | 0,333 | 0,167 | 67, 2, 5, 1 |
| Dois ativos inteiros só no teste | 120 (12) | 0,985 | 1,000 | 0,500 | 108, 0, 6, 6 |

Como ler:

- **O número alto não prova nada sobre o mundo real.** O rótulo é uma regra sobre duas das
  features, e a regra sozinha acerta 100%. O que o holdout mostra é que o Random Forest
  redescobre a regra em leituras que não viu (errou 1 de 6 falhas).
- **O teste tem só 6 falhas.** Um acerto a mais ou a menos move o recall em 17 pontos.
  O intervalo de confiança é largo e o número não deve ser citado sem esse contexto.
- **Tirando as duas features da regra, o recall cai para 0,167.** A AUC continua alta
  (0,949) porque os ativos que pioram também têm pressão maior no gerador: o modelo separa
  os ativos, não o momento da falha.
- **Com ativos inteiros fora do treino, o recall cai para 0,500.** Ver um ativo novo é mais
  difícil do que ver leituras novas de um ativo conhecido.
- **O rótulo é o estado presente, não uma falha futura.** Para virar previsão de verdade,
  o rótulo teria de olhar uma janela à frente (por exemplo: "houve `down` nas próximas
  24 h?"), e o dado teria de ser real.

Importância das features no modelo final: temperatura 0,573, vibração 0,238, pressão 0,153,
horas de operação 0,036. Isso também é reflexo da regra do gerador.

## Stack

| Camada | Escolha | Por quê |
|---|---|---|
| Front-end | React + Vite + TailwindCSS + Recharts | build rápido e série temporal sem reinventar gráfico |
| Back-end | FastAPI (Python) | mesma linguagem do pipeline de dados; OpenAPI em `/docs` |
| Banco | PostgreSQL (docker-compose) ou SQLite (padrão local) | SQLite deixa rodar sem Docker |
| ETL | Pandas | agregação por ativo em uma passagem |
| ML | scikit-learn: Isolation Forest + Random Forest | anomalia sem rótulo; risco com importância de features |
| Infra | Docker + docker-compose | banco, API e front com um comando |

```
eventos brutos ──► ETL (Pandas) ──► banco ──► modelos de ML ──► dashboard
 (sensores)        agrega e grava   métricas   anomalia e risco   KPIs e gráficos
                                    e alertas
```

## Rodar localmente

### Sem Docker (SQLite)

```bash
git clone https://github.com/pethrus07/desafio-tecnico-dev-.git
cd desafio-tecnico-dev-/backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python seed.py                  # dados sintéticos de demonstração (apaga e recria o banco)
uvicorn app.main:app --reload   # API em http://localhost:8000, docs em /docs

# em outro terminal
cd ../frontend
npm install
npm run dev                     # dashboard em http://localhost:5173
```

### Com Docker (PostgreSQL)

```bash
docker compose up --build
docker compose exec backend python seed.py
# frontend: http://localhost:5173 | backend: http://localhost:8000
```

No dashboard, o botão **"Rodar ETL + Modelos"** roda o ETL e os dois modelos e preenche
KPIs, telemetria, predições e alertas.

## Testes

```bash
cd backend
python -m pytest     # 9 testes: seed, ETL idempotente, holdout e os dois modelos
python avaliar.py    # o relatório de avaliação da tabela acima
```

Os testes usam SQLite em memória e não tocam o banco local nem o Postgres. O front-end
não tem testes.

## Pastas

```
backend/
  app/
    main.py              FastAPI, CORS e routers
    database.py          engine (Postgres ou SQLite) e sessão
    models.py            Asset, Event, Metric, Alert, Prediction
    routers/             events, metrics/etl, alerts, predictions (ML)
    etl/pipeline.py      ETL em Pandas, idempotente
    ml/anomaly.py        Isolation Forest
    ml/failure_prediction.py   Random Forest com holdout
  seed.py                gerador de dados sintéticos
  avaliar.py             avaliação do modelo de risco
  tests/                 pytest
frontend/
  src/App.jsx            dashboard
  src/components/        KPICard, MetricsChart, AlertsPanel, PredictionsPanel
docker-compose.yml
```

## Arquitetura

Casos de uso, processo e C4 em [docs/arquitetura/](docs/arquitetura/README.md).

## Decisões e limites conhecidos

- Dado sintético; a métrica mede o método, não o desempenho numa planta.
- Sem migrations: o banco nasce de `create_all`, e o seed apaga e recria tudo.
- Treino a cada requisição, sem modelo salvo.
- O ETL lê a tabela de eventos inteira; não há janela de tempo.
- O gráfico de telemetria usa a posição na lista no eixo X e mistura ativos.
- CORS aberto para desenvolvimento.
- Nunca foi publicado em produção.

## Roadmap

Não continuado. Tempo real, observabilidade e ingestão em escala ficaram no plano original
e não foram feitos.

## Licença

MIT, ver [LICENSE](LICENSE).
