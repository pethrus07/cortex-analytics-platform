"""Risco de falha por Random Forest, agregado por ativo.

Aprende a assinatura de sensor de uma leitura marcada `down` e devolve, por
ativo, a probabilidade média de as leituras dele se parecerem com isso.

Antes de pontuar, o modelo é avaliado num conjunto de teste separado
(holdout estratificado, semente fixa). A métrica volta na resposta da API e
`avaliar.py` imprime o relatório completo. Depois da avaliação o modelo é
retreinado com o histórico inteiro para pontuar os ativos, que é a prática
padrão: a avaliação mede o método, o modelo final usa todo o dado.

Cuidado com a leitura do número: é um retrato do histórico, não uma previsão
do futuro. O porquê está no comentário de `build_dataset`.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from .. import models

FEATURES = ["temperature", "vibration", "pressure", "runtime_hours"]
SEED = 42
TEST_SIZE = 0.25
# Mínimo de leituras da classe rara para a avaliação fazer sentido: com o
# split de 25%, menos do que isso deixa 1 ou 0 falhas no teste.
MIN_POSITIVES_FOR_EVAL = 8


def build_dataset(records: list[dict]) -> pd.DataFrame:
    """Monta o DataFrame de treino a partir de leituras (dicts).

    Duas ressalvas que mudam como o número deve ser lido:
    1) `failed` é o status da própria leitura, não uma falha que veio depois
       dela, então o modelo classifica o estado presente do ativo. Para virar
       previsão de verdade, o rótulo teria de olhar uma janela à frente do
       evento (por exemplo: "houve down nas próximas 24h?").
    2) No dado sintético do `seed.py`, `down` é uma regra fixa sobre
       temperatura e vibração. O modelo só precisa redescobrir essa regra.
    """
    df = pd.DataFrame(
        [
            {
                **{f: r[f] for f in FEATURES},
                "asset_id": r.get("asset_id"),
                "failed": 1 if r["status"] == "down" else 0,
            }
            for r in records
        ]
    )
    return df


def new_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=SEED
    )


def evaluate_holdout(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED) -> dict:
    """Treina num pedaço e mede no outro (split estratificado, semente fixa).

    Devolve as métricas no conjunto de teste e o tamanho de cada conjunto.
    Se a classe rara for pequena demais, devolve `evaluated: False` com o
    motivo, em vez de um número que não quer dizer nada.
    """
    positives = int(df["failed"].sum())
    if positives < MIN_POSITIVES_FOR_EVAL or (len(df) - positives) < MIN_POSITIVES_FOR_EVAL:
        return {
            "evaluated": False,
            "reason": f"classe rara pequena demais para holdout ({positives} falhas)",
        }

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["failed"], test_size=test_size, stratify=df["failed"], random_state=seed
    )
    model = new_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    return {
        "evaluated": True,
        "method": f"holdout estratificado {int((1 - test_size) * 100)}/{int(test_size * 100)}, seed {seed}",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_positives": int(y_test.sum()),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def predict_failures(db: Session) -> dict:
    """Avalia o modelo no holdout e depois estima o risco de cada ativo.

    O rótulo de falha vem do histórico (status == 'down'). Escolhi Random
    Forest pela robustez e por dar a importância das features de graça,
    o que ajuda a explicar o resultado pro time de operação.
    """
    rows = db.query(models.Event).all()
    if len(rows) < 20:
        return {"trained": False, "reason": "poucos dados (mínimo 20 eventos)"}

    df = build_dataset(
        [{**{f: getattr(r, f) for f in FEATURES}, "asset_id": r.asset_id, "status": r.status} for r in rows]
    )

    # Sem as duas classes (falhou / não falhou) não dá pra treinar.
    if df["failed"].nunique() < 2:
        return {"trained": False, "reason": "sem variabilidade de classes no histórico"}

    evaluation = evaluate_holdout(df)

    # Modelo final: todo o histórico. O escore por ativo inclui leituras que o
    # modelo viu no treino, por isso ele descreve o histórico; a métrica de
    # generalização é a do holdout acima.
    model = new_model()
    model.fit(df[FEATURES], df["failed"])
    df["risk"] = model.predict_proba(df[FEATURES])[:, 1]

    db.query(models.Prediction).filter(models.Prediction.target == "failure_risk").delete()

    summary = df.groupby("asset_id").agg(risk=("risk", "mean")).reset_index()

    scored = 0
    for _, row in summary.iterrows():
        risk = float(row["risk"])
        if risk >= 0.66:
            label = "high_risk"
        elif risk >= 0.33:
            label = "medium_risk"
        else:
            label = "low_risk"

        db.add(models.Prediction(
            asset_id=int(row["asset_id"]),
            model="random_forest",
            target="failure_risk",
            score=round(risk, 4),
            label=label,
        ))

        if label == "high_risk":
            db.add(models.Alert(
                asset_id=int(row["asset_id"]),
                severity="critical",
                message=f"Alto risco de falha previsto ({risk:.0%}).",
                source="failure_model",
            ))
        scored += 1

    db.commit()

    importances = {f: round(float(v), 3) for f, v in zip(FEATURES, model.feature_importances_)}
    return {
        "trained": True,
        "assets_scored": scored,
        "feature_importances": importances,
        "evaluation": evaluation,
    }
