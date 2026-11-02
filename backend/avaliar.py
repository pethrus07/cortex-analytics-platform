"""Avaliação honesta do modelo de risco de falha, sobre os dados do seed.

    python avaliar.py

Não toca em banco nenhum: gera as mesmas leituras sintéticas do `seed.py`
(semente 42) e avalia o Random Forest de `app/ml/failure_prediction.py`.

Imprime quatro coisas, nesta ordem:
1. a métrica antiga, treinando e medindo nas mesmas linhas (o erro original);
2. a métrica no holdout estratificado 75/25 (o número a reportar);
3. a regra que gerou o rótulo, aplicada ao mesmo teste (o teto trivial);
4. dois testes de estresse: sem as duas features da regra, e com um ativo
   inteiro guardado para teste.

A leitura correta está no README: o dado é sintético e o rótulo é uma regra
fixa sobre temperatura e vibração, então um número alto aqui mede se o modelo
redescobre a regra, não se ele preveria falha numa planta real.
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.ml.failure_prediction import (
    FEATURES,
    SEED,
    TEST_SIZE,
    build_dataset,
    evaluate_holdout,
    new_model,
)
from seed import synthetic_events


def load_seed_dataset() -> pd.DataFrame:
    records = []
    for asset_id, (spec, events) in enumerate(synthetic_events(SEED), start=1):
        for ev in events:
            records.append({**ev, "asset_id": asset_id, "asset": spec["name"]})
    df = build_dataset(records)
    df["asset"] = [r["asset"] for r in records]
    df["status"] = [r["status"] for r in records]
    return df


def metrics(y_true, proba) -> dict:
    pred = (pd.Series(proba) >= 0.5).astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba) if len(set(y_true)) == 2 else float("nan"),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "cm": (tn, fp, fn, tp),
        "n": len(y_true),
        "pos": int(sum(y_true)),
    }


def show(title: str, m: dict) -> None:
    tn, fp, fn, tp = m["cm"]
    print(f"\n{title}")
    print(f"  n={m['n']} (falhas={m['pos']})  ROC-AUC={m['roc_auc']:.3f}  "
          f"precisão={m['precision']:.3f}  recall={m['recall']:.3f}")
    print(f"  matriz de confusão: TN={tn} FP={fp} FN={fn} TP={tp}")


def main() -> None:
    df = load_seed_dataset()
    pos = int(df["failed"].sum())
    print("Dados: sintéticos, gerados por seed.py (semente 42)")
    print(f"  {len(df)} leituras de {df['asset'].nunique()} ativos; "
          f"{pos} marcadas 'down' ({pos / len(df):.1%})")
    print("  falhas por ativo:", df.groupby("asset")["failed"].sum().to_dict())

    # 1. O erro original: treino e escore nas mesmas linhas.
    m = new_model().fit(df[FEATURES], df["failed"])
    show("1. Método antigo (treina e mede nas MESMAS linhas, não usar)",
         metrics(df["failed"].to_numpy(), m.predict_proba(df[FEATURES])[:, 1]))

    # 2. Holdout estratificado: a mesma função que a API usa.
    ev = evaluate_holdout(df)
    cm = ev["confusion_matrix"]
    print(f"\n2. Holdout ({ev['method']}), o número a reportar")
    print(f"  treino={ev['train_size']}  teste={ev['test_size']} (falhas={ev['test_positives']})  "
          f"ROC-AUC={ev['roc_auc']:.3f}  precisão={ev['precision']:.3f}  recall={ev['recall']:.3f}")
    print(f"  matriz de confusão: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")

    # Mesmo split para comparar com a regra e com a ablação.
    tr, te = train_test_split(df, test_size=TEST_SIZE, stratify=df["failed"], random_state=SEED)

    # 3. A regra que gerou o rótulo, no mesmo teste.
    rule = ((te["temperature"] > 92) | (te["vibration"] > 9)).astype(float).to_numpy()
    show("3. Regra do seed (temp > 92 ou vib > 9) no mesmo teste: o teto trivial",
         metrics(te["failed"].to_numpy(), rule))

    # 4a. Sem as duas features da regra.
    rest = ["pressure", "runtime_hours"]
    m = new_model().fit(tr[rest], tr["failed"])
    show("4a. Estresse: só pressão e horas de operação (sem temperatura e vibração)",
         metrics(te["failed"].to_numpy(), m.predict_proba(te[rest])[:, 1]))

    # 4b. Um ativo com falha e um saudável inteiros fora do treino.
    held = {"Turbina D", "Esteira E"}
    tr2, te2 = df[~df["asset"].isin(held)], df[df["asset"].isin(held)]
    m = new_model().fit(tr2[FEATURES], tr2["failed"])
    show(f"4b. Estresse: ativos {sorted(held)} inteiros só no teste",
         metrics(te2["failed"].to_numpy(), m.predict_proba(te2[FEATURES])[:, 1]))

    imp = dict(zip(FEATURES, new_model().fit(df[FEATURES], df["failed"]).feature_importances_))
    print("\nImportância das features (modelo final, histórico inteiro):")
    for f, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"  {f:14s} {v:.3f}")


if __name__ == "__main__":
    main()
