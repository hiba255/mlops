
import mlflow

# ===== 1. Récupérer les runs de l'expérience =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# ===== 2. Vérifier que des runs existent =====
if runs.empty:
    print("Aucun run trouvé dans l'expérience churn-prediction.")

else:
    # ===== 3. Afficher les résultats =====
    colonnes = [
        "tags.mlflow.runName",
        "params.n_estimators",
        "params.max_depth",
        "metrics.accuracy",
        "metrics.f1"
    ]

    print(runs[colonnes].to_string(index=False))

    # ===== 4. Identifier le meilleur run =====
    meilleur = runs.iloc[0]

    print()

    print(
        "Meilleur run :",
        meilleur["tags.mlflow.runName"],
        "| F1 :",
        round(meilleur["metrics.f1"], 3),
        "| ID :",
        meilleur["run_id"][:8],
        "..."
    )
