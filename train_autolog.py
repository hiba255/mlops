
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ===== 1. Charger les données =====
df = pd.read_csv("churn_data.csv")

X = pd.get_dummies(
    df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
    drop_first=True
)

y = df["churn"]


# ===== 2. Diviser les données =====
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ===== 3. Configurer MLflow =====
mlflow.set_experiment("churn-prediction")

# Activer l'enregistrement automatique
mlflow.sklearn.autolog(
    serialization_format="cloudpickle"
)


# ===== 4. Entraîner le modèle =====
with mlflow.start_run(run_name="essai-auto"):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Calculer l'accuracy sur les données de test
    score = model.score(X_test, y_test)

    print("Score test :", round(score, 3))
