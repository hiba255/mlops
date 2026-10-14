
import pandas as pd  # Pour construire et manipuler le tableau de données
import numpy as np   # Pour générer des valeurs numériques aléatoires

# On fixe une graine aléatoire pour garantir la reproductibilité.
# Les mêmes données seront générées à chaque exécution.
np.random.seed(42)

n = 500  # Nombre de clients fictifs à générer

# Création du DataFrame : chaque colonne est une feature (variable d'entrée).
data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "salaire": np.random.randint(1200, 8000, n),
    "abonnement": np.random.choice(
        ["Basique", "Standard", "Premium"], n
    ),
    "anciennete": np.random.randint(0, 15, n),
    "consommation": np.random.randint(0, 100, n),
})

# Création de la target (variable à prédire) : churn.
# Un client est considéré comme à risque si :
# - sa consommation est inférieure à 30 ;
# - son ancienneté est inférieure à 2 ans.
data["churn"] = (
    (data["consommation"] < 30) &
    (data["anciennete"] < 2)
).astype(int)  # Conversion de True/False en 1/0

# Sauvegarde des données dans un fichier CSV.
data.to_csv("churn_data.csv", index=False)

# Vérification des résultats.
print(data.head())
print(f"Taux de churn : {data['churn'].mean():.1%}")

