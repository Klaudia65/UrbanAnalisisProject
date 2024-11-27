# UrbanAnalisisProject

# UrbanAnalisisProject

## Objectives
- Analyze the relationship between these variables and potentially predict or classify certain neighborhood characteristics.
- Understand the relationships between variables.
- Predict pollution.
- Identify groups of neighborhoods.

## Dataset

### Variables
- **Total area (km²)**
- **Vegetation area (km²)**
- **Fine particles pollution pm25 (mcg/m³)**

### Description
- **79 neighborhoods/roundabouts** in 3 different cities:
  - **NYC**: 42
  - **Berlin**: 12
  - **Seoul**: 25

The data was collected from various sources by myself, I had to process them and put them in a single CSV file (`data/data.csv`).

### Data Sources
- **Surface area**: [Wikipedia](https://www.wikipedia.org/)

- **Seoul**: [Seoul Open Data Plaza](https://data.seoul.go.kr/dataService/boardList.do)
- **New York City**: [NYC Health](https://a816-dohbesp.nyc.gov/IndicatorPublic/)
- **Berlin**:
  - [Oasis Hub](https://oasishub.co/dataset/berlin-germany-district-level-environmental-database/resource/be1739c5-1c58-4199-be5a-ea6f15299cb5?inner_span=True)
  - [Berlin Environmental Data](https://www.berlin.de/sen/uvk/_assets/natur-gruen/stadtgruen/daten-und-fakten/ausw_14.pdf)


<!-->

Analyse Non Supervisée

    Clustering:
        Objectif: Regrouper les quartiers en fonction de leur similarité sur les variables.
        Algorithmes: K-means, DBSCAN, Hierarchical Clustering.
        Intérêt: Identifier des clusters de quartiers avec des caractéristiques environnementales similaires (par exemple, quartiers très verts avec une faible pollution, quartiers densément peuplés avec une forte pollution).
    Analyse en Composantes Principales (ACP):
        Objectif: Réduire la dimensionnalité des données et identifier les principales sources de variation.
        Intérêt: Visualiser les quartiers dans un espace à plus faible dimension pour mieux comprendre leurs relations.

Analyse Supervisée

    Régression:
        Objectif: Prédire la valeur de la pollution (pm2.5) en fonction de la superficie et de la surface végétalisée.
        Algorithmes: Régression linéaire, Régression Ridge, Régression Lasso, Arbres de régression, Forêts aléatoires.
        Intérêt: Comprendre l'impact de la superficie et de la surface végétalisée sur la qualité de l'air.
    Classification:
        Objectif: Classer les quartiers en fonction d'un critère (par exemple, quartiers à faible, moyenne ou forte pollution).
        Algorithmes: Régression logistique, Arbres de décision, Forêts aléatoires, SVM.
        Intérêt: Créer un modèle prédictif pour identifier les quartiers à risque en termes de pollution.

Guidage dans vos Choix

1. Définir votre objectif principal:

    Comprendre les relations: Si vous cherchez à comprendre les relations entre les variables, l'analyse en composantes principales et la régression linéaire sont de bons choix.
    Prédire la pollution: Pour prédire la pollution, la régression (linéaire, non-linéaire) est idéale.
    Identifier des groupes de quartiers: Le clustering vous permettra de regrouper les quartiers en fonction de leurs caractéristiques.

2. Préparer vos données:

    Nettoyage: Vérifier la présence de valeurs manquantes, d'outliers.
    Normalisation: Normaliser les variables si nécessaire (par exemple, si les échelles sont très différentes).
    Encodage: Si vous avez des variables catégorielles (par exemple, le nom de la ville), il faudra les encoder.

3. Choisir les bons algorithmes:

    Expérimenter: Tester différents algorithmes pour voir lequel donne les meilleurs résultats.
    Évaluer les performances: Utiliser des métriques appropriées (RMSE, MAE, précision, rappel, F1-score).
    Considérer la complexité: Pour les grands datasets, les algorithmes plus rapides (comme les arbres de décision ou les forêts aléatoires) peuvent être préférables.

4. Interpréter les résultats:

    Visualiser: Utiliser des graphiques pour mieux comprendre les résultats.
    Valider: Valider les modèles sur un jeu de données de test.
