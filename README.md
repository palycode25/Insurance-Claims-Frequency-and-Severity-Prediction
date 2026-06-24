# Apport de la télématique automobile dans la modélisation actuarielle

Projet de modélisation en assurance — Mai 2026

## Résumé

Ce projet explore l’apport des données télématiques automobiles dans deux problématiques actuarielles complémentaires :

- la segmentation du risque (classification des conducteurs à risque),
- la tarification d’assurance (estimation de la prime pure via fréquence × sévérité).

Le dataset utilisé est un jeu de données synthétique issu de So, Boucher & Valdez (2021), simulant un portefeuille d’assurance automobile basé sur 100 000 polices d’assurance “usage-based”.

L’objectif est de comparer :
- variables traditionnelles vs télématiques
- modèles linéaires vs non-linéaires
- machine learning vs modèles actuariels classiques

---

## Résultats principaux

- Ajout des variables télématiques : +0,11 AUC (0,68 → 0,79 en régression logistique)
- Passage aux modèles non-linéaires (XGBoost) : +0,08 AUC supplémentaire
- Gain total entre baseline et meilleur modèle : +0,19 AUC (~+29%)

### Segmentation du risque
- Meilleur modèle : XGBoost + cost-sensitive learning
- AUC test ≈ 0,89
- Excellente capacité de ranking des conducteurs à risque

### Tarification actuarielle
- Pipeline Poisson × Gamma
- Erreur agrégée de prime ≈ 1,9%
- Forte discrimination des déciles de risque (~15× entre extrêmes)

---

## Méthodologie

### Segmentation
- Régression logistique (baseline)
- XGBoost (non-linéaire)
- Gestion du déséquilibre :
  - SMOTE
  - cost-sensitive learning
- Calibration : Platt scaling

### Tarification
- Modèle de fréquence : Poisson
- Modèle de sévérité : Gamma
- Prime pure = fréquence × sévérité

---

## Données

- Source : So, Boucher & Valdez (2021)
- 100 000 polices synthétiques
- 52 variables :
  - 11 traditionnelles
  - 39 télématiques
  - 2 cibles :
    - NB_Claim
    - AMT_Claim

Déséquilibre :
- ~95% sans sinistre
- ~5% avec sinistre

---

## Stack technique

- Python 3.12
- pandas, numpy
- scikit-learn
- xgboost
- statsmodels
- scipy
- matplotlib / plotly

---

## Structure du projet

data/           données brutes  
notebooks/      analyses et modèles  
app/            application Streamlit  
rapport/        rapport PDF  
presentation/   slides  

---

## Insights clés

- La télématique améliore fortement la prédiction du risque
- Les modèles non-linéaires capturent mieux les comportements de conduite
- SMOTE est moins performant que le cost-sensitive learning pour les arbres
- ML et actuariat sont complémentaires :
  - ML → scoring et segmentation
  - Actuariat → tarification robuste

---

## Références

- So, Boucher & Valdez (2020, 2021)
- Denuit et al. (2007)
- Akerlof (1970)
- Chawla et al. (2002)
- Niculescu-Mizil & Caruana (2005)
- Stocksieker (2024)

---

## Auteur

Fatoumata Rami Eunice
