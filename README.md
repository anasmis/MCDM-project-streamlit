# Atelier de décision — MCDM précis et flou

Application Streamlit en français pour comparer des alternatives dans différents domaines : performances numériques ou floues triangulaires, critères à maximiser ou à minimiser, méthodes au choix et analyse de sensibilité.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Le point d’entrée à la racine lance l’application dans [`mcdm_streamlit_app`](mcdm_streamlit_app). La [documentation technique](mcdm_streamlit_app/README.md) décrit les formats de fichiers, les conventions de calcul et les tests.

- **Neuf méthodes de classement précis :** TOPSIS, SAW/WSM, WPM, WASPAS, VIKOR, MOORA, COPRAS, EDAS et PROMETHEE II.
- **Classements flous :** TOPSIS flou et somme pondérée floue.
- **Pondération :** poids égaux, manuels, entropie, CRITIC, AHP, BWM linéaire, poids manuels flous et AHP flou.
- **Aide intégrée :** principe de chaque méthode, étapes, formule, interprétation du score et hypothèses, dans l’onglet « Comprendre les méthodes » et à côté des résultats.
- **Parcours :** saisir ou importer un problème, comparer les classements, consulter les calculs, faire varier les poids et exporter le projet ou les résultats.

L’application traite les matrices de décision finies. Chaque méthode possède ses hypothèses ; certaines formulations demandent un autre modèle. Les projets JSON existants restent compatibles. L’ancienne application d’inventaire est conservée dans `mcdm_streamlit_app/legacy_inventory_app.py`.
