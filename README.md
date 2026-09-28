# Arbitrage

Application française de décision multicritère. Comparez vos alternatives, justifiez la pondération des critères et observez l'effet du choix de méthode sur le classement.

## Lancer l'application

Dans PowerShell, placez-vous dans le dossier du projet et lancez :

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Sous Windows, double-cliquez sur `Lancer Arbitrage.bat` pour lancer l'application.

Le point d'entrée Streamlit est `app.py`. L'application s'ouvre dans votre navigateur sur une adresse locale.

## Parcours

1. Décrivez votre problème, indiquez si chaque critère est à maximiser ou à minimiser et complétez la matrice des performances.
2. Choisissez une méthode de pondération subjective ou objective. Les poids calculés totalisent 100 %.
3. Classez les alternatives avec TOPSIS ou WSM, comparez les deux résultats et faites varier un poids pour étudier la sensibilité.

Un exemple de sélection de fournisseur est proposé au premier lancement. Les fichiers JSON et CSV, exemples téléchargeables, reprennent le cadrage d'un nouveau problème. Les résultats s'exportent individuellement en CSV ou ensemble dans une archive ZIP.

### Fichiers de données

Le CSV contient une première colonne nommée `Alternative`, puis une colonne numérique par critère. Les colonnes décimales, les virgules, les points-virgules et les tabulations sont acceptés. Importer un CSV crée des critères réglés par défaut sur « Maximiser » ; vérifiez et corrigez leur sens avant le calcul. Le JSON enregistré par l'application conserve les noms, les sens, les unités, la description et toutes les performances.

## Méthodes et conventions

- **AHP** : poids par vecteur propre principal d'une matrice réciproque. Affiche l'indice et le ratio de cohérence ; un ratio supérieur à 10 % demande une confirmation.
- **BWM** : poids obtenus par programmation linéaire à partir des comparaisons du meilleur critère avec les autres et des autres avec le moins important. Le résidu optimal est affiché séparément ; il ne s'agit pas du ratio de cohérence AHP.
- **CRITIC** : normalisation min–max orientée, écart-type de population et corrélations entre critères variables. Les critères constants reçoivent un poids nul. L'application signale les critères variables impossibles à départager.
- **Entropie** : normalisation min–max orientée, proportions par critère, puis divergence par rapport à l'entropie maximale. La convention `0 × ln(0) = 0` est appliquée ; les critères constants reçoivent un poids nul.
- **WSM** : somme pondérée de performances min–max orientées ; une performance constante est représentée par 1 pour toutes les alternatives.
- **TOPSIS** : normalisation vectorielle, pondération, distance euclidienne à l'idéal et à l'anti-idéal, avec prise en compte des critères à minimiser. La proximité relative appartient à [0, 1] ; lorsque les deux distances sont nulles, le score conventionnel est 0,5.

Le premier rang est attribué avec une tolérance numérique définie à 12 décimales ; les ex æquo partagent ce rang. Les scores sont propres à leur méthode. Comparer TOPSIS et WSM par rang, et non par score.

## Organisation

```text
app.py                Point d'entrée Streamlit
mcdm/models.py        Problème, critères et résultats
mcdm/weighting/       AHP, BWM, CRITIC et entropie
mcdm/ranking/         WSM et TOPSIS
mcdm/analysis.py      Recommandations et sensibilité
mcdm/io.py            Exemples, import et export
ui/pages.py           Parcours et pages Streamlit
ui/charts.py          Graphiques interactifs
ui/style.css          Styles de l'application
```
