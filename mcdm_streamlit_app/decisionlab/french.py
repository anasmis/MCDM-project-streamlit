"""French presentation labels and explanations; project identifiers remain stable."""

import re

from .project import example

LABELS = {
    "Crisp": "Valeurs précises",
    "Fuzzy": "Logique floue",
    "Benefit": "Maximiser",
    "Cost": "Minimiser",
    "Equal": "Poids égaux",
    "Manual": "Pondération manuelle",
    "Entropy": "Entropie",
    "Fuzzy manual": "Pondération manuelle floue",
    "Fuzzy AHP": "AHP flou",
    "Fuzzy TOPSIS": "TOPSIS flou",
    "Fuzzy SAW": "Somme pondérée floue",
    "Supplier selection": "Sélection de fournisseurs",
    "Location planning": "Choix d’implantation",
    "Fuzzy assessment": "Évaluation floue",
    "Blank problem": "Nouveau problème",
    "Very low": "Très faible",
    "Low": "Faible",
    "Medium": "Moyen",
    "High": "Élevé",
    "Very high": "Très élevé",
    "normalized": "Matrice normalisée",
    "weighted": "Matrice normalisée et pondérée",
    "positive_distance": "Distance à la solution idéale",
    "negative_distance": "Distance à la solution anti-idéale",
    "utility": "Écart global S",
    "regret": "Regret maximal R",
    "acceptable_advantage": "Condition d’avantage acceptable",
    "acceptable_stability": "Condition de stabilité acceptable",
    "fuzzy_totals": "Sommes triangulaires pondérées",
}


def label(value):
    return LABELS.get(value, value)


def detail_label(method, name):
    if method == "EDAS" and name in {"positive_distance", "negative_distance"}:
        return (
            "Écart favorable pondéré à la moyenne"
            if name == "positive_distance"
            else "Écart défavorable pondéré à la moyenne"
        )
    return label(name)


def localized_example(name="Supplier selection"):
    p = example(name)
    translations = {
        "Price": "Prix",
        "Quality": "Qualité",
        "Delivery days": "Délai de livraison",
        "Sustainability": "Durabilité",
        "Rent": "Loyer",
        "Access": "Accessibilité",
        "Capacity": "Capacité",
        "Resilience": "Résilience",
        "North": "Nord",
        "Central": "Centre",
        "South": "Sud",
        "West": "Ouest",
    }
    p["title"] = label(name)
    p["alternatives"] = [
        translations.get(n, n.replace("Supplier", "Fournisseur")) for n in p["alternatives"]
    ]
    for c in p["criteria"]:
        c["name"] = translations.get(c["name"], c["name"].replace("Criterion", "Critère"))
    p["values"] = [[label(v) for v in row] for row in p["values"]]
    p["scale"] = {label(k): v for k, v in p["scale"].items()}
    return p


RANKING_GUIDES = {
    "TOPSIS": {
        "summary": "Choisir l’alternative la plus proche de l’idéal et la plus éloignée de l’anti-idéal.",
        "steps": [
            "Normaliser chaque critère par la longueur de son vecteur pour rendre les unités comparables, puis appliquer les poids.",
            "Construire l’idéal avec la meilleure valeur de chaque critère et l’anti-idéal avec la pire, selon le sens maximiser ou minimiser.",
            "Calculer les distances euclidiennes D⁺ et D⁻, puis le coefficient de proximité C.",
        ],
        "formula": r"C_i=\frac{D_i^-}{D_i^++D_i^-}",
        "interpretation": "Un score proche de 1 indique une meilleure proximité de l’idéal ; un score proche de 0 indique une proximité de l’anti-idéal. Si les deux distances sont nulles, le score conventionnel est 0,5.",
        "limits": "Les valeurs négatives sont acceptées. Les résultats dépendent des alternatives présentes et des poids : le score n’est pas une probabilité de réussite.",
    },
    "SAW / WSM": {
        "summary": "Additionner les performances normalisées, chacune multipliée par son importance.",
        "steps": [
            "Normaliser chaque critère entre 0 et 1 par min–max, en inversant les critères à minimiser.",
            "Multiplier chaque performance par le poids du critère.",
            "Additionner ces contributions pour obtenir le score de l’alternative.",
        ],
        "formula": r"S_i=\sum_j w_jr_{ij}",
        "interpretation": "Le score le plus élevé est le meilleur. Par exemple, des performances normalisées de 0,8 et 0,5, pondérées à 60 % et 40 %, donnent 0,68.",
        "limits": "Méthode compensatoire : une excellente performance peut compenser une faiblesse. Un critère constant vaut 1 pour toutes les alternatives.",
    },
    "WPM": {
        "summary": "Combiner les critères par un produit pondéré plutôt que par une somme.",
        "steps": [
            "Pour un bénéfice, utiliser x/max(x) ; pour un coût, utiliser min(x)/x.",
            "Élever chaque rapport à la puissance correspondant au poids du critère.",
            "Multiplier les rapports pondérés et classer par score décroissant.",
        ],
        "formula": r"P_i=\prod_j r_{ij}^{w_j}",
        "interpretation": "Un produit élevé est préférable. Une faible performance pénalise le produit, même si les autres critères sont bons.",
        "limits": "Toutes les valeurs des critères de poids non nul doivent être strictement positives. Ne pas ajouter une constante arbitraire pour contourner cette condition : cela change les rapports.",
    },
    "WASPAS": {
        "summary": "Combiner une somme pondérée et un produit pondéré.",
        "steps": [
            "Normaliser les performances par les rapports bénéfice/coût, comme pour WPM.",
            "Calculer la somme pondérée S et le produit pondéré P.",
            "Les combiner avec λ : 1 donne uniquement la somme, 0 donne uniquement le produit, et 0,5 partage les deux approches.",
        ],
        "formula": r"Q_i=\lambda S_i+(1-\lambda)P_i",
        "interpretation": "Le score le plus élevé est préférable. Faire varier λ permet de vérifier si le choix reste stable entre les deux façons d’agréger.",
        "limits": "Valeurs strictement positives requises. La somme utilise ici une normalisation par rapports, différente du min–max de SAW/WSM.",
    },
    "VIKOR": {
        "summary": "Rechercher un compromis entre la satisfaction globale et le critère le plus défavorable.",
        "steps": [
            "Mesurer, pour chaque critère, l’écart normalisé à la meilleure performance observée, puis appliquer les poids.",
            "Calculer S, la somme des écarts, et R, le plus grand écart pondéré.",
            "Normaliser S et R et les combiner dans Q. Le paramètre v favorise la satisfaction globale lorsqu’il se rapproche de 1.",
        ],
        "formula": r"Q_i=v\frac{S_i-S_{min}}{S_{max}-S_{min}}+(1-v)\frac{R_i-R_{min}}{R_{max}-R_{min}}",
        "interpretation": "Le plus petit Q est le meilleur rang. Un terme dont le dénominateur est nul vaut 0. L’application vérifie aussi l’avantage et la stabilité acceptables ; si une condition manque, le premier rang ne suffit pas à recommander un choix unique.",
        "limits": "Les critères constants ne contribuent pas aux écarts. Les valeurs négatives sont acceptées. Examiner les alternatives de tête comme un ensemble de compromis lorsque les conditions ne sont pas réunies.",
    },
    "MOORA": {
        "summary": "Mettre en balance les bénéfices et les coûts après normalisation vectorielle.",
        "steps": [
            "Diviser chaque colonne par sa norme euclidienne.",
            "Multiplier les performances normalisées par leurs poids.",
            "Additionner les bénéfices et soustraire les coûts.",
        ],
        "formula": r"Y_i=\sum_{j\in B}w_jr_{ij}-\sum_{j\in C}w_jr_{ij}",
        "interpretation": "Le plus grand Y est préférable. Un score négatif reste valide : c’est la comparaison entre alternatives qui compte.",
        "limits": "Valeurs signées acceptées. La compensation entre bénéfices et coûts dépend des poids ; un score n’est pas comparable à celui d’une autre méthode.",
    },
    "COPRAS": {
        "summary": "Évaluer l’utilité relative en combinant les bénéfices et une pénalité liée aux coûts.",
        "steps": [
            "Diviser chaque colonne par sa somme et appliquer les poids.",
            "Calculer S⁺, la somme des bénéfices, et S⁻, la somme des coûts pondérés.",
            "Combiner S⁺ avec une contribution inversement proportionnelle à S⁻, puis rapporter l’utilité à la meilleure utilité obtenue.",
        ],
        "formula": r"Q_i=S_i^+ + \frac{\sum_k S_k^-}{S_i^-\sum_k 1/S_k^-}",
        "interpretation": "Le score affiché Q/max(Q) le plus élevé est préférable ; le meilleur vaut 1. En l’absence de critères à minimiser, seule la somme des bénéfices est utilisée.",
        "limits": "Les valeurs actives doivent être strictement positives. Les poids nuls sont exclus avant les calculs.",
    },
    "EDAS": {
        "summary": "Comparer les alternatives à une solution moyenne plutôt qu’à une solution idéale.",
        "steps": [
            "Calculer la moyenne de chaque critère.",
            "Mesurer les écarts favorables et défavorables à cette moyenne en tenant compte du sens du critère, puis les pondérer.",
            "Normaliser les sommes d’écarts favorables et défavorables et calculer leur score d’évaluation moyen.",
        ],
        "formula": r"AS_i=\frac{NSP_i+NSN_i}{2}",
        "interpretation": "Un score élevé signifie davantage d’écarts favorables et moins d’écarts défavorables. Si toutes les alternatives sont identiques, le score conventionnel vaut 0,5.",
        "limits": "Les performances actives doivent être strictement positives. La solution moyenne change lorsque des alternatives sont ajoutées ou retirées.",
    },
    "PROMETHEE II": {
        "summary": "Comparer chaque alternative aux autres et mesurer son bilan de préférences.",
        "steps": [
            "Comparer les alternatives deux à deux sur chaque critère : une amélioration stricte vaut une préférence de 1, une égalité vaut 0.",
            "Pondérer ces préférences pour calculer le flux sortant, ce que l’alternative domine, et le flux entrant, ce qui la domine.",
            "Soustraire le flux entrant du flux sortant pour obtenir le flux net.",
        ],
        "formula": r"\phi_i=\phi_i^+-\phi_i^-",
        "interpretation": "Le flux net le plus élevé est préférable. Un flux positif indique un bilan de préférences favorable face aux autres alternatives.",
        "limits": "Cette version utilise la fonction de préférence usuelle, sans seuil d’indifférence : un petit avantage compte comme un grand avantage sur un même critère. Les valeurs négatives sont acceptées.",
    },
    "Fuzzy TOPSIS": {
        "summary": "Appliquer la proximité à l’idéal tout en conservant l’incertitude des performances triangulaires.",
        "steps": [
            "Représenter chaque performance par un triangle (l, m, u) : borne inférieure, valeur la plus plausible et borne supérieure.",
            "Normaliser les bénéfices par le plus grand u ; normaliser les coûts par min(l)/(u,m,l). Multiplier par les poids, éventuellement triangulaires.",
            "Construire les idéaux à partir des bornes pondérées observées. Additionner les distances de sommets sur les critères, puis calculer le coefficient de proximité.",
        ],
        "formula": r"d(a,b)=\sqrt{\frac{(l_a-l_b)^2+(m_a-m_b)^2+(u_a-u_b)^2}{3}},\qquad C_i=\frac{D_i^-}{D_i^++D_i^-}",
        "interpretation": "Un C élevé est préférable. Les trois bornes interviennent jusqu’au calcul des distances ; deux triangles de même centre peuvent donc donner des résultats différents.",
        "limits": "Triangles non négatifs ; borne inférieure strictement positive pour les coûts. Cette variante emploie les idéaux aux bornes observées et une somme de distances par critère : sa limite précise peut différer du TOPSIS euclidien.",
    },
    "Fuzzy SAW": {
        "summary": "Additionner des performances floues pondérées et interpréter le triangle agrégé.",
        "steps": [
            "Convertir les termes linguistiques en triangles et normaliser les rapports bénéfice/coût.",
            "Multiplier les triangles par les poids et additionner les contributions critère par critère.",
            "Défuzzifier uniquement le triangle final par son centre de gravité, la moyenne de ses trois bornes.",
        ],
        "formula": r"\widetilde S_i=\sum_j\widetilde w_j\widetilde r_{ij},\qquad S_i=\frac{l_i+m_i+u_i}{3}",
        "interpretation": "Le centre de gravité le plus élevé est préférable. Le détail du calcul affiche aussi les trois bornes du résultat agrégé.",
        "limits": "Triangles non négatifs ; coûts strictement positifs. La défuzzification résume l’incertitude en un nombre : des triangles différents peuvent partager le même score final.",
    },
}

WEIGHT_GUIDES = {
    "Equal": "Tous les critères reçoivent 1/n. Ce choix convient lorsque vous souhaitez leur accorder la même importance ; il ne tient pas compte des variations des données.",
    "Manual": "Saisissez les importances dans l’onglet Problème. L’application divise chaque poids par le total : 3 et 1 deviennent 75 % et 25 %. Un poids nul exclut le critère du classement.",
    "Entropy": "Après normalisation min–max orientée, l’entropie mesure la dispersion de chaque critère. Les critères qui distinguent davantage les alternatives reçoivent plus de poids. Un critère constant reçoit 0. Ces poids décrivent les données, pas vos préférences.",
    "CRITIC": "La méthode combine l’écart-type, qui mesure le contraste, et les corrélations, qui mesurent la redondance entre critères normalisés. Un critère variable et peu redondant reçoit plus de poids. Les critères constants reçoivent 0 ; des critères parfaitement corrélés peuvent être impossibles à départager.",
    "AHP": "Comparez l’importance des critères deux à deux. La matrice est réciproque : si A vaut 3 fois B, B vaut 1/3 de A. Le vecteur propre principal fournit les poids. Un ratio de cohérence CR supérieur à 0,10 invite à revoir les jugements ; ce n’est pas une mesure de performance des alternatives.",
    "BWM": "Désignez le critère le plus important et le moins important, puis comparez le meilleur aux autres et les autres au moins important sur une échelle de 1 à 9. Une optimisation linéaire minimise le plus grand écart aux comparaisons. Le résidu affiché mesure cet écart ; il ne s’agit pas du CR de l’AHP.",
    "Fuzzy manual": "Exprimez chaque importance par un triangle positif ou nul. Une mise à l’échelle commune normalise la somme des valeurs modales. Les triangles complets alimentent le classement flou ; les centres de gravité normalisés servent à afficher les importances.",
    "Fuzzy AHP": "Remplacez les comparaisons précises par des triangles. La méthode de Buckley calcule les moyennes géométriques des lignes, puis les normalise par une somme triangulaire inversée. Les poids restent flous pendant le classement ; le CR affiché vérifie seulement la matrice des valeurs modales.",
}

ERRORS = {
    "Fuzzy decision matrix must be alternatives × criteria × 3, with one direction per criterion.": "La matrice floue doit contenir trois bornes par alternative et critère, avec un sens pour chaque critère.",
    "Weights must be finite, nonnegative, and have a positive total.": "Les poids doivent être finis, positifs ou nuls, avec un total strictement positif.",
    "Set a benefit/cost direction for each criterion.": "Indiquez le sens maximiser ou minimiser pour chaque critère.",
    "Enter a nonempty matrix of finite numeric values; missing values are not filled automatically.": "Saisissez une matrice non vide de nombres finis. Les valeurs manquantes ne sont pas complétées automatiquement.",
    "Calculation exceeded numerical limits. Rescale the criterion units and try again.": "Le calcul dépasse les limites numériques. Changez l’échelle des unités et réessayez.",
    "VIKOR v and WASPAS lambda must be between 0 and 1.": "Les paramètres v de VIKOR et λ de WASPAS doivent être compris entre 0 et 1.",
    "Fuzzy values must contain three finite numbers (lower, modal, upper).": "Une valeur floue doit contenir trois nombres finis : borne inférieure, valeur modale et borne supérieure.",
    "Every triangle must satisfy lower ≤ modal ≤ upper.": "Chaque triangle doit respecter : borne inférieure ≤ valeur modale ≤ borne supérieure.",
    "Enter at least one alternative and one criterion.": "Saisissez au moins une alternative et un critère.",
    "Fuzzy weights must be nonnegative triangles with a positive modal total.": "Les poids flous doivent être des triangles non négatifs dont le total modal est strictement positif.",
    "Fuzzy ratio normalization requires nonnegative triangles.": "La normalisation floue par rapports exige des triangles non négatifs.",
    "Fuzzy cost criteria require strictly positive lower endpoints for reciprocal normalization.": "Les critères à minimiser exigent des bornes inférieures strictement positives pour la normalisation floue inverse.",
    "AHP needs a finite positive square comparison matrix.": "L’AHP exige une matrice de comparaison carrée, finie et strictement positive.",
    "AHP comparisons must be reciprocal and have a diagonal of 1.": "Les comparaisons AHP doivent être réciproques et la diagonale doit valoir 1.",
    "Fuzzy AHP needs a positive square matrix of triangles.": "L’AHP flou exige une matrice carrée de triangles strictement positifs.",
    "Fuzzy AHP comparisons must be reciprocal: (l,m,u) ↔ (1/u,1/m,1/l).": "Les comparaisons AHP floues doivent être réciproques : (l,m,u) ↔ (1/u,1/m,1/l).",
    "Fuzzy AHP self-comparisons must be (1,1,1).": "Les comparaisons d’un critère avec lui-même doivent valoir (1,1,1).",
    "BWM needs distinct best and worst criteria and matching comparisons.": "BWM exige deux critères extrêmes distincts et des comparaisons de dimensions cohérentes.",
    "BWM comparisons must be finite values between 1 and 9.": "Les comparaisons BWM doivent être des nombres finis compris entre 1 et 9.",
    "BWM self-comparisons must equal 1 and both best/worst comparisons must agree.": "Les comparaisons BWM d’un critère avec lui-même doivent valoir 1 ; les deux comparaisons meilleur/moins important doivent coïncider.",
    "Unsupported project format. Use a version 1 Decision Lab JSON file.": "Format non pris en charge. Utilisez un projet JSON Decision Lab de version 1.",
    "Alternative names must be nonempty and unique.": "Les noms des alternatives doivent être renseignés et uniques.",
    "Each criterion needs a nonempty name.": "Chaque critère doit avoir un nom.",
    "Criterion directions must be Benefit or Cost.": "Choisissez Maximiser ou Minimiser pour chaque critère.",
    "Criterion weights must be finite and nonnegative.": "Les poids des critères doivent être finis, positifs ou nuls.",
    "Criterion units must be text.": "Les unités doivent être du texte.",
    "Criterion names must be unique; 'Alternative' is reserved for row labels.": "Les noms des critères doivent être uniques ; « Alternative » est réservé aux noms des lignes.",
    "The decision matrix dimensions do not match the alternatives and criteria.": "Les dimensions de la matrice ne correspondent pas aux alternatives et aux critères.",
    "Mode must be Crisp or Fuzzy.": "Choisissez le modèle à valeurs précises ou le modèle flou.",
    "The linguistic scale must map nonempty terms to triangles.": "L’échelle linguistique doit associer des termes non vides à des triangles.",
    "Linguistic terms must be unique, ignoring case.": "Les termes linguistiques doivent être uniques, sans distinction de majuscules.",
    "Project settings must be an object.": "Les paramètres du projet doivent être un objet JSON.",
    "Selected methods must be a list of names.": "Les méthodes sélectionnées doivent former une liste de noms.",
    "BWM preference must be between 1 and 9.": "La préférence BWM doit être comprise entre 1 et 9.",
    "BWM comparisons must be numeric values from 1 to 9.": "Les comparaisons BWM doivent être numériques, de 1 à 9.",
    "Matrix cells must be numbers or text (for example, '2;3;5').": "Les cellules doivent contenir des nombres ou du texte, par exemple « 2;3;5 ».",
    "Invalid JSON project file.": "Le fichier de projet JSON n’est pas valide.",
    "Could not read the CSV. Check encoding and delimiter.": "Lecture du CSV impossible. Vérifiez l’encodage et le séparateur.",
    "CSV needs a label column, at least one criterion column and one alternative row.": "Le CSV doit contenir une colonne de noms, au moins un critère et une alternative.",
    "CSV column names must be unique.": "Les noms des colonnes du CSV doivent être uniques.",
}


def error_message(error):
    message = str(error)
    if message in ERRORS:
        return ERRORS[message]
    if "requires strictly positive values in all active criteria" in message:
        return f"{label(message.split(' requires')[0])} exige des valeurs strictement positives pour les critères de poids non nul. Choisissez une autre méthode ou revoyez l’échelle de mesure."
    if "cannot derive weights: all criteria are constant" in message:
        return "Impossible de calculer les poids : tous les critères sont constants. Choisissez des poids égaux ou manuels."
    if "found no distinguishing information" in message:
        return "Les données ne permettent pas de distinguer les critères, par exemple s’ils sont parfaitement corrélés. Choisissez des poids égaux ou manuels."
    match = re.match(r"Alternative (\d+), criterion (\d+): (.*)", message)
    if match:
        return f"Alternative {match[1]}, critère {match[2]} : {error_message(match[3])}"
    if message.startswith("Unknown value '"):
        value = message.split("'. Enter", 1)[0].removeprefix("Unknown value '")
        return f"Valeur inconnue « {value} ». Saisissez un nombre, un triangle comme 2;3;5 ou un terme de l’échelle linguistique."
    if "needs one numeric value or three triangle endpoints" in message:
        return "Saisissez un nombre ou trois bornes séparées par des points-virgules."
    if message.startswith("Project '"):
        return "Le projet doit contenir des listes non vides d’alternatives, de critères et de performances."
    if message.startswith(("Setting '", "Project ")):
        return "Un paramètre du projet a un type ou une valeur invalide. Vérifiez le fichier JSON."
    if message.startswith("could not convert") or "invalid literal" in message:
        return "Valeur numérique invalide. Utilisez un point décimal dans les tableaux, par exemple 0.5, ou 2;3;5 pour un triangle."
    if message.startswith("BWM optimization failed"):
        return "L’optimisation BWM a échoué. Vérifiez les comparaisons et réessayez."
    return message
