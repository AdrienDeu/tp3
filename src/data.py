# src/data.py
"""
TP3 — on passe à une VRAIE tâche : la NLU (analyse d'intention), façon
assistant vocal. Une phrase -> une STRUCTURE fermée :

    <intent> | <slot_type>=<valeur> | ...
    ex :  "mets une alarme à sept heures"  ->  alarm_set | time=sept heures

Pourquoi changer de tâche ? Parce que la porte de qualité ne devient vraiment
parlante que sur une tâche où un *system prompt* ne suffit PAS : ici, il
faudrait enseigner toute l'ontologie d'étiquettes au modèle — ça s'apprend par
l'exemple (fine-tuning), pas par une consigne. C'est ce que la porte va mesurer.

QUATRE ensembles disjoints :
  • TRAIN (40) : appris.
  • VALIDATION (10) : pour RÉGLER (nombre de pas, prompt...), autant qu'on veut.
  • TEST (12) : SCELLÉ — phrasings inédits, lu UNE fois pour JUGER (la porte).
  • RÉSERVE (12) : un second test, gardé sous clé pour le chapitre 7 du guide
    (démonstration de l'overfitting adaptatif). Ne pas le lire avant !
"""
from datasets import Dataset

ALLOWED_INTENTS = [
    "alarm_set", "alarm_remove", "alarm_query", "weather_query", "play_music",
    "calendar_set", "calendar_query", "news_query", "transport_query", "cooking_recipe",
]
SLOT_TYPES = [
    "time", "date", "place_name", "weather_descriptor", "artist_name",
    "music_genre", "song_name", "event_name", "news_topic", "transport_type", "food_type",
]


def build_target(intent: str, slots) -> str:
    if not slots:
        return intent
    return intent + " | " + " | ".join(f"{t}={v}" for t, v in slots)


# (phrase, intent, [(slot_type, valeur)])
_TRAIN = [
    ("mets une alarme à sept heures", "alarm_set", [("time", "sept heures")]),
    ("réveille-moi demain à six heures et demie", "alarm_set", [("date", "demain"), ("time", "six heures et demie")]),
    ("programme un réveil pour lundi matin", "alarm_set", [("date", "lundi"), ("time", "matin")]),
    ("ajoute une alarme à vingt heures ce soir", "alarm_set", [("time", "vingt heures"), ("date", "ce soir")]),
    ("supprime l'alarme de huit heures", "alarm_remove", [("time", "huit heures")]),
    ("annule mon réveil de demain", "alarm_remove", [("date", "demain")]),
    ("enlève toutes les alarmes", "alarm_remove", []),
    ("efface l'alarme de lundi matin", "alarm_remove", [("date", "lundi"), ("time", "matin")]),
    ("quelles sont mes alarmes", "alarm_query", []),
    ("à quelle heure est mon réveil demain", "alarm_query", [("date", "demain")]),
    ("montre-moi mes alarmes du week-end", "alarm_query", [("date", "week-end")]),
    ("est-ce que j'ai une alarme à sept heures", "alarm_query", [("time", "sept heures")]),
    ("quel temps fait-il à Paris", "weather_query", [("place_name", "paris")]),
    ("va-t-il pleuvoir demain", "weather_query", [("date", "demain"), ("weather_descriptor", "pluie")]),
    ("donne-moi la météo de la semaine à Lyon", "weather_query", [("place_name", "lyon"), ("date", "semaine")]),
    ("fait-il froid ce matin", "weather_query", [("weather_descriptor", "froid"), ("time", "matin")]),
    ("joue de la musique", "play_music", []),
    ("mets une chanson de Daft Punk", "play_music", [("artist_name", "daft punk")]),
    ("lance la playlist jazz", "play_music", [("music_genre", "jazz")]),
    ("joue Get Lucky", "play_music", [("song_name", "get lucky")]),
    ("ajoute une réunion lundi à dix heures", "calendar_set", [("date", "lundi"), ("time", "dix heures"), ("event_name", "réunion")]),
    ("planifie un rendez-vous chez le dentiste vendredi", "calendar_set", [("date", "vendredi"), ("event_name", "rendez-vous chez le dentiste")]),
    ("crée un événement anniversaire le douze mai", "calendar_set", [("date", "douze mai"), ("event_name", "anniversaire")]),
    ("note un appel avec Marie demain après-midi", "calendar_set", [("date", "demain"), ("time", "après-midi"), ("event_name", "appel avec marie")]),
    ("qu'est-ce que j'ai de prévu demain", "calendar_query", [("date", "demain")]),
    ("montre mon agenda de la semaine", "calendar_query", [("date", "semaine")]),
    ("ai-je une réunion lundi", "calendar_query", [("date", "lundi"), ("event_name", "réunion")]),
    ("quels sont mes rendez-vous aujourd'hui", "calendar_query", [("date", "aujourd'hui")]),
    ("donne-moi les actualités", "news_query", []),
    ("quelles sont les nouvelles sur le sport", "news_query", [("news_topic", "sport")]),
    ("lis-moi les infos économiques", "news_query", [("news_topic", "économie")]),
    ("actualités sur la politique en France", "news_query", [("news_topic", "politique"), ("place_name", "france")]),
    ("à quelle heure passe le prochain bus", "transport_query", [("transport_type", "bus")]),
    ("quand part le train pour Marseille", "transport_query", [("transport_type", "train"), ("place_name", "marseille")]),
    ("y a-t-il un métro vers la gare", "transport_query", [("transport_type", "métro"), ("place_name", "gare")]),
    ("horaires des trains demain", "transport_query", [("transport_type", "train"), ("date", "demain")]),
    ("donne-moi une recette de crêpes", "cooking_recipe", [("food_type", "crêpes")]),
    ("comment faire un gâteau au chocolat", "cooking_recipe", [("food_type", "gâteau au chocolat")]),
    ("recette de soupe à l'oignon", "cooking_recipe", [("food_type", "soupe à l'oignon")]),
    ("je veux cuisiner des lasagnes", "cooking_recipe", [("food_type", "lasagnes")]),
]
_VAL = [
    ("réveille-moi à neuf heures", "alarm_set", [("time", "neuf heures")]),
    ("supprime le réveil de demain matin", "alarm_remove", [("date", "demain"), ("time", "matin")]),
    ("quelles alarmes ai-je programmées", "alarm_query", []),
    ("quel temps à Bordeaux ce week-end", "weather_query", [("place_name", "bordeaux"), ("date", "week-end")]),
    ("mets du rock", "play_music", [("music_genre", "rock")]),
    ("ajoute un déjeuner mardi midi", "calendar_set", [("date", "mardi"), ("time", "midi"), ("event_name", "déjeuner")]),
    ("qu'ai-je prévu ce soir", "calendar_query", [("date", "ce soir")]),
    ("les actualités sur la technologie", "news_query", [("news_topic", "technologie")]),
    ("le prochain tram pour le centre", "transport_query", [("transport_type", "tram"), ("place_name", "centre")]),
    ("recette de tarte aux pommes", "cooking_recipe", [("food_type", "tarte aux pommes")]),
]
# TEST SCELLÉ : phrasings inédits, mêmes intentions (test de généralisation)
_TEST = [
    ("programme une alarme pour cinq heures du matin", "alarm_set", [("time", "cinq heures du matin")]),
    ("mets un réveil à minuit", "alarm_set", [("time", "minuit")]),
    ("annule toutes mes alarmes de la semaine", "alarm_remove", [("date", "semaine")]),
    ("est-ce que j'ai un réveil demain", "alarm_query", [("date", "demain")]),
    ("va-t-il neiger à Grenoble", "weather_query", [("place_name", "grenoble"), ("weather_descriptor", "neige")]),
    ("joue la chanson Imagine de John Lennon", "play_music", [("song_name", "imagine"), ("artist_name", "john lennon")]),
    ("ajoute un cours de yoga jeudi à dix-huit heures", "calendar_set", [("date", "jeudi"), ("time", "dix-huit heures"), ("event_name", "cours de yoga")]),
    ("montre-moi mes événements de vendredi", "calendar_query", [("date", "vendredi")]),
    ("donne les nouvelles internationales", "news_query", [("news_topic", "international")]),
    ("quand arrive le bus numéro douze", "transport_query", [("transport_type", "bus")]),
    ("comment préparer une ratatouille", "cooking_recipe", [("food_type", "ratatouille")]),
    ("fait-il beau à Nice aujourd'hui", "weather_query", [("place_name", "nice"), ("date", "aujourd'hui"), ("weather_descriptor", "beau")]),
]
# RÉSERVE : un second test, jamais consulté pendant le TP (chapitre 7 seulement)
_RESERVE = [
    ("règle un réveil à six heures quinze", "alarm_set", [("time", "six heures quinze")]),
    ("retire l'alarme de vendredi", "alarm_remove", [("date", "vendredi")]),
    ("combien d'alarmes ai-je ce soir", "alarm_query", [("date", "ce soir")]),
    ("quelle météo à Toulouse demain", "weather_query", [("place_name", "toulouse"), ("date", "demain")]),
    ("fera-t-il chaud cet après-midi", "weather_query", [("weather_descriptor", "chaud"), ("time", "après-midi")]),
    ("mets un morceau de Stromae", "play_music", [("artist_name", "stromae")]),
    ("lance de la musique classique", "play_music", [("music_genre", "classique")]),
    ("inscris un dîner samedi à vingt heures", "calendar_set", [("date", "samedi"), ("time", "vingt heures"), ("event_name", "dîner")]),
    ("qu'est-ce qui est prévu mardi", "calendar_query", [("date", "mardi")]),
    ("quoi de neuf en sciences", "news_query", [("news_topic", "sciences")]),
    ("à quelle heure part le prochain train pour Lille", "transport_query", [("transport_type", "train"), ("place_name", "lille")]),
    ("une recette de quiche lorraine", "cooking_recipe", [("food_type", "quiche lorraine")]),
]

_SPLITS = {"train": _TRAIN, "validation": _VAL, "test": _TEST, "reserve": _RESERVE}
SPLIT_NAMES = tuple(_SPLITS)


def _to_dataset(rows) -> Dataset:
    return Dataset.from_dict({
        "instruction": [u for (u, _i, _s) in rows],
        "response": [build_target(i, s) for (_u, i, s) in rows],
    })


def get_train_dataset() -> Dataset:
    return _to_dataset(_TRAIN)


def get_validation_dataset() -> Dataset:
    return _to_dataset(_VAL)


def get_test_dataset() -> Dataset:
    return _to_dataset(_TEST)


def get_reserve_dataset() -> Dataset:
    return _to_dataset(_RESERVE)


def get_split(name: str) -> Dataset:
    """Un jeu par son nom : train, validation, test ou reserve."""
    if name not in _SPLITS:
        raise ValueError(f"Jeu inconnu : {name} (attendu : {', '.join(SPLIT_NAMES)})")
    return _to_dataset(_SPLITS[name])


def examples_for_prompt(name: str, one_per_intent: bool = False):
    """Paires (phrase, cible) d'un jeu, pour les écrire dans un prompt."""
    pairs, seen = [], set()
    for u, i, s in _SPLITS[name]:
        if one_per_intent and i in seen:
            continue
        seen.add(i)
        pairs.append((u, build_target(i, s)))
    return pairs


def assert_disjoint() -> None:
    """Garde-fou : aucune phrase commune entre deux jeux."""
    sets = {n: {u for (u, *_r) in rows} for n, rows in _SPLITS.items()}
    names = list(sets)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            commun = sets[names[a]] & sets[names[b]]
            assert not commun, f"Fuite de données entre {names[a]} et {names[b]} : {commun}"


# ── Parsing + scoring de la sortie ─────────────────────────────────────
_ARTICLES = {"le", "la", "les", "un", "une", "des"}


def _norm_value(v: str) -> str:
    """minuscules + retrait d'un article en tête (la valeur ; pas le type)."""
    v = v.strip().lower()
    if v.startswith("l'"):
        v = v[2:].strip()
    toks = v.split()
    if toks and toks[0] in _ARTICLES:
        v = " ".join(toks[1:])
    return v.strip()


def parse_target(text: str):
    """Ligne -> (intent, set[(type,valeur)]) normalisés."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    parts = [p.strip() for p in line.split("|")]
    intent = parts[0].strip().lower() if parts else ""
    slots = set()
    for p in parts[1:]:
        if "=" in p:
            t, v = p.split("=", 1)
            slots.add((t.strip().lower(), _norm_value(v)))
    return intent, slots


def score(pred: str, gold: str):
    """(intent_ok, slot_f1, exact_match)."""
    pi, ps = parse_target(pred)
    gi, gs = parse_target(gold)
    intent_ok = (pi == gi)
    tp = len(ps & gs)
    prec = tp / len(ps) if ps else (1.0 if not gs else 0.0)
    rec = tp / len(gs) if gs else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else (1.0 if not gs and not ps else 0.0)
    exact = intent_ok and (ps == gs)
    return intent_ok, f1, exact


def format_chat(example, tokenizer) -> str:
    """Met un exemple au format conversation (renvoie une chaîne)."""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
