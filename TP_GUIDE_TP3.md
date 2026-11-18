# TP3 guidé — Régler, juger… et ne pas se mentir

**Public :** débutant. **Pré-requis :** TP2 fait (MLflow + RustFS), Docker et uv installés.
**Idée :** au TP2, tu as comparé des essais sur la **validation**. Mais la
validation a servi à **choisir** tes réglages : elle ne peut donc plus servir à
**juger**. Le TP3 ajoute :

- un **test scellé**, qu'on ne lit **qu'une fois** ;
- une **porte de qualité** qui rend un verdict **PASS / FAIL** ;
- la question que tout le monde pose : *« un simple prompt système ne suffirait-il
  pas ? »*.

Il se termine par le piège le plus courant de l'évaluation : l'**overfitting
adaptatif**.

> Convention : un cadre `comme ceci` = commande à taper dans un terminal,
> **depuis le dossier `tp3`** (celui qui contient `main.py` et `.env`).

---

## 0. Ce qui change par rapport au TP2

| TP2 | TP3 |
|---|---|
| une tâche de **format** (trois lignes) | une vraie tâche de **NLU** (intention + slots) |
| 2 jeux : train, validation | **4 jeux** : train, validation, **test**, **réserve** |
| un programme (`main.py`) | **régler** (`main.py`) ≠ **juger** (`juger.py`) |
| — | un **labo de prompt** (`prompt_lab.py`) : le modèle de base, guidé par une consigne |

La plateforme est **la même qu'au TP2** (MLflow + RustFS) : seuls les noms des
conteneurs changent (`tp3-…`).

---

## 1. La tâche : la NLU

NLU = *Natural Language Understanding*. Pour un assistant vocal, il s'agit de
transformer une phrase en une **structure fermée** :

- une **intention**, choisie dans une liste de 10 : ce que veut l'utilisateur ;
- des **slots**, d'un des 11 types autorisés : les détails utiles.

```
"mets une alarme à sept heures"  ->  alarm_set | time=sept heures
"va-t-il pleuvoir demain"        ->  weather_query | date=demain | weather_descriptor=pluie
```

Le modèle doit répondre **exactement** dans ce format, avec **ces** étiquettes.
Écrire `weather_condition=neige` au lieu de `weather_descriptor=neige`, c'est
faux. Cette liste fermée d'étiquettes s'appelle une **ontologie**. On verra
qu'elle change tout dans la comparaison avec un prompt.

Le petit corpus du TP est **écrit à la main** sur le modèle du jeu de référence
MASSIVE (Amazon, partie fr-FR) : mêmes familles d'intentions, mêmes types de
slots, mais quelques dizaines de phrases au lieu d'un million.

---

## 2. Les quatre jeux (et ce qu'on a le droit d'en faire)

Ouvre **`src/data.py`** : les phrases y sont rangées en quatre listes,
**sans aucune phrase commune** (une fonction, `assert_disjoint`, le vérifie à
chaque lancement).

| Jeu | Taille | Rôle | On a le droit de… |
|---|---|---|---|
| `train` | 40 | **apprendre** | s'en servir pour entraîner, et comme exemples dans un prompt |
| `validation` | 10 | **régler** | le regarder **autant qu'on veut** pour choisir un réglage |
| `test` | 12 | **juger** | le lire **une seule fois**, à la fin, pour le verdict |
| `reserve` | 12 | **démontrer** | ne pas l'ouvrir : il ne sert qu'au chapitre 7 |

> **La règle d'or :** la validation sert à **choisir**, le test sert à **juger**.
> Dès qu'un chiffre du test influence un choix, il ne juge plus rien.

Le projet applique cette règle **dans le code** :

- **`main.py` ne lit jamais le test**. Il entraîne, puis mesure sur la validation.
- **`juger.py` est le seul programme qui lit le test**. Chaque lecture est
  **comptée** et le compteur s'affiche dans MLflow.

---

## 3. Démarrer

1. Si la plateforme du TP2 tourne encore, arrête-la : les deux TP utilisent les
   mêmes ports. Dans le dossier `tp2` :

   ```
   make clean
   ```

2. Dans le dossier `tp3`, démarre la plateforme et installe les bibliothèques :

   ```
   make setup
   uv sync
   ```

   👀 **À regarder :** comme au TP2, 3 conteneurs : `tp3-rustfs` (healthy),
   `tp3-rustfs-setup` (Exited (0), c'est normal), `tp3-mlflow` (healthy).

3. Lance les tests unitaires (données et calcul du score), en quelques secondes :

   ```
   make test
   ```

   👀 **À regarder :** `4 passed`. Ils vérifient notamment que les quatre jeux
   n'ont **aucune** phrase en commun.

---

## 4. Régler le LoRA, sur la validation

### 4.1 Un premier essai

Le `.env` est réglé sur `TRAIN_MAX_STEPS=150` : la NLU demande beaucoup plus
d'entraînement que le format du TP2. Pour commencer, mets **`TRAIN_MAX_STEPS=30`**
dans `.env`, enregistre, puis :

```
uv run python main.py
```

👀 **À regarder, dans le terminal :**

1. `MLflow run: … (lora 30 pas)` : cette fois, le run porte un **nom lisible**.
2. `Train: 40 | Validation: 10  (le test reste scellé)`.
3. L'entraînement habituel (`loss`, `eval_loss`, meilleure perte de validation).
4. **`Mesure sur la VALIDATION`** : c'est la nouveauté. On ne mesure plus
   seulement une perte, mais **la tâche elle-même** :
   - `intent_acc` : la part des phrases dont l'**intention** est juste ;
   - `slot_f1` : la qualité des **slots** (0 % = aucun bon, 100 % = tous bons) ;
   - `exact` : la part des réponses **entièrement** justes.
5. La liste des **erreurs**, avec pour chacune la réponse attendue et la réponse
   obtenue. Tu as le droit de les lire : c'est la validation.

### 4.2 Faire varier le nombre de pas

Relance avec **`TRAIN_MAX_STEPS=60`**, puis **`100`**, puis **`150`**. À 150 pas,
compte environ 6 à 7 minutes sur un ordinateur portable.

Dans MLflow (http://localhost:5000, mode **Model training**, expérience
**`tp3-nlu-cycle`**), ajoute les colonnes avec **Columns** :

- dans **Metrics**, `val_intent_acc`, `val_exact` et `best_val_loss` ;
- dans **Parameters**, `train_max_steps`.

Remplis ce tableau :

| pas | val_intent_acc | val_exact | best_val_loss |
|---|---|---|---|
| 30 | | | |
| 60 | | | |
| 100 | | | |
| 150 | | | |

👀 **Ce qu'on observe en général :** avec trop peu de pas, le modèle est
**sous-entraîné**. Il répond mal ou à côté du format, et `val_intent_acc` est
bas. Puis les scores montent et finissent par se stabiliser.

### 4.3 Lire les erreurs (un exemple réel)

Voici la fin d'un vrai run à **150 pas**, avec Qwen3-0.6B, en 6 minutes environ
sur un ordinateur portable :

```
=== Mesure sur la VALIDATION (sert à régler) ===
  intent_acc   100%
     slot_f1    73%
       exact    50%
```

**Les intentions sont toutes justes : les erreurs viennent des slots.** La liste
des erreurs en montre trois familles :

| Famille | Exemple (attendu → obtenu) | Pourquoi |
|---|---|---|
| **slot inventé** | `music_genre=rock` → `song=rock` ; `event_name=déjeuner` → `event=déjeuner` | `music_genre` n'apparaît qu'**une fois** dans le train : le modèle invente un nom plausible au lieu de celui de l'ontologie |
| **slot oublié** | `date=demain \| time=matin` → `date=demain` | les phrases avec une date **et** une heure sont rares dans le train (6 sur 40) |
| **valeur modifiée** | `place_name=centre` → `place_name=center` | le modèle « traduit » la valeur en anglais au lieu de la recopier |

**Que faire ?** Tout cela se règle **sur la validation**, c'est permis :

- **plus de pas** : la perte de validation baissait encore à la fin
  (`eval_loss` 1,213 → 1,181 → 1,151). Essaie 200 pas ;
- **plus d'exemples des slots rares** dans le train. C'est un changement de
  données : il faut ensuite relancer `main.py`.

> Remarque du TP2 : la ligne de fin `train_loss: 1.703` est la **moyenne** de
> tout l'entraînement, alors que les dernières pertes sont autour de 0,7. C'est
> pour cela que MLflow enregistre `final_train_loss` et `mean_train_loss`
> séparément.

> **À discuter :** la porte du chapitre 6 ne regarde que `intent_acc`. Avec
> `exact` à 50 %, un assistant vocal comprendrait l'intention mais se tromperait
> souvent de détail. Faudrait-il ajouter un seuil sur `slot_f1` ou sur `exact` ?
> Si oui, **décide-le maintenant**, avant de lire le test : par exemple
> `EVAL_GATE_MIN_SLOT_F1=0.6` dans `.env` (0 = aucune exigence sur les slots).
> Fixer un seuil **après** avoir vu le résultat du test, c'est encore de
> l'overfitting adaptatif.

### 4.4 Choisir

**Choisis ton meilleur réglage en regardant uniquement la validation.** Pas
besoin de noter quoi que ce soit : `juger.py` saura retrouver ce run tout seul.

---

## 5. Donner sa meilleure chance au prompt système

La question de fond : *faut-il vraiment fine-tuner ?* Peut-être qu'une bonne
**consigne** (un **prompt système**) donnée au modèle de **base**, sans aucun
entraînement, ferait aussi bien.

Pour que la comparaison soit honnête, on donne au prompt **sa meilleure
chance**, et on la règle **aussi sur la validation**. C'est le rôle du **labo
de prompt** : il ne fait aucun entraînement, il ne dure donc qu'une minute ou
deux.

1. Lis le prompt de départ, **`prompts/systeme.txt`**. Il donne la liste des
   intentions, celle des types de slots, le format et un exemple.

2. Mesure-le sur la validation :

   ```
   make prompt-systeme
   ```

3. Fabrique une version **enrichie d'exemples** : un exemple par intention,
   **pris dans le train**. Puis mesure-la sur la validation :

   ```
   make prompt-exemples
   ```

   Ouvre le fichier produit, **`prompts/exemples_train.txt`**, pour voir ce qui
   a été ajouté.

   > Prendre des exemples dans le **train**, c'est **légitime** : c'est fait
   > pour apprendre.

4. Compare les deux dans MLflow, expérience **`tp3-prompt-lab`**. Les runs
   s'appellent `systeme @ validation` et `exemples_train @ validation`. Ajoute
   la colonne `intent_acc` avec **Columns**.

5. Tu peux aussi écrire **ton propre** prompt : copie un fichier du dossier
   `prompts/`, modifie-le, puis mesure-le sur la validation, par exemple :

   ```
   uv run python prompt_lab.py prompts/mon_prompt.txt validation
   ```

6. Garde **le meilleur prompt selon la validation**, et indique-le dans `.env`.
   Par exemple :

   ```
   PROMPT_SYSTEME_FILE=prompts/exemples_train.txt
   ```

C'est ce prompt qui défendra la colonne **base+prompt** au moment du jugement.

---

## 6. Juger : la porte de qualité, sur le test scellé

Tous les choix sont faits : le nombre de pas pour le LoRA, la consigne pour le
prompt. **Maintenant seulement**, on ouvre le test :

```
make juger
```

👀 **À regarder, dans le terminal, dans l'ordre :**

1. **`Runs candidats, classés sur la VALIDATION`** : `juger.py` prend le run qui
   a la meilleure `val_intent_acc`. Le test n'intervient **pas** dans ce choix.
2. Le modèle **n'est pas réentraîné** : son adaptateur LoRA est **récupéré dans
   MLflow** (donc dans RustFS, comme au TP2).
3. `🔒 Première lecture du TEST SCELLÉ.`
4. Le tableau des **trois colonnes**. Exemple de résultat (tes valeurs seront
   différentes) :

```
=== Résultats sur le TEST SCELLÉ (NLU) ===
                      LoRA          base   base+prompt
  intent_acc          100%            0%           42%
     slot_f1           60%            0%           12%
       exact           42%            0%            8%

   avance LoRA vs base+prompt : +58 points
   GATE : PASS ✅  (seuil intent >= 70%, marge >= 30%)
```

### 6.1 Lire les trois colonnes

- **base** : le modèle nu, sans aide. Il ne connaît ni la tâche ni le format.
- **base+prompt** : le modèle nu, avec la **meilleure consigne** choisie au
  chapitre 5. Il comprend l'idée, mais se trompe souvent d'étiquette ou de format.
- **LoRA** : notre modèle fine-tuné, choisi au chapitre 4.

### 6.2 Le critère PASS / FAIL

La porte exige **deux** conditions :

1. `intent_acc(LoRA)` **≥ seuil**, 70 % par défaut : le modèle est **compétent** ;
2. `intent_acc(LoRA) − intent_acc(base+prompt)` **≥ marge**, 30 points par défaut :
   le fine-tuning apporte un gain **net** par rapport au prompt.

> **Pourquoi la 2ᵉ condition ?** Pour répondre à l'objection « un prompt
> suffirait ». Si le LoRA ne battait pas nettement le meilleur prompt, fine-tuner
> **ne vaudrait pas la peine** : c'est plus long, plus cher, et plus lourd à
> maintenir. La porte encode cette exigence. Un FAIL pour cette raison n'est pas
> un échec du TP : c'est une **bonne décision**.

### 6.3 Pourquoi la NLU donne l'avantage au fine-tuning

Compare avec le TP2 :

- le **format en trois lignes** se décrit **en une phrase** (« réponds en trois
  lignes : Définition, Intuition, À retenir »). Un prompt le fait presque aussi
  bien qu'un fine-tuning ;
- la **NLU** exige de connaître une **ontologie** : 10 intentions, 11 types de
  slots, et surtout **quand** utiliser chacun. Cela ne se résume pas en une
  consigne courte. Le prompt **suggère**, le fine-tuning **grave** cette
  correspondance dans les poids.

> **Règle à retenir :** si le comportement voulu **se décrit en mots** (un
> format, un ton), un prompt suffit souvent. S'il faut **apprendre une
> correspondance** qu'on ne sait pas décrire simplement (une ontologie, une
> grammaire de sortie), le fine-tuning devient nécessaire.

### 6.4 Dans MLflow

Le jugement est **ajouté au run jugé** (expérience `tp3-nlu-cycle`) :

- des métriques `test_lora_intent_acc`, `test_base_prompt_intent_acc`,
  `test_margin_over_prompt` et `gate_passed` (1 = PASS, 0 = FAIL) ;
- un tag `gate_passed` ;
- un fichier `predictions_test.txt` dans l'onglet **Artifacts**.

### 6.5 Rendre la porte bloquante

Par défaut, un FAIL **n'arrête pas** le programme, c'est plus pratique en cours.
En production, on veut qu'un mauvais modèle soit **refusé** :

```
make juger-strict
```

Si le critère n'est pas atteint, le programme **s'arrête en erreur** (code de
sortie 1). C'est ce comportement qu'une chaîne automatique (CI/CD) utilisera
plus tard : un modèle qui rate la porte n'est pas livré. Pour durcir ou
assouplir la porte : `EVAL_GATE_MIN_INTENT=0.9` ou `EVAL_GATE_MARGIN=0.5` dans
`.env`.

> Attention : `make juger-strict` **relit le test**. Regarde le compteur au
> chapitre suivant.

---

## 7. Le piège : l'overfitting adaptatif

### 7.1 L'idée

Au TP1, tu as vu le **surapprentissage** : à force de voir les mêmes exemples,
le **modèle** les apprend par cœur. Il existe une version plus sournoise, où
c'est **toi** qui apprends le jeu par cœur, sans t'en rendre compte.

Chaque fois que tu regardes un score du test **et que ce score influence une
décision**, une petite information sur le test passe dans tes choix. Exemples
de décisions : changer le prompt, garder ce nombre de pas plutôt qu'un autre…
Après dix ou vingt allers-retours, tes réglages sont **ajustés à ces 12 phrases
précises**. Le score du test monte, mais il ne mesure plus ce que le modèle
vaut sur des phrases **nouvelles**. C'est l'**overfitting adaptatif** :
**adaptatif**, parce que chaque essai s'adapte au résultat du précédent.

### 7.2 L'expérience : tricher exprès, puis se faire prendre

On va le provoquer volontairement, sur le prompt, parce que c'est rapide.

1. **Tricher.** Fabrique un prompt qui contient… les phrases du test et leurs
   réponses. C'est exactement ce qu'on finit par faire, sans s'en rendre compte,
   quand on corrige un prompt en regardant les erreurs du test.

   ```
   uv run python prompt_lab.py --construire triche-test
   uv run python prompt_lab.py prompts/triche_test.txt test
   ```

   👀 **À regarder :** le score de ce prompt sur le test. Compare-le à celui de
   ton meilleur prompt honnête et à celui du LoRA, au chapitre 6.

2. **Se faire prendre.** Ouvre maintenant la **réserve** : 12 phrases nouvelles,
   jamais vues par personne. Mesure le prompt triché et ton meilleur prompt
   honnête dessus :

   ```
   uv run python prompt_lab.py prompts/triche_test.txt reserve
   uv run python prompt_lab.py prompts/exemples_train.txt reserve
   ```

3. **Comparer.** Dans MLflow, expérience **`tp3-prompt-lab`**, ajoute les
   colonnes `intent_acc`, `exact` et le paramètre `jeu`. Remplis :

   | prompt | test | réserve |
   |---|---|---|
   | `triche_test` | | |
   | `exemples_train` (ou ton meilleur) | — | |

👀 **Ce qu'on s'attend à voir :**

- sur le **test**, le prompt triché a l'air excellent, peut-être même meilleur
  que le LoRA ;
- sur la **réserve**, son avance **fond** : il ne fait pas mieux qu'un prompt
  honnête.

Son « talent » n'était qu'une connaissance des 12 phrases du test, pas une
compétence. Sur la réserve, il retombe à son vrai niveau.

### 7.3 Le même piège, sans tricher exprès

Personne ne copie le test dans son prompt. Mais le même mécanisme agit, à plus
petite dose, dès qu'on **choisit** en regardant le test. Imagine qu'au chapitre 4
tu aies choisi le nombre de pas en lisant le **test** à chaque essai :

- 12 phrases seulement : **une** phrase de plus ou de moins, c'est **8 points** ;
- en essayant beaucoup de réglages et en gardant le meilleur score, tu
  sélectionnes en partie la **chance**, pas le meilleur modèle ;
- le chiffre annoncé est alors **trop optimiste**, et rien ne le signale.

C'est pour cela que, dans ce TP, `main.py` ne lit jamais le test et que
`juger.py` choisit le run **sur la validation**.

### 7.4 Le compteur de lectures

Dans MLflow, va sur la page **Experiments** (menu de gauche). La colonne
**Tags** de `tp3-nlu-cycle` affiche **`lectures_test: N`**, et
`lectures_reserve: N` (clique sur `+1` pour le voir).

- Chaque `make juger` et chaque mesure du labo sur `test` ajoute **1** à
  `lectures_test`.
- Le terminal te prévient dès la 2ᵉ lecture : `⚠️ TEST SCELLÉ lu N fois`.

Le compteur ne t'interdit rien : il **rend visible** combien de fois le test a
été consulté. Une lecture qui ne change aucune décision ne coûte rien. Une
lecture suivie d'un changement de réglage **use** le test.

### 7.5 Que faire en vrai ?

- **Tout décider sur la validation** (réglages, prompt, choix du modèle), et lire
  le test **une fois**, à la fin.
- Si le test a servi à décider, il est **brûlé** : son score n'est plus fiable.
  Il faut un **nouveau** test. C'est le rôle qu'a joué la réserve, et elle est
  maintenant brûlée à son tour.
- **Tracer chaque lecture** du test, comme le fait le compteur : on sait alors
  combien de fois on l'a consulté, et on peut le dire.
- Plus le test est **petit**, plus il s'use vite : 12 phrases, c'est très peu.
  En vrai, on vise des centaines ou des milliers d'exemples.

> **La phrase à retenir :** *la validation sert à choisir, le test scellé sert à
> juger. Chaque fois qu'on regarde le test pour décider, on le transforme un peu
> plus en validation, et son verdict perd sa valeur.*

---

## 8. Ce qui manque encore (→ TP4)

Tu as maintenant un cycle complet : **entraîner → régler → juger → tracer**.
Mais il tient en **plusieurs scripts** lancés à la main, dans le bon ordre. Si
une étape plante, tu ne vois pas clairement laquelle. En vrai, on veut un
**pipeline** : des étapes nommées, qui s'enchaînent, qu'on peut relancer et
surveiller une par une.

C'est le **TP4** : **Prefect** transforme ces scripts en pipeline orchestré,
avec une interface qui montre chaque étape, verte ou rouge.

---

## 9. Arrêter la plateforme

- `make clean` : arrête les services. **Tes runs sont conservés**.
- `docker compose down -v` : arrête **et efface tout**, runs, fichiers et
  compteurs compris.

---

## 10. Dépannage

- **Page « Something went wrong » dans MLflow.** Retourne sur
  http://localhost:5000 et clique sur **`Model training`** avant de cliquer sur
  l'expérience, comme au TP2.
- **`make setup` : port 5000, 9000 ou 9001 déjà utilisé.** La plateforme du TP2
  tourne sans doute encore : `make clean` dans le dossier `tp2`. Tu peux le
  vérifier avec `docker ps`.
- **`intent_acc` LoRA très bas, réponses vides ou absurdes.** C'est un
  sous-entraînement : remonte `TRAIN_MAX_STEPS`, jusqu'à 150.
- **`juger.py` : « Aucun run mesuré sur la validation ».** Lance d'abord au moins
  un `uv run python main.py`.
- **`GATE: FAIL` alors que le LoRA semble bon.** Vérifie la **marge** : le LoRA
  doit battre base+prompt d'au moins 30 points. Si un prompt honnête s'en
  approche, le verdict est juste : fine-tuner n'en vaut pas la peine. Si tu
  modifies `EVAL_GATE_MARGIN`, fais-le **avant** de lire le test, jamais après
  avoir vu le résultat. Sinon, c'est encore de l'overfitting adaptatif.
- **`Fichier de prompt introuvable`.** Le chemin est relatif au dossier `tp3` :
  `prompts/…`. `prompts/exemples_train.txt` n'existe qu'après
  `make prompt-exemples`.
- **Erreur d'upload d'artefact (S3, `NoCredentialsError`).** Vérifie que
  `tp3-rustfs` tourne (`make health`), et lance les commandes **depuis le dossier
  `tp3`**, sinon le `.env` n'est pas lu.
- **Je veux recommencer à zéro (compteurs compris).** Lance
  `docker compose down -v`, puis `make setup`.
