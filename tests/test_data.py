# tests/test_data.py  —  lancer avec :  make test
import pytest

from src.data import (
    get_train_dataset, get_validation_dataset, get_test_dataset, get_reserve_dataset,
    get_split, assert_disjoint, score, examples_for_prompt,
)


def test_sizes_and_disjoint():
    assert len(get_train_dataset()) == 40
    assert len(get_validation_dataset()) == 10
    assert len(get_test_dataset()) == 12
    assert len(get_reserve_dataset()) == 12
    assert_disjoint()  # ne lève pas : aucune phrase partagée entre deux jeux


def test_get_split():
    assert len(get_split("reserve")) == 12
    with pytest.raises(ValueError):
        get_split("inconnu")


def test_scoring_exact_and_intent():
    # tout juste
    assert score("alarm_set | time=sept heures", "alarm_set | time=sept heures") == (True, 1.0, True)
    # bonne intention, slot de mauvais TYPE -> intention ok mais pas exact
    io, f1, ex = score("weather_query | weather_condition=neige",
                       "weather_query | weather_descriptor=neige")
    assert io is True and ex is False
    # normalisation d'article sur la valeur
    assert score("alarm_remove | date=la semaine", "alarm_remove | date=semaine")[2] is True


def test_prompt_examples():
    # un exemple par intention depuis le train : 10 intentions
    pairs = examples_for_prompt("train", one_per_intent=True)
    assert len(pairs) == 10
    assert pairs[0] == ("mets une alarme à sept heures", "alarm_set | time=sept heures")
    # le prompt « triche » contient bien les 12 phrases du test
    assert len(examples_for_prompt("test")) == 12
