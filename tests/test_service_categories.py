"""Unit tests for the service-category mapping (pipeline/service_categories.py)
used by pipeline/calibrate_constants.py — see that module's docstring for why
only 3 broad categories are used, not one per exact service label."""

from __future__ import annotations

from pipeline.service_categories import CATEGORIES, GENERAL_OTHER, IRN, OTHER_SPECIALIZED, categorize


def test_categories_are_exactly_three() -> None:
    assert set(CATEGORIES) == {IRN, GENERAL_OTHER, OTHER_SPECIALIZED}


def test_irn_services_categorized_correctly() -> None:
    assert categorize("Cartão de Cidadão - Pedido/Renovação") == IRN
    assert categorize("Passaporte - Levantamento / Inf.") == IRN
    assert categorize("Registo Civil - Certidões / Inf.") == IRN


def test_specialized_services_categorized_correctly() -> None:
    assert categorize("Execuções Fiscais") == OTHER_SPECIALIZED  # AT
    assert categorize("Carta de Condução/ Driving License") == OTHER_SPECIALIZED  # IMT
    assert categorize("ID cidadão estrangeiro/ Foreign Citizen ID") == OTHER_SPECIALIZED  # SEF
    assert categorize("Galp - Eletricidade e Gás Natural") == OTHER_SPECIALIZED  # private partner


def test_general_and_unknown_services_fall_back_to_general_other() -> None:
    assert categorize("Atendimento Geral") == GENERAL_OTHER
    assert categorize("Triagem") == GENERAL_OTHER
    assert categorize("Some Brand New Service Label Never Seen Before") == GENERAL_OTHER


def test_categorize_is_case_insensitive() -> None:
    assert categorize("CARTÃO DE CIDADÃO - OUTROS") == IRN
