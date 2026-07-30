"""Unit tests for pipeline/reconcile_siga_services.py."""

from __future__ import annotations

from pipeline.reconcile_siga_services import extract_siga_service_names, match_services, name_similarity, normalize_name


def test_normalize_name_strips_accents_and_case() -> None:
    assert normalize_name("Atendimento Geral") == normalize_name("ATENDIMENTO GERAL")
    assert normalize_name("Certidões") == "certidoes"


def test_name_similarity_scores_substring_containment_above_plain_ratio() -> None:
    # SequenceMatcher.ratio() length-normalizes over both strings combined,
    # so it undervalues clean containment -- the dominant real pattern here
    # (SIGA prefixes a department onto the canonical name).
    assert name_similarity("Câmara Municipal - Atendimento Geral", "Atendimento Geral") >= 0.7


def test_extract_siga_service_names_dedupes_and_skips_missing() -> None:
    locations = [
        {"servico": {"nome": "Geral"}},
        {"servico": {"nome": "Geral"}},  # duplicate, same real desk at another location
        {"servico": {"nome": "Tesouraria"}},
        {"servico": {}},  # no nome -- must not produce a blank entry
        {},  # no servico at all
    ]

    names = extract_siga_service_names(locations)

    assert names == ["Geral", "Tesouraria"]


def test_match_services_skips_names_already_in_the_vocabulary() -> None:
    vocabulary = ("Atendimento Geral", "Tesouraria")

    crosswalk = match_services(["Atendimento Geral"], vocabulary=vocabulary)

    assert crosswalk == []  # exact match already -- no crosswalk entry needed


def test_match_services_reconciles_a_qualified_siga_name_to_its_canonical_base() -> None:
    # The dominant real pattern: SIGA prefixes a department/entity onto the
    # canonical name. Safe because the canonical name is contained verbatim.
    vocabulary = ("Atendimento Geral", "Tesouraria")

    crosswalk = match_services(["Câmara Municipal - Atendimento Geral"], vocabulary=vocabulary)

    assert len(crosswalk) == 1
    assert crosswalk[0]["siga_service_name"] == "Câmara Municipal - Atendimento Geral"
    assert crosswalk[0]["canonical_service_name"] == "Atendimento Geral"
    assert 0.0 < crosswalk[0]["match_confidence"] <= 1.0


def test_match_services_matches_on_casing_whitespace_and_accents_only() -> None:
    vocabulary = ("Atendimento Geral",)

    crosswalk = match_services(["ATENDIMENTO GERAL ", "Atendimento geral"], vocabulary=vocabulary)

    assert {e["siga_service_name"] for e in crosswalk} == {"ATENDIMENTO GERAL ", "Atendimento geral"}
    assert all(e["canonical_service_name"] == "Atendimento Geral" for e in crosswalk)


def test_match_services_leaves_unrelated_names_unmatched() -> None:
    vocabulary = ("Atendimento Geral", "Tesouraria")

    crosswalk = match_services(["Galp - Eletricidade e Gás Natural"], vocabulary=vocabulary)

    assert crosswalk == []


def test_match_services_rejects_a_similar_looking_but_different_service() -> None:
    # Found 2026-07-30 in the first real run: 'Atendimento email' scored
    # 0.909 against 'Atendimento EMEL' -- an email desk vs. Lisbon's parking
    # authority. Higher than several correct matches, so no threshold fixes
    # it; only the structural containment rule does. Silently merging two
    # distinct real services is worse than leaving one unreconciled.
    vocabulary = ("Atendimento EMEL",)

    crosswalk = match_services(["Atendimento email"], vocabulary=vocabulary)

    assert crosswalk == []


def test_match_services_rejects_narrowing_a_generic_name_to_a_specific_one() -> None:
    # Same run: 'Licenciamento' -> 'Licenciamento de Festas' would narrow
    # generic licensing to party licensing specifically. Rejected because
    # containment must run canonical-inside-SIGA (SIGA is the more qualified
    # name), never the reverse.
    vocabulary = ("Licenciamento de Festas",)

    crosswalk = match_services(["Licenciamento"], vocabulary=vocabulary)

    assert crosswalk == []
