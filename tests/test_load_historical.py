"""Unit tests for the real dados.gov.pt ingestion/cleaning logic. No network
calls — exercises parse_and_clean directly against a small in-memory .xlsx
fixture shaped like the real dataset (Data/Distrito/Concelho/Loja/Servico/
Atendimentos), verified against the live API in pipeline/load_historical.py's
module docstring.
"""

from __future__ import annotations

import pandas as pd

from pipeline.load_historical import RESOURCE_MONTH_PATTERN, parse_and_clean


def _write_fixture_xlsx(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_excel(path, index=False)


def test_resource_month_pattern_extracts_year_and_month() -> None:
    match = RESOURCE_MONTH_PATTERN.search("servicos-lojas-cidadao-202607.xlsx")
    assert match is not None
    assert match.group(1) == "2026"
    assert match.group(2) == "07"


def test_parse_and_clean_renames_and_normalizes_columns(tmp_path) -> None:
    fixture_path = tmp_path / "servicos-lojas-cidadao-202601.xlsx"
    _write_fixture_xlsx(
        fixture_path,
        [
            {
                "Data": "2026-01-05",
                "Distrito": " Lisboa ",
                "Concelho": " Lisboa ",
                "Loja": " Loja de Cidadão das Laranjeiras ",
                "Servico": " Atendimento Geral ",
                "Atendimentos": 42,
            },
            {
                "Data": "2026-01-06",
                "Distrito": "Porto",
                "Servico": "Passaporte - Pedido",
                "Concelho": "Porto",
                "Loja": "Loja de Cidadão do Porto",
                "Atendimentos": "17",  # string numeric, should be coerced
            },
        ],
    )

    cleaned = parse_and_clean([fixture_path])

    assert list(cleaned.columns) == ["date", "district", "municipality", "store_name", "service_type", "total_attendances"]
    assert len(cleaned) == 2
    # Whitespace must be stripped from text fields.
    assert cleaned.iloc[0]["store_name"] == "Loja de Cidadão das Laranjeiras"
    assert cleaned.iloc[0]["service_type"] == "Atendimento Geral"
    # String-typed attendance counts must be coerced to int.
    assert cleaned.iloc[1]["total_attendances"] == 17
    assert cleaned["total_attendances"].dtype.kind in "iu"


def test_parse_and_clean_skips_files_missing_required_columns(tmp_path) -> None:
    fixture_path = tmp_path / "malformed.xlsx"
    _write_fixture_xlsx(fixture_path, [{"Distrito": "Lisboa", "Atendimentos": 5}])  # missing Data/Loja/Servico/Concelho

    cleaned = parse_and_clean([fixture_path])

    assert cleaned.empty
    assert list(cleaned.columns) == ["date", "district", "municipality", "store_name", "service_type", "total_attendances"]


def test_parse_and_clean_drops_duplicate_rows(tmp_path) -> None:
    fixture_path = tmp_path / "servicos-lojas-cidadao-202602.xlsx"
    row = {
        "Data": "2026-02-01",
        "Distrito": "Braga",
        "Concelho": "Braga",
        "Loja": "Loja de Cidadão de Braga",
        "Servico": "Tesouraria",
        "Atendimentos": 9,
    }
    _write_fixture_xlsx(fixture_path, [row, row])  # exact duplicate row

    cleaned = parse_and_clean([fixture_path])

    assert len(cleaned) == 1
