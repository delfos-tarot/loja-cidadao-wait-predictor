"""Unit tests for the real dados.gov.pt IALC-M ingestion/cleaning logic
(pipeline/load_ialc.py). No network calls — exercises parse_and_clean
directly against a small in-memory .xlsx fixture shaped like the real
dataset (Data/Loja/Distrito/Concelho/Total_Senhas/Total_Atendimentos/
Total_Desistencias/Tempo_Medio_Espera_Min/Tempo_Medio_Atendimento_Min).
"""

from __future__ import annotations

import pandas as pd

from pipeline.load_ialc import CANONICAL_COLUMNS, parse_and_clean


def _write_fixture_xlsx(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_excel(path, index=False)


def _row(**overrides) -> dict:
    base = {
        "Data": "2026-01-05",
        "Loja": "Loja de Cidadão das Laranjeiras",
        "Distrito": "Lisboa",
        "Concelho": "Lisboa",
        "Total_Senhas": 100,
        "Total_Atendimentos": 80,
        "Total_Desistencias": 20,
        "Tempo_Medio_Espera_Min": 25.0,
        "Tempo_Medio_Atendimento_Min": 6.0,
    }
    base.update(overrides)
    return base


def test_parse_and_clean_renames_columns_and_derives_branch_id(tmp_path) -> None:
    fixture_path = tmp_path / "indicadores-atendimento-lojas-cidadao-202601.xlsx"
    _write_fixture_xlsx(fixture_path, [_row()])

    cleaned = parse_and_clean([fixture_path])

    assert list(cleaned.columns) == CANONICAL_COLUMNS
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["branch_id"] == "loja_de_cidadao_das_laranjeiras"
    assert cleaned.iloc[0]["avg_wait_minutes"] == 25.0
    assert cleaned.iloc[0]["avg_service_minutes"] == 6.0


def test_parse_and_clean_skips_files_missing_required_columns(tmp_path) -> None:
    fixture_path = tmp_path / "malformed.xlsx"
    _write_fixture_xlsx(fixture_path, [{"Loja": "Loja de Cidadão de Braga", "Total_Atendimentos": 5}])

    cleaned = parse_and_clean([fixture_path])

    assert cleaned.empty
    assert list(cleaned.columns) == CANONICAL_COLUMNS


def test_parse_and_clean_drops_rows_with_null_wait_or_duration(tmp_path) -> None:
    # Real data (2026-07-27 audit) has a handful of rows with a null wait or
    # duration -- these must be dropped, not silently coerced to 0/NaN and
    # fed downstream as if they were real measurements.
    fixture_path = tmp_path / "indicadores-atendimento-lojas-cidadao-202602.xlsx"
    _write_fixture_xlsx(
        fixture_path,
        [
            _row(),
            _row(Loja="Loja de Cidadão do Porto", Tempo_Medio_Espera_Min=None),
            _row(Loja="Loja de Cidadão de Braga", Tempo_Medio_Atendimento_Min=None),
        ],
    )

    cleaned = parse_and_clean([fixture_path])

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["branch_id"] == "loja_de_cidadao_das_laranjeiras"


def test_parse_and_clean_drops_duplicate_branch_day_rows(tmp_path) -> None:
    fixture_path = tmp_path / "indicadores-atendimento-lojas-cidadao-202603.xlsx"
    row = _row()
    _write_fixture_xlsx(fixture_path, [row, row])  # exact duplicate row

    cleaned = parse_and_clean([fixture_path])

    assert len(cleaned) == 1
