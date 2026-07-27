"""Thin client for the real, public SIGA queue-status API at
siga.marcacaodeatendimento.pt — verified 2026-07-26 by inspecting the site's
own "Consultar filas de espera" page and replaying its exact AJAX calls with
a bare `requests.post` (no browser session, no auth, no anti-forgery token
required).

Three endpoints, called in this order by the real page:
  1. GetEntidadesNoDistrito(IdDistrito) -> [{id, nome, idInstituicao, ...}]
     the organizations ("entidades") operating in that district.
  2. GetSenhas(IdDistrito, IdEntidade) -> [{id, nome}]
     the service/ticket types ("senhas") that entidade offers.
  3. GetLocais(IdDistrito, IdEntidade, IdSenha) -> [{nome, morada, latitude,
     longitude, servico: {estado, tempoRealEspera, tempoMedAtendimento,
     utentesEmEspera, ...}, ...}]
     the actual locations offering that (entidade, senha), each with live
     queue state for that service.

IMPORTANT: IdEntidade and IdSenha are both required — confirmed by testing
(omitting either causes a 302 redirect to an error page). There is no bulk
"give me everything in this district" shortcut; a full crawl is a genuine
3-level nest (district x entidade x senha), which is why
pipeline/siga_discovery.py exists as a separate one-time step rather than
something scrapers/siga_scraper.py redoes on every 15-minute poll.

The request body is the real page's entire form, serialized
(x-www-form-urlencoded) — most fields are unused boilerplate the ASP.NET MVC
model binder still expects present, confirmed by replaying the exact
production payload.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://siga.marcacaodeatendimento.pt"
REQUEST_TIMEOUT_SECONDS = 15

# Real district id -> name, read directly from the site's own <select> options.
# Mainland only (1-18); island regions (Madeira=31/32, Azores=41-49) exist on
# the real site too but aren't covered by this project's branch registry yet.
MAINLAND_DISTRITOS: dict[int, str] = {
    1: "Aveiro", 2: "Beja", 3: "Braga", 4: "Bragança", 5: "Castelo Branco",
    6: "Coimbra", 7: "Évora", 8: "Faro", 9: "Guarda", 10: "Leiria",
    11: "Lisboa", 12: "Portalegre", 13: "Porto", 14: "Santarém", 15: "Setúbal",
    16: "Viana do Castelo", 17: "Vila Real", 18: "Viseu",
}

# Boilerplate fields the real form always submits; only the ID fields
# actually vary per call. Verified against a captured production request.
_BOILERPLATE_FIELDS: dict[str, str] = {
    "HtmlAutenticacaoUtilizador": "False",
    "Latitude": "0",
    "Longitude": "0",
    "EstadoLocalizacao": "0",
    "ListDistritosAtivosSenhas": "System.Web.Mvc.SelectList",
    "IdLocalSelecionado": "",
    "IdInstituicaoSenha": "",
    "IdEntidadeSenha": "",
    "IdLocalSenha": "",
    "IdServicoSenha": "",
    "QRCodeImage": "",
    "DadosEncriptados": "",
    "PedidoNovoToken": "False",
    "EmailClienteNovoToken": "",
    "DescLocalAtendimento": "",
    "EmailCliente": "",
    "g-recaptcha-response": "",
    "CodigoConfirmacao": "",
}


class SigaClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def _post(self, endpoint: str, fields: dict[str, Any]) -> list[dict[str, Any]]:
        payload = {**_BOILERPLATE_FIELDS, **fields}
        response = self._session.post(
            f"{BASE_URL}/Senhas/{endpoint}",
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()

        # Undocumented: on no results, these endpoints return an error-shaped
        # object instead of an empty list, e.g.
        # {"idErro": 2, "servicosSimples": [], "mensagemErro": "Pesquisa não
        # devolveu resultados."} — found by testing, not documented anywhere.
        # Treat it as "no results" rather than let callers iterate a dict's
        # keys as if they were result rows.
        if isinstance(result, dict) and "idErro" in result:
            logger.debug("%s returned no results: %s", endpoint, result.get("mensagemErro"))
            return []
        return result

    def get_entidades(self, id_distrito: int) -> list[dict[str, Any]]:
        """Returns [{id, nome, idInstituicao, ...}] for organizations
        operating in this district."""
        return self._post("GetEntidadesNoDistrito", {"IdInstituicaoSelecionada": "0", "IdDistrito": id_distrito})

    def get_senhas(self, id_distrito: int, id_entidade: int, id_instituicao: int = 0) -> list[dict[str, Any]]:
        """Returns [{id, nome}] service/ticket types this entidade offers."""
        return self._post(
            "GetSenhas",
            {"IdInstituicaoSelecionada": id_instituicao, "IdDistrito": id_distrito, "IdEntidade": id_entidade},
        )

    def get_locais(
        self, id_distrito: int, id_entidade: int, id_senha: int, id_instituicao: int = 0
    ) -> list[dict[str, Any]]:
        """Returns real locations + live queue state for this (entidade, senha)."""
        return self._post(
            "GetLocais",
            {
                "IdInstituicaoSelecionada": id_instituicao,
                "IdDistrito": id_distrito,
                "IdEntidade": id_entidade,
                "IdSenha": id_senha,
            },
        )
