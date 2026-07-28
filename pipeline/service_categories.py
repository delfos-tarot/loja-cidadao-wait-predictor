"""Maps the real dataset's ~221 distinct `service_type` labels down to a small
number of broad categories, for use by pipeline/calibrate_constants.py.

WHY ONLY 3 CATEGORIES (not 7, not all ~221 individual labels):
An identifiability check (2026-07-27) against the real SLC-M attendance data
found that finer splits aren't supported by the actual service-mix variation
across branches:
  - GENERAL_OTHER and IRN (the two dominant categories, 64.4% and 22.0% of all
    real attendance volume) are strongly negatively correlated across
    branches (-0.82) — expected since they're most of each branch's mix and
    shares sum to 1, but it leaves only ~33% independent variance to separate
    them, so even this split is imprecise, not exact.
  - AT and ISS (2 of the smaller categories) are moderately positively
    correlated (+0.45) — branches with more tax-authority business tend to
    also have more social-security business, making them hard to separate
    from each other specifically.
  - IMT, SEF, and PRIVATE_PARTNER (Galp) are tiny and concentrated at a
    handful of branches (coefficient of variation 0.9-5.3), so any
    category-specific parameter fit for them would rest on very few
    informative branches.
Rather than fit ~221 (or even 7) independent parameters against data that
can't separably identify them — which would just fit noise while looking
precise — everything except IRN and GENERAL_OTHER is folded into a single
OTHER_SPECIALIZED bucket. See the calibration script's chronological holdout
for whether even this 3-way split holds up out of sample.

CATEGORY MEANINGS (heuristic, keyword-based on the label text itself — there
is no official government service-to-entity crosswalk published alongside
this dataset):
  - IRN: registries/notary/passport/civil-registry desks
  - GENERAL_OTHER: the general/triage/default desk — not tied to one entity,
    and the single largest category by volume (this is "the general Loja do
    Cidadão visit" most citizens experience, especially in Lisbon: 40% of
    Loja de Cidadão das Laranjeiras' total volume, vs. 8.3% for Cartão de
    Cidadão specifically)
  - OTHER_SPECIALIZED: everything else (AT/tax, ISS/social security,
    IMT/driving-vehicle, SEF/immigration, private-partner desks like Galp),
    folded together because none of them is separably identifiable alone
"""

from __future__ import annotations

IRN = "IRN"
GENERAL_OTHER = "GENERAL_OTHER"
OTHER_SPECIALIZED = "OTHER_SPECIALIZED"

CATEGORIES = (IRN, GENERAL_OTHER, OTHER_SPECIALIZED)

_IRN_KEYWORDS = (
    "cartão de cidadão",
    "registo civil",
    "registo predial",
    "passaporte",
    "nacionalidade",
    "certidão",
)
_SPECIALIZED_KEYWORDS = (
    # AT (tax authority)
    "património",
    "execuções fiscais",
    "finanças",
    "irs",
    "iva",
    # IMT (transport authority)
    "registo veículos",
    "carta de condução",
    "condutores",
    "dísticos",
    # ISS (social security)
    "segurança social",
    "rendimento",
    "pensõ",
    "abono",
    # SEF/AIMA (immigration)
    "cidadão estrangeiro",
    "foreign citizen",
    # Private partner desks co-located at some branches
    "galp",
)


def categorize(service_type: str) -> str:
    """Maps one real `service_type` label to a broad category. See module
    docstring for why only 3 categories are used."""
    s = service_type.lower()
    if any(keyword in s for keyword in _IRN_KEYWORDS):
        return IRN
    if any(keyword in s for keyword in _SPECIALIZED_KEYWORDS):
        return OTHER_SPECIALIZED
    return GENERAL_OTHER
