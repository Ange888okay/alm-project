# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 11:23:58 2026

@author: GAMER
"""

import re
import math
import numpy as np
import pandas as pd


SHEETS = {
    "bilan": "Bilan au 31_12_2025 ",
    "loi": "Loi d'écoulement ",
    "courbe": "Courbe des taux",
    "stress_liq": "Stress test de liquidité",
    "stress_taux": "stress test de taux",
}


# =========================
# Utilitaires
# =========================
def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_upper(x) -> str:
    return norm_text(x).upper()


def to_float(x):
    return pd.to_numeric(x, errors="coerce")


def safe_sheet_read(path: str, sheet: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Impossible de lire l'onglet '{sheet}'. Détail: {e}")


# =========================
# 1) Lecture + parsing du bilan
# =========================
def read_and_parse_balance(path: str) -> pd.DataFrame:
    df = safe_sheet_read(path, SHEETS["bilan"], header=0)

    df = df.rename(
        columns={
            "Date d’arrêté": "Date",
            "Poste du bilan": "Poste",
            "Montant (en k€)": "Montant_kEUR",
            "Taux d'intérèt moyen": "Taux_moyen",
        }
    )

    df["Bilan"] = df["Bilan"].map(norm_upper)
    df["Poste"] = df["Poste"].map(norm_text)
    df["Montant_kEUR"] = to_float(df["Montant_kEUR"])
    df["Taux_moyen"] = to_float(df["Taux_moyen"])

    df = df[~df["Montant_kEUR"].isna()].copy()

    poste_low = df["Poste"].str.lower().str.strip()
    df = df[~poste_low.isin(["total actif", "total actif ", "total passif", "total passif "])].copy()
    df = df[~df["Poste"].str.strip().str.startswith("*")].copy()

    # reconstruire catégorie depuis l’ordre du bilan
    df = df.reset_index(drop=True)
    cat_from_balance = []
    is_detail = []
    current_cat = ""

    for poste in df["Poste"]:
        p = poste.strip()
        if p.startswith("-"):
            cat_from_balance.append(current_cat)
            is_detail.append(True)
        else:
            current_cat = p
            cat_from_balance.append(p)
            is_detail.append(False)

    df["Cat_from_balance"] = [norm_text(x) for x in cat_from_balance]
    df["Is_detail_dash"] = is_detail

    # supprimer les "parents" qui ont des enfants "-"
    keep = np.ones(len(df), dtype=bool)
    i = 0
    while i < len(df):
        if not df.loc[i, "Is_detail_dash"]:
            j = i + 1
            has_child_dash = False
            while j < len(df) and df.loc[j, "Is_detail_dash"]:
                has_child_dash = True
                j += 1
            if has_child_dash:
                keep[i] = False
            i = j
        else:
            i += 1

    df = df[keep].copy().reset_index(drop=True)

    df.loc[~df["Is_detail_dash"], "Cat_from_balance"] = ""

    # éviter double comptage fonds propres
    if (df["Poste"].str.strip().str.lower() == "capitaux propres").any():
        sub_fp = {
            "frbg",
            "capital souscrit",
            "prime d'émission",
            "réserves",
            "report à nouveau",
            "résultat de l'exercice",
            "dividende",
        }
        if df["Poste"].str.strip().str.lower().isin(sub_fp).any():
            df = df[df["Poste"].str.strip().str.lower() != "capitaux propres"].copy()

    df["BilanN"] = df["Bilan"].map(norm_upper)
    df["PosteN"] = df["Poste"].map(norm_text)
    df["CatN"] = df["Cat_from_balance"].map(norm_upper)

    df["Side"] = np.where(df["BilanN"] == "ACTIF", 1.0, -1.0)

    return df.reset_index(drop=True)


# =========================
# 2) Lecture des règles d'écoulement
# =========================
def read_runoff_rules(path: str) -> pd.DataFrame:
    r = safe_sheet_read(path, SHEETS["loi"], header=1).copy()

    r = r.rename(
        columns={
            "Catégories Bilan": "Categorie",
            "Poste du bilan": "Poste",
            "Loi d'écoulement": "Loi_Liq",
            "Durée moyenne (en mois)": "Duree_Liq",
            "Loi d'écoulement.1": "Loi_Taux",
            "Durée moyenne (en mois).1": "Duree_Taux",
        }
    )

    r["BilanN"] = r["Bilan"].map(norm_upper)
    r["CatN"] = r["Categorie"].map(norm_upper)
    r["PosteN"] = r["Poste"].map(norm_text)

    r["Loi_Liq"] = r["Loi_Liq"].map(norm_text)
    r["Loi_Taux"] = r["Loi_Taux"].map(norm_text)
    r["Duree_Liq"] = to_float(r["Duree_Liq"])
    r["Duree_Taux"] = to_float(r["Duree_Taux"])

    return r[["BilanN", "CatN", "PosteN", "Loi_Liq", "Duree_Liq", "Loi_Taux", "Duree_Taux"]].copy()


def attach_rules(positions: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    p = positions.copy()

    merged = p.merge(
        rules,
        on=["BilanN", "PosteN", "CatN"],
        how="left",
        suffixes=("", "_r"),
    )

    counts = rules.groupby(["BilanN", "PosteN"]).size()
    unique_keys = set([k for k, v in counts.items() if v == 1])

    def fill_from_unique(row):
        if pd.isna(row["Loi_Liq"]):
            k = (row["BilanN"], row["PosteN"])
            if k in unique_keys:
                rr = rules[(rules["BilanN"] == k[0]) & (rules["PosteN"] == k[1])].iloc[0]
                row["CatN"] = rr["CatN"]
                row["Loi_Liq"] = rr["Loi_Liq"]
                row["Duree_Liq"] = rr["Duree_Liq"]
                row["Loi_Taux"] = rr["Loi_Taux"]
                row["Duree_Taux"] = rr["Duree_Taux"]
        return row

    merged = merged.apply(fill_from_unique, axis=1)

    merged["CatN"] = merged["CatN"].replace("", np.nan).fillna("INCONNU")
    merged["Loi_Liq"] = merged["Loi_Liq"].fillna("Linéaire")
    merged["Duree_Liq"] = merged["Duree_Liq"].fillna(12.0)
    merged["Loi_Taux"] = merged["Loi_Taux"].fillna(merged["Loi_Liq"])
    merged["Duree_Taux"] = merged["Duree_Taux"].fillna(merged["Duree_Liq"])

    # colonnes utiles pour l’UI
    merged["Categorie"] = merged["CatN"]
    merged["Poste"] = merged["PosteN"]
    merged["Bilan"] = merged["BilanN"]

    return merged.reset_index(drop=True)


# =========================
# 3) Lois d'écoulement -> poids mensuels
# =========================
def weights_law(law: str, duration_months: int) -> np.ndarray:
    s = norm_text(law).lower()
    s = s.replace("é", "e").replace("è", "e").replace("ê", "e")
    s = s.replace("ine fine", "in fine").replace("ine  fine", "in fine")

    d = int(max(1, round(duration_months)))

    if "lineaire" in s:
        w = np.ones(d)
        return w / w.sum()

    if "constant" in s:
        w = np.ones(d)
        return w / w.sum()

    if "in fine" in s:
        w = np.zeros(d)
        w[-1] = 1.0
        return w

    m = re.search(r"90%\s*-\s*20%\s*\*\s*t.*t\s*<=\s*(\d+)", s)
    if m:
        tmax = min(int(m.group(1)), d)
        t = np.arange(1, tmax + 1, dtype=float)
        w = 0.90 - 0.20 * t
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            w = np.zeros(tmax)
            w[-1] = 1.0
        w = w / w.sum()
        if tmax < d:
            w = np.concatenate([w, np.zeros(d - tmax)])
            w[-1] += (1.0 - w.sum())
        return w

    m = re.search(r"90%\s*\*\s*exp\s*\(\s*-\s*20%\s*\*\s*t\s*\).*t\s*<=\s*(\d+)", s)
    if m:
        tmax = min(int(m.group(1)), d)
        t = np.arange(1, tmax + 1, dtype=float)
        w = 0.90 * np.exp(-0.20 * t)
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            w = np.zeros(tmax)
            w[-1] = 1.0
        w = w / w.sum()
        if tmax < d:
            w = np.concatenate([w, np.zeros(d - tmax)])
            w[-1] += (1.0 - w.sum())
        return w

    w = np.ones(d)
    return w / w.sum()


# =========================
# 4) Cashflows + GAP liquidité
# =========================
def compute_cashflows(positions: pd.DataFrame, max_months: int = 240) -> pd.DataFrame:
    month_cols = [f"M{m}" for m in range(1, max_months + 1)]
    rows = []

    for _, r in positions.iterrows():
        amount = float(r["Montant_kEUR"])
        d = int(max(1, round(float(r["Duree_Liq"]))))
        d = min(d, max_months)

        w = weights_law(r["Loi_Liq"], d)
        cf = np.zeros(max_months)
        cf[:d] = amount * w

        row = {
            "Bilan": r["BilanN"],
            "Categorie": r["CatN"],
            "Poste": r["PosteN"],
            "Montant_kEUR": amount,
            "Loi_Liq": r["Loi_Liq"],
            "Duree_Liq": float(r["Duree_Liq"]),
        }
        row.update({month_cols[i]: cf[i] for i in range(max_months)})
        rows.append(row)

    return pd.DataFrame(rows)


def liquidity_gap(cashflows: pd.DataFrame, max_months: int = 240):
    month_cols = [f"M{m}" for m in range(1, max_months + 1)]
    side = np.where(cashflows["Bilan"].str.upper() == "ACTIF", 1.0, -1.0)

    gap_month = (cashflows[month_cols].mul(side, axis=0)).sum(axis=0).reset_index()
    gap_month.columns = ["Mois", "Gap_kEUR"]

    bucket = []
    for m in range(1, 13):
        bucket.append((f"M{m}", float(gap_month.loc[gap_month["Mois"] == f"M{m}", "Gap_kEUR"].iloc[0])))
    bucket.append(
        ("M>12", float(gap_month.loc[~gap_month["Mois"].isin([f"M{i}" for i in range(1, 13)]), "Gap_kEUR"].sum()))
    )

    gap_bucket = pd.DataFrame(bucket, columns=["Bucket", "Gap_kEUR"])
    return gap_month, gap_bucket


def cumulative_gap(gap_month: pd.DataFrame) -> pd.DataFrame:
    g = gap_month.copy()
    g["Gap_cumule_kEUR"] = g["Gap_kEUR"].cumsum()
    return g


# =========================
# 5) Stress test de liquidité (depuis l'onglet)
# =========================
def read_stress_liquidity(path: str) -> dict:
    """
    Retour:
    {
      "Scenario 1: ...": {("ACTIF","CAISSE, ..."): -0.10, ("PASSIF","CPTES ..."): -0.02, ...},
      ...
    }
    Les % sont lus tels quels (ex: -15% -> -0.15)
    """
    df = safe_sheet_read(path, SHEETS["stress_liq"], header=None)
    scenarios = {}
    current = None
    current_side = None

    for i in range(len(df)):
        c0 = norm_text(df.iloc[i, 0])
        c1 = df.iloc[i, 1]

        if c0.lower().startswith("scenario"):
            current = c0
            scenarios[current] = {}
            current_side = None
            continue

        if current is None:
            continue

        if c0.lower() in ["actifs", "passifs"]:
            current_side = "ACTIF" if c0.lower() == "actifs" else "PASSIF"
            continue

        if not c0 or pd.isna(c1) or current_side is None:
            continue

        shock = to_float(c1)
        if pd.isna(shock):
            continue

        scenarios[current][(current_side, norm_upper(c0))] = float(shock)

    return scenarios


def apply_liquidity_stress_to_positions(positions: pd.DataFrame, scenarios: dict) -> pd.DataFrame:
    """
    Applique les chocs sur Montant_kEUR par (Side, Categorie).
    """
    out = []
    pos = positions.copy()
    pos["CategorieN"] = pos["CatN"].map(norm_upper)

    for scen, rules in scenarios.items():
        p = pos.copy()
        p["Shock"] = 0.0

        for (side, cat), shock in rules.items():
            mask = (p["BilanN"] == side) & (p["CategorieN"] == cat)
            p.loc[mask, "Shock"] = shock

        p["Montant_kEUR_Stress"] = p["Montant_kEUR"] * (1.0 + p["Shock"])
        p["Scenario"] = scen
        out.append(p)

    return pd.concat(out, ignore_index=True)


def compute_liquidity_stress_results(positions: pd.DataFrame, scenarios: dict, max_months: int = 240):
    """
    Pour chaque scénario :
      - positions stressées
      - cashflows stressés
      - gap stressé (mensuel + bucket + cumul)
    """
    stressed_positions = apply_liquidity_stress_to_positions(positions, scenarios)

    all_gap_month = []
    all_gap_bucket = []
    all_gap_cum = []

    for scen in stressed_positions["Scenario"].unique():
        p = stressed_positions[stressed_positions["Scenario"] == scen].copy()
        p2 = p.copy()
        p2["Montant_kEUR"] = p2["Montant_kEUR_Stress"]

        cf = compute_cashflows(p2, max_months=max_months)
        gm, gb = liquidity_gap(cf, max_months=max_months)
        gc = cumulative_gap(gm)

        gm["Scenario"] = scen
        gb["Scenario"] = scen
        gc["Scenario"] = scen

        all_gap_month.append(gm)
        all_gap_bucket.append(gb)
        all_gap_cum.append(gc)

    return stressed_positions, pd.concat(all_gap_month), pd.concat(all_gap_bucket), pd.concat(all_gap_cum)


# =========================
# 6) Courbe des taux + Stress taux + EVE (PV cashflows)
# =========================
def read_rate_curve(path: str) -> pd.DataFrame:
    """
    Lecture simple de l’onglet 'Courbe des taux' basé sur ta capture (maturité, zero coupon).
    """
    df = safe_sheet_read(path, SHEETS["courbe"], header=None)
    # repère les colonnes A/B généralement
    df = df.iloc[1:].copy()
    df.columns = ["Maturite", "Zero", "X"]
    df["Maturite"] = df["Maturite"].map(norm_text)
    df["Zero"] = to_float(df["Zero"])

    def maturity_to_months(m):
        m = norm_text(m).lower()
        if "mois" in m:
            n = re.findall(r"\d+", m)
            return float(n[0]) if n else np.nan
        if "an" in m:
            n = re.findall(r"\d+", m)
            return float(n[0]) * 12.0 if n else np.nan
        return np.nan

    df["Mois"] = df["Maturite"].map(maturity_to_months)
    df = df[~df["Mois"].isna()].copy().sort_values("Mois")
    return df[["Mois", "Zero"]].reset_index(drop=True)


def interp_rate(curve: pd.DataFrame, months: float) -> float:
    x = curve["Mois"].values.astype(float)
    y = curve["Zero"].values.astype(float)
    if months <= x.min():
        return float(y[0])
    if months >= x.max():
        return float(y[-1])
    return float(np.interp(months, x, y))


def read_rate_stress_table(path: str) -> pd.DataFrame:
    """
    Table 'stress test de taux':
    colonnes: Type de scenarios | Catégories | scenarios | 1..n (mois)
    valeurs = chocs (en décimal) par maturité mensuelle
    """
    df = safe_sheet_read(path, SHEETS["stress_taux"], header=0).copy()
    df = df.rename(columns={"Type de scenarios": "Type", "Catégories": "Categorie", "scenarios": "Scenario"})
    df["Type"] = df["Type"].map(norm_text)
    df["Categorie"] = df["Categorie"].map(norm_text)
    df["Scenario"] = df["Scenario"].map(norm_text)

    month_cols = [c for c in df.columns if isinstance(c, (int, float))]
    if len(month_cols) == 0:
        raise RuntimeError("Aucune colonne de mois détectée dans l'onglet 'stress test de taux'.")

    return df[["Type", "Categorie", "Scenario"] + month_cols].copy()


def pv_from_cashflows(cf_row: pd.Series, curve: pd.DataFrame, max_months: int) -> float:
    """
    PV = somme CF_t / (1+r_t)^(t/12)
    r_t interpolé depuis la courbe
    """
    pv = 0.0
    for m in range(1, max_months + 1):
        c = float(cf_row.get(f"M{m}", 0.0))
        if c == 0:
            continue
        r = interp_rate(curve, float(m))
        pv += c / ((1.0 + r) ** (m / 12.0))
    return pv


def build_discounted_eve(positions: pd.DataFrame, curve: pd.DataFrame, max_months: int = 240):
    """
    EVE base = PV(Actifs) - PV(Passifs) (signe via Side)
    On valorise via cashflows de liquidité (approx pédagogique).
    """
    cf = compute_cashflows(positions, max_months=max_months)
    month_cols = [f"M{m}" for m in range(1, max_months + 1)]

    # PV par ligne
    pv_list = []
    for _, row in cf.iterrows():
        pv = pv_from_cashflows(row, curve, max_months=max_months)
        side = 1.0 if row["Bilan"].upper() == "ACTIF" else -1.0
        pv_list.append(side * pv)

    eve = float(np.sum(pv_list))
    return eve, cf


def apply_rate_shock_to_curve(curve: pd.DataFrame, shock_by_month: np.ndarray) -> pd.DataFrame:
    """
    Construit une courbe choquée:
    r_shocked(m) = r_base(m) + shock[m-1] (si m dans range)
    """
    max_m = len(shock_by_month)
    out = curve.copy()
    out["Zero_shocked"] = out["Zero"]
    # on stocke juste la base; l’interpolation utilisera un wrapper
    return out


def interp_rate_shocked(curve_base: pd.DataFrame, months: float, shock_by_month: np.ndarray) -> float:
    r = interp_rate(curve_base, months)
    m_int = int(round(months))
    if 1 <= m_int <= len(shock_by_month):
        return float(r + float(shock_by_month[m_int - 1]))
    return float(r + float(shock_by_month[-1]))


def pv_from_cashflows_shocked(cf_row: pd.Series, curve: pd.DataFrame, shock_by_month: np.ndarray, max_months: int) -> float:
    pv = 0.0
    for m in range(1, max_months + 1):
        c = float(cf_row.get(f"M{m}", 0.0))
        if c == 0:
            continue
        r = interp_rate_shocked(curve, float(m), shock_by_month)
        pv += c / ((1.0 + r) ** (m / 12.0))
    return pv


def compute_eve_under_rate_stress(positions: pd.DataFrame, curve: pd.DataFrame, stress_table: pd.DataFrame, max_months: int = 240) -> pd.DataFrame:
    """
    Calcule ΔEVE pour chaque scénario de taux (PV choqué - PV base)
    """
    base_eve, cf = build_discounted_eve(positions, curve, max_months=max_months)

    month_cols = [c for c in stress_table.columns if isinstance(c, (int, float))]
    month_cols_sorted = sorted(month_cols)

    results = []
    for _, srow in stress_table.iterrows():
        shock = np.array([float(srow[m]) for m in month_cols_sorted], dtype=float)

        pv_list = []
        for _, row in cf.iterrows():
            pv = pv_from_cashflows_shocked(row, curve, shock, max_months=max_months)
            side = 1.0 if row["Bilan"].upper() == "ACTIF" else -1.0
            pv_list.append(side * pv)

        eve_shocked = float(np.sum(pv_list))
        results.append(
            {
                "Type": srow["Type"],
                "Categorie": srow["Categorie"],
                "Scenario": srow["Scenario"],
                "EVE_base_kEUR": base_eve,
                "EVE_stress_kEUR": eve_shocked,
                "Delta_EVE_kEUR": eve_shocked - base_eve,
            }
        )

    return pd.DataFrame(results).sort_values("Delta_EVE_kEUR")


# =========================
# 7) Export complet Excel
# =========================
def write_results_full(
    path_out: str,
    positions: pd.DataFrame,
    cashflows: pd.DataFrame,
    gap_month: pd.DataFrame,
    gap_bucket: pd.DataFrame,
    gap_cum: pd.DataFrame,
    stress_liq_positions: pd.DataFrame = None,
    stress_liq_gap_month: pd.DataFrame = None,
    stress_liq_gap_bucket: pd.DataFrame = None,
    stress_liq_gap_cum: pd.DataFrame = None,
    eve_stress: pd.DataFrame = None,
):
    with pd.ExcelWriter(path_out, engine="openpyxl") as w:
        positions.to_excel(w, sheet_name="Positions", index=False)
        cashflows.to_excel(w, sheet_name="Cashflows_Mensuels", index=False)
        gap_month.to_excel(w, sheet_name="Gap_Mensuel", index=False)
        gap_bucket.to_excel(w, sheet_name="Gap_Buckets", index=False)
        gap_cum.to_excel(w, sheet_name="Gap_Cumule", index=False)

        if stress_liq_positions is not None:
            stress_liq_positions.to_excel(w, sheet_name="StressLiq_Positions", index=False)
        if stress_liq_gap_month is not None:
            stress_liq_gap_month.to_excel(w, sheet_name="StressLiq_Gap_Mensuel", index=False)
        if stress_liq_gap_bucket is not None:
            stress_liq_gap_bucket.to_excel(w, sheet_name="StressLiq_Gap_Buckets", index=False)
        if stress_liq_gap_cum is not None:
            stress_liq_gap_cum.to_excel(w, sheet_name="StressLiq_Gap_Cumule", index=False)

        if eve_stress is not None:
            eve_stress.to_excel(w, sheet_name="StressTaux_EVE", index=False)


# =========================
# RUN complet (sans UI)
# =========================
def run_all(input_file: str, output_file: str = "ALM_Resultats_COMPLET.xlsx", max_months: int = 240):
    pos = read_and_parse_balance(input_file)
    rules = read_runoff_rules(input_file)
    pos = attach_rules(pos, rules)

    cashflows = compute_cashflows(pos, max_months=max_months)
    gap_month, gap_bucket = liquidity_gap(cashflows, max_months=max_months)
    gap_cum = cumulative_gap(gap_month)

    # stress liquidité
    stress_liq = read_stress_liquidity(input_file)
    st_pos, st_gm, st_gb, st_gc = compute_liquidity_stress_results(pos, stress_liq, max_months=max_months)

    # stress taux (EVE)
    curve = read_rate_curve(input_file)
    stress_taux = read_rate_stress_table(input_file)
    eve_stress = compute_eve_under_rate_stress(pos, curve, stress_taux, max_months=max_months)

    write_results_full(
        output_file,
        positions=pos,
        cashflows=cashflows,
        gap_month=gap_month,
        gap_bucket=gap_bucket,
        gap_cum=gap_cum,
        stress_liq_positions=st_pos,
        stress_liq_gap_month=st_gm,
        stress_liq_gap_bucket=st_gb,
        stress_liq_gap_cum=st_gc,
        eve_stress=eve_stress,
    )

    tot_act = pos.loc[pos["BilanN"] == "ACTIF", "Montant_kEUR"].sum()
    tot_pas = pos.loc[pos["BilanN"] == "PASSIF", "Montant_kEUR"].sum()

    return {
        "positions": pos,
        "cashflows": cashflows,
        "gap_month": gap_month,
        "gap_bucket": gap_bucket,
        "gap_cum": gap_cum,
        "stress_liq_positions": st_pos,
        "stress_liq_gap_month": st_gm,
        "stress_liq_gap_bucket": st_gb,
        "stress_liq_gap_cum": st_gc,
        "eve_stress": eve_stress,
        "tot_act": float(tot_act),
        "tot_pas": float(tot_pas),
        "output_file": output_file,
    }


if __name__ == "__main__":
    # Spyder : tu peux lancer directement
    res = run_all("Projet_ALM_M2_IFIM.xlsx", "ALM_Resultats_COMPLET.xlsx", max_months=240)
    print("OK - Export :", res["output_file"])
    print("Total ACTIF  :", res["tot_act"])
    print("Total PASSIF :", res["tot_pas"])
