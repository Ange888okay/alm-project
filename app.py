# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 19:07:13 2026

@author: GAMER
"""
import os
import tempfile
import pandas as pd
import streamlit as st

from alm_engine import run_all


st.set_page_config(page_title="Outil ALM", layout="wide")
st.title("📊 Outil ALM – Projet")

st.markdown(
    """
Cette application permet :
- d’**importer** le fichier Excel du projet,
- de lancer le **traitement** (écoulements, gaps, stress liquidité, stress taux/EVE),
- de **visualiser** les résultats,
- d’**exporter** un Excel de sortie.
"""
)

uploaded = st.file_uploader("📁 Importer le fichier Excel (Projet_ALM_M2_IFIM.xlsx)", type=["xlsx"])

colA, colB, colC = st.columns(3)
max_months = colA.number_input("Horizon (mois)", min_value=12, max_value=360, value=240, step=12)
run_button = colB.button("🚀 Lancer l’analyse")
show_details = colC.checkbox("Afficher tables détaillées", value=False)

if uploaded and run_button:
    with st.spinner("Calculs en cours..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_in:
            tmp_in.write(uploaded.read())
            input_path = tmp_in.name

        output_path = os.path.join(os.getcwd(), "ALM_Resultats_COMPLET.xlsx")
        res = run_all(input_path, output_file=output_path, max_months=int(max_months))

    st.success("Analyse terminée ✅")

    # ===== KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Actif (kEUR)", f"{res['tot_act']:,.2f}".replace(",", " "))
    k2.metric("Total Passif (kEUR)", f"{res['tot_pas']:,.2f}".replace(",", " "))
    min_gap = float(res["gap_month"]["Gap_kEUR"].min())
    k3.metric("Min Gap mensuel (kEUR)", f"{min_gap:,.2f}".replace(",", " "))
    eve_worst = float(res["eve_stress"]["Delta_EVE_kEUR"].min())
    k4.metric("Worst ΔEVE (kEUR)", f"{eve_worst:,.2f}".replace(",", " "))

    # ===== Graphs
    st.subheader("📉 Gap de liquidité (mensuel)")
    gm = res["gap_month"].copy()
    gm["M"] = gm["Mois"].str.replace("M", "").astype(int)
    gm = gm.sort_values("M")
    st.line_chart(gm.set_index("M")["Gap_kEUR"])

    st.subheader("📈 Gap cumulé")
    gc = res["gap_cum"].copy()
    gc["M"] = gc["Mois"].str.replace("M", "").astype(int)
    gc = gc.sort_values("M")
    st.line_chart(gc.set_index("M")["Gap_cumule_kEUR"])

    st.subheader("🧨 Stress liquidité — Gap cumulé par scénario")
    st_gc = res["stress_liq_gap_cum"].copy()
    st_gc["M"] = st_gc["Mois"].str.replace("M", "").astype(int)
    st_gc = st_gc.sort_values("M")
    pivot = st_gc.pivot_table(index="M", columns="Scenario", values="Gap_cumule_kEUR")
    st.line_chart(pivot)

    st.subheader("📌 Stress taux — ΔEVE par scénario")
    eve = res["eve_stress"].copy()
    eve["Label"] = eve["Type"] + " | " + eve["Scenario"]
    eve_plot = eve[["Label", "Delta_EVE_kEUR"]].set_index("Label")
    st.bar_chart(eve_plot)

    # ===== Tables (optionnelles)
    if show_details:
        st.subheader("Positions")
        st.dataframe(res["positions"])
        st.subheader("Gap Buckets")
        st.dataframe(res["gap_bucket"])
        st.subheader("Stress taux (table)")
        st.dataframe(res["eve_stress"])

    # ===== Export
    st.subheader("⬇️ Export")
    with open(res["output_file"], "rb") as f:
        st.download_button(
            label="Télécharger l’Excel de résultats",
            data=f,
            file_name="ALM_Resultats_COMPLET.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
