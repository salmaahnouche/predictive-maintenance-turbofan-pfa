"""
dashboard.py — Tableau de bord Streamlit (Cloud)
==================================================
Visualisation temps réel des prédictions RUL et diagnostics de panne.

Lancer : streamlit run dashboard.py
"""

import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config import CLOUD_API_URL, FAULT_COLORS, RUL_CRITIQUE, RUL_DEGRADE

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALERT_CSS = {
    "CRITIQUE":      "background:#fde8e8;color:#a32d2d;border-left:4px solid #e24b4a;",
    "AVERTISSEMENT": "background:#fef9ec;color:#854f0b;border-left:4px solid #ef9f27;",
    "NORMAL":        "background:#eaf3de;color:#3b6d11;border-left:4px solid #639922;",
}
ALERT_ICON = {"CRITIQUE": "🔴", "AVERTISSEMENT": "🟡", "NORMAL": "🟢"}


# ── Helpers API ───────────────────────────────────────────────────────────────
def _get(endpoint: str, default=None):
    try:
        r = requests.get(f"{CLOUD_API_URL}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Maintenance Prédictive")
    st.markdown("---")

    auto_refresh     = st.toggle("Actualisation automatique", value=False)
    refresh_interval = st.slider("Intervalle (s)", 5, 60, 15, disabled=not auto_refresh)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Vue globale", "Analyse moteur", "Historique sessions", "Export données"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("🔄 Actualiser maintenant"):
        st.rerun()

    health = _get("/health", {})
    if health.get("status") == "ok":
        st.success("API connectée ✅")
        st.caption(f"Modèle RF : {'chargé ✅' if health.get('clf_loaded') else 'absent ⚠️'}")
    else:
        st.error("API non disponible ❌")
        st.caption(f"URL : {CLOUD_API_URL}")

    st.markdown("---")
    st.caption(f"Mis à jour : {datetime.now().strftime('%H:%M:%S')}")

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE GLOBALE
# ═════════════════════════════════════════════════════════════════════════════
if page == "Vue globale":
    st.header("Vue d'ensemble — État de la flotte")

    summary = _get("/stats/summary", {})
    latest  = _get("/predictions/latest", [])

    if not summary:
        st.warning("Aucune donnée disponible. Démarrez l'API et envoyez des données depuis le PC Edge.")
        st.stop()

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Moteurs supervisés", summary.get("total_units", 0))
    c2.metric("🔴 Critique",        summary.get("critiques", 0))
    c3.metric("🟡 Avertissement",   summary.get("avertissements", 0))
    c4.metric("🟢 Normal",          summary.get("normaux", 0))
    avg = summary.get("avg_rul")
    c5.metric("RUL moyen",          f"{avg:.1f} cycles" if avg else "—")

    st.divider()

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("État des moteurs")
        if latest:
            df_l = pd.DataFrame(latest).sort_values(
                "alert_level",
                key=lambda s: s.map({"CRITIQUE": 0, "AVERTISSEMENT": 1, "NORMAL": 2}),
            )
            for _, row in df_l.iterrows():
                alert = row.get("alert_level", "NORMAL")
                fault = row.get("fault_name") or "—"
                rul   = row.get("rul_derniere")
                conf  = row.get("confidence")
                cyc   = row.get("n_cycles_total", "?")
                style    = ALERT_CSS.get(alert, "")
                icon     = ALERT_ICON.get(alert, "⚪")
                conf_str = f" ({conf:.0%})" if conf else ""
                rul_str  = f"{rul:.1f}" if rul is not None else "—"
                st.markdown(
                    f"<div style='padding:8px 12px;margin:4px 0;"
                    f"border-radius:6px;{style}'>"
                    f"{icon} <b>Moteur #{row['unit_number']}</b> — "
                    f"RUL : <b>{rul_str}</b> | {fault}{conf_str} | {cyc} cycles"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Aucune prédiction reçue.")

    with col_r:
        st.subheader("Répartition des pannes")
        fdist = summary.get("fault_distribution", [])
        if fdist:
            df_f = pd.DataFrame(fdist)
            fig = px.pie(
                df_f, names="fault_name", values="n",
                color="fault_name",
                color_discrete_map={
                    "HPC Degradation": FAULT_COLORS[0],
                    "Fan Degradation":  FAULT_COLORS[1],
                },
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas encore de classifications.")

    st.divider()
    st.subheader("Distribution RUL — flotte complète")
    if latest:
        df_l = pd.DataFrame(latest).dropna(subset=["rul_derniere"])
        df_l["alert_level"] = df_l["alert_level"].fillna("NORMAL")
        fig = px.scatter(
            df_l, x="unit_number", y="rul_derniere",
            color="alert_level",
            color_discrete_map={
                "CRITIQUE": "#e24b4a", "AVERTISSEMENT": "#ef9f27", "NORMAL": "#639922",
            },
            symbol="fault_name",
            hover_data=["fault_name", "confidence", "n_cycles_total"],
            labels={"unit_number": "Moteur n°", "rul_derniere": "RUL (cycles)"},
            height=350,
        )
        fig.add_hline(y=RUL_CRITIQUE, line_dash="dash", line_color="#e24b4a",
                      annotation_text="Seuil critique")
        fig.add_hline(y=RUL_DEGRADE,  line_dash="dash", line_color="#ef9f27",
                      annotation_text="Seuil dégradé")
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE MOTEUR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Analyse moteur":
    st.header("Analyse détaillée d'un moteur")

    latest   = _get("/predictions/latest", [])
    unit_ids = sorted({r["unit_number"] for r in latest}) if latest else []

    if not unit_ids:
        st.warning("Aucun moteur en base.")
        st.stop()

    unit    = st.selectbox("Sélectionner un moteur", unit_ids,
                           format_func=lambda u: f"Moteur #{u}")
    history = _get(f"/predictions/history/{unit}", [])
    preds   = _get(f"/predictions?unit_number={unit}", [])

    if not history:
        st.info("Historique vide pour ce moteur.")
        st.stop()

    df_hist = pd.DataFrame(history)
    df_hist["time_cycles"] = pd.to_numeric(df_hist["time_cycles"])
    df_hist["rul"]         = pd.to_numeric(df_hist["rul"], errors="coerce")

    last_pred = next((r for r in latest if r["unit_number"] == unit), {})
    alert     = last_pred.get("alert_level", "NORMAL")
    fault     = last_pred.get("fault_name") or "—"
    conf      = last_pred.get("confidence")
    rul_now   = last_pred.get("rul_derniere")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("État actuel",    f"{ALERT_ICON.get(alert,'')} {alert}")
    c2.metric("RUL actuel",     f"{rul_now:.1f} cycles" if rul_now is not None else "—")
    c3.metric("Type de panne",  fault)
    c4.metric("Confiance IA",   f"{conf:.0%}" if conf is not None else "—")

    st.divider()

    # Courbe RUL
    st.subheader("Évolution du RUL")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hist["time_cycles"], y=df_hist["rul"],
        mode="lines+markers", name="RUL",
        line=dict(color="#3498db", width=2), marker=dict(size=4),
    ))
    fig.add_hline(y=RUL_CRITIQUE, line_dash="dash", line_color="#e24b4a",
                  annotation_text=f"Critique (≤{RUL_CRITIQUE})")
    fig.add_hline(y=RUL_DEGRADE,  line_dash="dash", line_color="#ef9f27",
                  annotation_text=f"Dégradé (≤{RUL_DEGRADE})")
    fig.update_layout(xaxis_title="Cycles", yaxis_title="RUL restant",
                      height=350, margin=dict(t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Alertes
    st.subheader("Historique des alertes")
    df_alerts = df_hist[df_hist["alert_level"].isin(["CRITIQUE", "AVERTISSEMENT"])]
    if df_alerts.empty:
        st.success("Aucune alerte enregistrée pour ce moteur.")
    else:
        st.dataframe(
            df_alerts[["time_cycles", "rul", "alert_level", "session_id"]].rename(columns={
                "time_cycles": "Cycle", "rul": "RUL",
                "alert_level": "Alerte", "session_id": "Session",
            }),
            use_container_width=True, hide_index=True,
        )

    if preds:
        with st.expander("Toutes les prédictions (détail)"):
            st.dataframe(pd.DataFrame(preds), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HISTORIQUE SESSIONS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Historique sessions":
    st.header("Historique des sessions Edge")

    sessions = _get("/sessions", [])
    if not sessions:
        st.info("Aucune session enregistrée.")
        st.stop()

    df_s = pd.DataFrame(sessions)
    df_s["premiere"]  = pd.to_datetime(df_s["premiere"])
    df_s["rul_moyen"] = df_s["rul_moyen"].round(1)
    df_s = df_s.sort_values("premiere", ascending=False)

    st.dataframe(
        df_s.rename(columns={
            "session_id":   "Session ID",
            "premiere":     "Première réception",
            "nb_moteurs":   "Moteurs",
            "rul_moyen":    "RUL moyen",
            "total_cycles": "Cycles totaux",
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Chronologie")
    fig = px.timeline(
        df_s.assign(fin=df_s["premiere"] + pd.Timedelta(minutes=5)),
        x_start="premiere", x_end="fin", y="session_id",
        color="rul_moyen", color_continuous_scale="RdYlGn",
        labels={"session_id": "Session", "rul_moyen": "RUL moyen"},
        height=300,
    )
    fig.update_layout(margin=dict(t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPORT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Export données":
    st.header("Export des données")

    latest = _get("/predictions/latest", [])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Export CSV — toutes les prédictions")
        if st.button("⬇️ Générer le CSV"):
            try:
                r = requests.get(f"{CLOUD_API_URL}/export/csv", timeout=10)
                st.download_button(
                    label="💾 Télécharger",
                    data=r.content,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Erreur : {e}")

    with col2:
        st.subheader("Résumé — JSON")
        summary = _get("/stats/summary", {})
        if summary:
            st.json(summary)

    st.divider()
    st.subheader("Dernières prédictions")
    if latest:
        df = pd.DataFrame(latest)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export tableau affiché",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="dernieres_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Aucune prédiction disponible.")