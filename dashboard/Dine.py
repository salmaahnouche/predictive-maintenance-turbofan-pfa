import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config import CLOUD_API_URL, FAULT_COLORS, RUL_CRITIQUE, RUL_DEGRADE

# ── Configuration page & Session State ────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance SOC",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "current_page" not in st.session_state:
    st.session_state.current_page = "Supervision Cockpit"
if "selected_engine" not in st.session_state:
    st.session_state.selected_engine = None

# ── Injection du CSS Custom (Style Dark SOC) ──────────────────────────────────
st.markdown("""
<style>
/* Style des blocs de métriques */
div[data-testid="metric-container"] {
    background-color: #16213e;
    border: 1px solid #0f3460;
    padding: 5% 10% 5% 10%;
    border-radius: 8px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
}
/* Classes CSS pour les bandeaux */
.alert-critique { background:#3a1515; color:#ff6b6b; border-left:4px solid #ff4757; }
.alert-warning  { background:#332616; color:#feca57; border-left:4px solid #ff9f43; }
.alert-normal   { background:#152b1b; color:#1dd1a1; border-left:4px solid #10ac84; }
</style>
""", unsafe_allow_html=True)

ALERT_CSS = {
    "CRITIQUE":      "alert-critique",
    "AVERTISSEMENT": "alert-warning",
    "NORMAL":        "alert-normal",
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
    st.title("⚙️ Maintenance SOC")
    st.markdown("---")

    auto_refresh     = st.toggle("Actualisation auto", value=False)
    refresh_interval = st.slider("Intervalle (s)", 5, 60, 15, disabled=not auto_refresh)

    st.markdown("---")
    
    liste_pages = ["Supervision Cockpit", "Vue Globale", "Analyse Détaillée", "Historique & Export"]
    index_defaut = liste_pages.index(st.session_state.current_page)
    
    page = st.radio("Navigation", liste_pages, index=index_defaut, label_visibility="collapsed")
    
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Actualiser maintenant"):
        st.rerun()

    health = _get("/health", {})
    if health.get("status") == "ok":
        st.success("API connectée ✅")
    else:
        st.error("API non disponible ❌")

    st.markdown("---")
    st.caption(f"Mis à jour : {datetime.now().strftime('%H:%M:%S')}")



# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SUPERVISION COCKPIT (Les 3 moteurs en direct)
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.current_page == "Supervision Cockpit":
    st.header("Supervision Temps Réel de la Flotte")

    latest = _get("/predictions/latest", [])
    if not latest:
        st.warning("Aucun moteur en base ou en attente de données...")
        st.stop()

    # ALERTE GÉNÉRALE (RUL < 30)
    moteurs_en_danger = [str(m["unit_number"]) for m in latest if m.get("rul_derniere", 100) < 30]
    if moteurs_en_danger:
        st.error(
            f"🚨 **ALERTE CRITIQUE : INTERVENTION IMMÉDIATE !** 🚨\n\n"
            f"Moteurs en danger : **#{', '.join(moteurs_en_danger)}** (RUL < 30 cycles)."
        )
    else:
        st.success("✅ Flotte stable. Aucun moteur sous le seuil critique (< 30 cycles).")

    st.markdown("---")

    # DISPOSITION EN COLONNES
    colonnes = st.columns(len(latest))

    for i, row in enumerate(latest):
        unit  = row.get("unit_number")
        alert = row.get("alert_level", "NORMAL")
        fault = row.get("fault_name") or "Aucune anomalie"
        conf  = row.get("confidence")
        rul   = row.get("rul_derniere")
        
        style = ALERT_CSS.get(alert, "")
        icon  = ALERT_ICON.get(alert, "⚪")

        with colonnes[i]:
            st.markdown(
                f"<div class='{style}' style='padding:15px; border-radius:10px; margin-bottom:15px;'>"
                f"<h3 style='margin-top:0;'>{icon} Moteur #{unit}</h3>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            st.metric("RUL Restant", f"{rul:.1f} cycles" if rul else "—")
            st.metric("Diagnostic IA", fault)
            if conf:
                st.caption(f"Confiance : **{conf:.0%}**")
            
            history = _get(f"/predictions/history/{unit}", [])
            if history:
                df_hist = pd.DataFrame(history)
                df_hist["time_cycles"] = pd.to_numeric(df_hist["time_cycles"])
                df_hist["rul"]         = pd.to_numeric(df_hist["rul"], errors="coerce")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_hist["time_cycles"], y=df_hist["rul"],
                    mode="lines", name="RUL",
                    line=dict(color="#3498db", width=3),
                ))
                fig.add_hline(y=30, line_dash="dot", line_color="#ff4757")
                fig.update_layout(
                    template="plotly_dark", height=200, 
                    margin=dict(t=10, b=10, l=10, r=10),
                    xaxis=dict(visible=False), yaxis=dict(title="RUL"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button(f"🔍 Analyser", key=f"cockpit_btn_{unit}", use_container_width=True):
                    st.session_state.selected_engine = unit
                    st.session_state.current_page = "Analyse Détaillée"
                    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — VUE GLOBALE (Stats)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "Vue Globale":
    st.header("Statistiques Globales")

    summary = _get("/stats/summary", {})
    latest  = _get("/predictions/latest", [])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Supervisés", summary.get("total_units", 0))
    c2.metric("🔴 Critique", summary.get("critiques", 0))
    c3.metric("🟡 Avertissement", summary.get("avertissements", 0))
    c4.metric("🟢 Normal", summary.get("normaux", 0))
    c5.metric("RUL moyen", f"{summary.get('avg_rul', 0):.1f}" if summary.get('avg_rul') else "—")

    st.divider()
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Liste des Moteurs")
        if latest:
            df_l = pd.DataFrame(latest).sort_values("unit_number")
            for _, row in df_l.iterrows():
                u_num = row['unit_number']
                alert = row.get("alert_level", "NORMAL")
                
                st.markdown(
                    f"<div class='{ALERT_CSS.get(alert, '')}' style='padding:10px;margin:5px 0;border-radius:6px;'>"
                    f"{ALERT_ICON.get(alert, '⚪')} <b>Moteur #{u_num}</b> — RUL : {row.get('rul_derniere', 0):.1f} | {row.get('fault_name', '—')}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button(f"👁️ Détails Moteur #{u_num}", key=f"global_btn_{u_num}"):
                    st.session_state.selected_engine = u_num
                    st.session_state.current_page = "Analyse Détaillée"
                    st.rerun()

    with col_r:
        st.subheader("Répartition des pannes")
        fdist = summary.get("fault_distribution", [])
        if fdist:
            fig = px.pie(
                pd.DataFrame(fdist), names="fault_name", values="n",
                color="fault_name", hole=0.4, template="plotly_dark",
                color_discrete_map={"HPC Degradation": FAULT_COLORS[0], "Fan Degradation": FAULT_COLORS[1]}
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Distribution RUL")
    if latest:
        df_l = pd.DataFrame(latest).dropna(subset=["rul_derniere"])
        df_l["alert_level"] = df_l["alert_level"].fillna("NORMAL")
        fig = px.scatter(
            df_l, x="unit_number", y="rul_derniere", color="alert_level",
            color_discrete_map={"CRITIQUE": "#ff4757", "AVERTISSEMENT": "#ff9f43", "NORMAL": "#10ac84"},
            symbol="fault_name", template="plotly_dark", height=350,
        )
        fig.add_hline(y=RUL_CRITIQUE, line_dash="dash", line_color="#ff4757")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYSE DÉTAILLÉE
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "Analyse Détaillée":
    st.header("Analyse Profonde")

    latest   = _get("/predictions/latest", [])
    unit_ids = sorted({r["unit_number"] for r in latest}) if latest else []

    if not unit_ids:
        st.warning("Aucun moteur en base.")
        st.stop()

    index_defaut = unit_ids.index(st.session_state.selected_engine) if st.session_state.selected_engine in unit_ids else 0
    unit = st.selectbox("Sélectionner un moteur", unit_ids, index=index_defaut, format_func=lambda u: f"Moteur #{u}")
    st.session_state.selected_engine = unit

    history = _get(f"/predictions/history/{unit}", [])
    if not history:
        st.info("Historique vide.")
        st.stop()

    df_hist = pd.DataFrame(history)
    df_hist["time_cycles"] = pd.to_numeric(df_hist["time_cycles"])
    df_hist["rul"]         = pd.to_numeric(df_hist["rul"], errors="coerce")

    last_pred = next((r for r in latest if r["unit_number"] == unit), {})
    c1, c2, c3 = st.columns(3)
    c1.metric("RUL", f"{last_pred.get('rul_derniere', 0):.1f}")
    c2.metric("Panne", last_pred.get("fault_name", "—"))
    c3.metric("Confiance", f"{last_pred.get('confidence', 0):.0%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hist["time_cycles"], y=df_hist["rul"], mode="lines+markers", 
        name="RUL", line=dict(color="#3498db", width=2)
    ))
    fig.add_hline(y=RUL_CRITIQUE, line_dash="dash", line_color="#ff4757", annotation_text="Critique")
    fig.add_hline(y=RUL_DEGRADE,  line_dash="dash", line_color="#ff9f43", annotation_text="Dégradé")
    fig.update_layout(template="plotly_dark", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HISTORIQUE & EXPORT
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == "Historique & Export":
    st.header("Données & Exports")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬇️ Exporter toutes les prédictions (CSV)"):
            try:
                r = requests.get(f"{CLOUD_API_URL}/export/csv", timeout=10)
                st.download_button("💾 Confirmer le téléchargement", data=r.content, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error("Export API non configuré ou indisponible.")
                
    with col2:
        if st.button("🗑️ Vider la base de données (Reset)"):
            st.error("Fonction de réinitialisation requiert un Token Admin côté API.")

    st.subheader("Dernières sessions")
    sessions = _get("/sessions", [])
    if sessions:
        st.dataframe(pd.DataFrame(sessions), use_container_width=True, hide_index=True)

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
