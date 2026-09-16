import json

import altair as alt
import pandas as pd
import streamlit as st

from pipeline import (
    NUMERIC_COLS,
    describe_rules,
    detect_anomalies,
    fit_incident_tree,
    load_dataset,
    make_report,
    severity_from_deviation,
)

st.set_page_config(layout="wide")
st.title("Analyse de logs")

# une seule série par graphique : bleu pour la métrique, une échelle rouge →
# orange → jaune pour la sévérité des relevés anormaux. Teintes assombries en
# thème clair (le jaune vif est illisible sur fond blanc), éclaircies en sombre.
DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
SERIES = "#3987e5" if DARK else "#2a78d6"
SURFACE = "#1a1a19" if DARK else "#fcfcfb"
SEVERITY_COLORS = ({"high": "#e35d5d", "medium": "#f0902f", "low": "#e0c03c"} if DARK
                   else {"high": "#d03b3b", "medium": "#d97a1e", "low": "#b89320"})
alt.data_transformers.disable_max_rows()  # 12 métriques x N relevés dépassent la limite par défaut


def anomaly_points(anomalies):
    """Un relevé anormal par ligne, quelles que soient les règles qui l'ont classé.

    L'arbre classe le relevé entier et non la métrique qui porte le seuil : le
    point rouge se pose donc sur les 12 courbes au même instant, et l'infobulle
    dit quelle règle l'a déclenché."""
    points = pd.DataFrame(anomalies, columns=["timestamp", "rule", "deviation"])
    points = points.groupby("timestamp", as_index=False).agg(
        rule=("rule", lambda s: " et ".join(dict.fromkeys(s))),
        deviation=("deviation", lambda s: s.loc[s.abs().idxmax()]),
    )
    # sévérité recalculée sur l'écart retenu
    points["severity"] = points["deviation"].map(severity_from_deviation)
    return points


def metrics_chart(dataset, anomalies, metrics):
    """Petits multiples : une métrique par graphique (échelles indépendantes),
    avec les relevés classés incident marqués en rouge sur chacune."""
    values = dataset.melt(id_vars="timestamp", value_vars=metrics,
                          var_name="metric", value_name="value")
    # rattaché au relevé (timestamp), pas au couple (relevé, métrique)
    values = values.merge(anomaly_points(anomalies), on="timestamp", how="left")

    base = alt.Chart(values).encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("value:Q", title=None).scale(zero=False),
        tooltip=["timestamp:T", "value:Q",
                 alt.Tooltip("rule:N", title="règle franchie"),
                 alt.Tooltip("deviation:Q", title="écart au seuil de la règle (%)"),
                 alt.Tooltip("severity:N", title="sévérité")],
    )
    return alt.layer(
        base.mark_line(color=SERIES, strokeWidth=2),
        # cible de survol invisible : la valeur de chaque relevé au passage
        base.mark_point(opacity=0, size=60),
        base.transform_filter("isValid(datum.rule)").mark_point(
            filled=True, size=45, stroke=SURFACE, strokeWidth=2, opacity=1).encode(
            # domaine fixe : la même sévérité garde la même couleur d'un fichier
            # à l'autre, même si un niveau est absent du jeu chargé.
            color=alt.Color("severity:N", title="Sévérité").scale(
                domain=list(SEVERITY_COLORS), range=list(SEVERITY_COLORS.values())
            ).legend(orient="top", symbolType="circle", symbolSize=80)),
    ).properties(width=300, height=110).facet(
        facet=alt.Facet("metric:N", title=None, sort=metrics,
                        header=alt.Header(labelFontWeight="bold")),
        columns=3,
    ).resolve_scale(x="shared", y="independent")


uploaded = st.file_uploader("Fichier de logs (.json)", type="json")

if uploaded:
    dataset, raw_status_df = load_dataset(uploaded)
    tree = fit_incident_tree(dataset)
    anomalies = detect_anomalies(dataset, tree)

    with st.expander("Règles apprises (arbre de décision, profondeur 2)"):
        st.code(describe_rules(tree))

    metrics = st.multiselect("Métriques", NUMERIC_COLS, default=NUMERIC_COLS)
    if metrics:
        st.altair_chart(metrics_chart(dataset, anomalies, metrics), use_container_width=False)
        st.caption(f"Points colorés : {len({a['timestamp'] for a in anomalies})} relevés classés "
                   "incident par l'arbre, marqués sur toutes les métriques et colorés par "
                   "sévérité — la règle franchie ne porte que sur une métrique, la dégradation "
                   "se lit sur les autres.")

    if st.button("Générer le rapport"):
        with st.spinner("Analyse en cours..."):
            report = make_report(dataset, raw_status_df)
        st.json(report, expanded=False)
        st.download_button(
            "Télécharger rapport.json",
            json.dumps(report, indent=2, ensure_ascii=False),
            file_name="rapport.json",
            mime="application/json",
        )
