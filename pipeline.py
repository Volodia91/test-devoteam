import json
import os
from textwrap import indent

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from sklearn.tree import DecisionTreeClassifier, export_text

load_dotenv()

NUMERIC_COLS = ['cpu_usage', 'memory_usage', 'latency_ms', 'disk_usage', 'network_in_kbps',
                'network_out_kbps', 'io_wait', 'thread_count', 'active_connections',
                'error_rate', 'temperature_celsius', 'power_consumption_watts']
SERVICES = ["database", "api_gateway", "cache"]
STATE_COLS = [f"{s}_raw" for s in SERVICES]
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
STATUS_RANK = {"online": 0, "degraded": 1, "offline": 2}
STATUS_SEVERITY = {"online": "low", "degraded": "medium", "offline": "high"}


def load_dataset(source):
    """Rend (dataset, raw_status_df) à partir d'un fichier ou d'un buffer JSON."""
    dataset = pd.read_json(source)
    raw_status_df = dataset["service_status"].apply(pd.Series)
    return dataset, raw_status_df


# ---------- détection d'anomalies ----------

def incident_labels(df):
    """Cible apprise : 1 dès qu'un service n'est pas `online` sur le relevé."""
    status = df["service_status"].apply(pd.Series)
    return (status != "online").any(axis=1).astype(int)


def fit_incident_tree(df):
    """Arbre de décision peu profond appris sur la période : 12 métriques en
    entrée, l'incident en cible.

    Profondeur 2 volontairement : les seuils restent lisibles (`export_text`) et
    directement exploitables comme règles d'alerte, là où le z-score ne fait que
    mesurer un écart à la moyenne globale sans lien avec l'état des services.

    Rend None si la période ne contient qu'une classe : sans contre-exemple,
    aucun seuil n'est apprenable.
    """
    y = incident_labels(df)
    if y.nunique() < 2:
        return None
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, random_state=0)
    tree.fit(df[NUMERIC_COLS], y)
    return tree


def describe_rules(tree):
    """Règles apprises, telles qu'elles sont données au LLM."""
    if tree is None:
        return "  Aucune règle apprise : une seule classe sur la période."
    return indent(export_text(tree, feature_names=NUMERIC_COLS).rstrip(), "  ")


def relative_gap(value: float, bound: float):
    """Écart signé au seuil appris, en % (en valeur absolue si le seuil est nul)."""
    return round((value - bound) / bound * 100, 1) if bound else round(value - bound, 2)


def severity_from_deviation(deviation: float):
    """Sévérité = amplitude du franchissement, en % du seuil appris.

    L'arbre ne gradue pas le risque : sur une feuille pure, tous les relevés
    classés incident ont une confiance de 1.0 et seraient tous `high`. L'écart au
    seuil, lui, sépare le relevé qui l'effleure (+1 %) de celui qui le double."""
    gap = abs(deviation)
    return "high" if gap >= 50 else "medium" if gap >= 20 else "low"


def detect_anomalies(df, tree=None, min_confidence: float = 0.5):
    """Liste toutes les anomalies du dataset : pour chaque relevé classé incident
    par l'arbre, une entrée par condition du chemin de décision qui l'a classé
    (une métrique x le seuil appris franchi)."""
    tree = tree if tree is not None else fit_incident_tree(df)
    if tree is None:
        return []

    X = df[NUMERIC_COLS]
    confidences = tree.predict_proba(X)[:, 1]
    paths = tree.decision_path(X)
    inner = tree.tree_

    anomalies = []
    for i, (idx, confidence) in enumerate(zip(df.index, confidences)):
        if confidence < min_confidence:
            continue
        for node in paths.indices[paths.indptr[i]:paths.indptr[i + 1]]:
            feature = inner.feature[node]
            if feature < 0:  # feuille : plus aucune condition à expliciter
                continue
            col = NUMERIC_COLS[feature]
            value = float(df.loc[idx, col])
            bound = float(inner.threshold[node])
            # sens du franchissement : la branche empruntée par ce relevé
            op = "<=" if value <= bound else ">"
            bound = round(bound, 2)
            deviation = relative_gap(value, bound)
            anomalies.append({
                "timestamp": df.loc[idx, "timestamp"],
                "metric": col,
                "value": value,
                "threshold": bound,
                "deviation": deviation,
                "confidence": round(float(confidence), 2),
                "rule": f"{col} {op} {bound}",
                "severity": severity_from_deviation(deviation),
                "description": f"{col} a atteint {value}, règle `{col} {op} {bound}` "
                               f"de l'arbre vérifiée : relevé classé incident "
                               f"(écart au seuil {deviation:+.0f} %, confiance {confidence:.0%}).",
            })

    # par relevé chronologique, puis de l'écart le plus fort au plus faible
    anomalies.sort(key=lambda a: (a["timestamp"], -abs(a["deviation"])))
    return anomalies


def compute_insights(df):
    insight = {
        "average_latency_ms": round(float(df["latency_ms"].mean()), 2),
        "max_cpu_usage": int(df["cpu_usage"].max()),
        "max_memory_usage": int(df["memory_usage"].max()),
        "error_rate": round(float(df["error_rate"].mean()), 4),
        "uptime_seconds": int(df["uptime_seconds"].max()),
    }
    return insight


# ---------- regroupement par état des services ----------

def worst(severities):
    return max(severities, key=SEVERITY_ORDER.get)


def compute_baseline(dataset_states):
    """Baseline = les relevés où les trois services sont `online`.

    Rend (deltas, stats) :
    - deltas : écart en % de la moyenne de chaque groupe d'état à cette baseline.
      Second signal, complémentaire de l'arbre : celui-ci ne retient que les 1 à 3
      métriques qui séparent le mieux les incidents et reste muet sur les autres,
      même quand elles dérivent nettement sur un groupe (ex. api_gateway+cache
      degraded : +68 % de latence, qui n'apparaît dans aucune règle) ;
    - stats : plages saines par métrique, pour que les paramètres proposés
      (plafonds, seuils) restent compatibles avec le trafic normal.
    """
    all_online = (dataset_states[STATE_COLS] == "online").all(axis=1)
    if not all_online.any():
        return pd.DataFrame(columns=NUMERIC_COLS), pd.DataFrame()
    healthy = dataset_states.loc[all_online, NUMERIC_COLS]
    group_mean = dataset_states.groupby(STATE_COLS)[NUMERIC_COLS].mean()
    deltas = ((group_mean / healthy.mean().replace(0, pd.NA) - 1) * 100).round(0)
    stats = healthy.agg(["mean", "min", "max"]).T.join(
        healthy.quantile(0.95).rename("p95")
    ).round(2)
    return deltas, stats


def format_baseline_ranges(stats):
    return "\n".join(
        f"  - {metric}: moyenne {row['mean']}, p95 {row['p95']}, "
        f"observé {row['min']}→{row['max']}"
        for metric, row in stats.iterrows()
    )


def summarize_by_service_state(dataset, raw_status_df, anomalies):
    """Regroupe les relevés par état des services (database, api_gateway, cache).

    Rend (state_summary, metric_profile, baseline_deltas, baseline_stats) :
    - state_summary : un état par ligne — relevés observés, relevés anormaux, sévérité ;
    - metric_profile : (état x métrique) — récurrence et amplitude des règles franchies ;
    - baseline_deltas : (état x métrique) — écart de la moyenne du groupe au tout-online.
    """
    dataset_states = dataset.join(raw_status_df[SERVICES].add_suffix("_raw"))
    # colonnes explicites : une période sans aucun relevé classé incident (arbre
    # non apprenable, ou tout `online`) donne une liste vide, pas un merge cassé.
    anomalies_df = pd.DataFrame(
        anomalies,
        columns=["timestamp", "metric", "value", "threshold", "deviation", "rule", "severity"],
    ).merge(dataset_states[["timestamp", *STATE_COLS]], on="timestamp")

    metric_profile = (
        anomalies_df.groupby([*STATE_COLS, "metric"])
        .agg(n_events=("timestamp", "nunique"), rule=("rule", "first"),
             threshold=("threshold", "first"),
             value_min=("value", "min"), value_max=("value", "max"),
             mean_deviation=("deviation", lambda s: round(s.mean(), 1)),
             max_deviation=("deviation", lambda s: round(float(s.loc[s.abs().idxmax()]), 1)))
        .sort_values(["n_events", "max_deviation"], ascending=False,
                     key=lambda c: c.abs() if c.name == "max_deviation" else c)
    )

    state_summary = (
        dataset_states.groupby(STATE_COLS).size().rename("n_samples").to_frame()
        .join(anomalies_df.groupby(STATE_COLS).agg(
            n_events=("timestamp", "nunique"),
            timestamps=("timestamp", lambda s: sorted(s.unique())),
            anomaly_severity=("severity", worst)))
    )
    state_summary["n_events"] = state_summary["n_events"].fillna(0).astype(int)
    state_summary["severity"] = [
        worst([STATUS_SEVERITY[max(state, key=STATUS_RANK.get)],
               severity if isinstance(severity, str) else "low"])
        for state, severity in zip(state_summary.index, state_summary["anomaly_severity"])
    ]

    # états à traiter : un service non-online, ou des relevés anormaux. Un service
    # `offline` sans métrique hors seuil reste un incident (cas database=offline).
    degraded = pd.Series([any(v != "online" for v in state) for state in state_summary.index],
                         index=state_summary.index)
    state_summary = (
        state_summary[degraded | (state_summary["n_events"] > 0)]
        .drop(columns="anomaly_severity")
        .sort_values(["severity", "n_events"],
                     key=lambda c: c.map(SEVERITY_ORDER) if c.name == "severity" else c,
                     ascending=False)
    )
    baseline_deltas, baseline_stats = compute_baseline(dataset_states)
    return state_summary, metric_profile, baseline_deltas, baseline_stats


def build_clusters(state_summary, metric_profile, baseline_deltas, baseline_stats, rules):
    """Un groupe = une combinaison d'états des services, prête pour le prompt :
    c'est l'unité de diagnostic, les mêmes services dégradés ont la même cause racine."""
    profiles = {state: df.reset_index(level=STATE_COLS, drop=True).reset_index().to_dict("records")
                for state, df in metric_profile.groupby(level=STATE_COLS)}
    return [{
        "id": "|".join(f"{s}={v}" for s, v in zip(SERVICES, state)),
        "service_state": dict(zip(SERVICES, state)),
        "n_samples": int(row["n_samples"]),
        "n_events": int(row["n_events"]),
        "severity": row["severity"],
        "timestamps": row["timestamps"] if isinstance(row["timestamps"], list) else [],
        "metrics": profiles.get(state, []),
        "baseline_delta": (baseline_deltas.loc[state].dropna().to_dict()
                           if state in baseline_deltas.index else {}),
        "baseline_ranges": format_baseline_ranges(baseline_stats),
        "model_rules": rules,
    } for state, row in state_summary.iterrows()]


# ---------- génération des recommandations ----------

class Recommendation(BaseModel):
    id: str
    action: str
    target: str
    parameters: dict
    benefit_estimate: str


llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.4"), temperature=0)

# ---------- 1 état des services: 1 recommandation ----------
cluster_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es SRE. On te donne tous les relevés qui partagent le MÊME état des services "
     "(database / api_gateway / cache). Tu proposes UNE action corrective qui traite la cause "
     "racine commune à ce groupe, jamais une action par métrique. `target` est un service "
     "existant (database, api_gateway ou cache), `parameters` contient les valeurs concrètes "
     "de l'action (seuils, tailles, durées) — cohérentes avec les plages normales indiquées.\n"
     "Deux signaux distincts te sont donnés :\n"
     "- les règles d'un arbre de décision de profondeur 2, appris à prédire l'incident "
     "(un service non `online`) à partir des 12 métriques : ce sont les seuils qui séparent "
     "le mieux les relevés en incident des relevés sains, donc les métriques déclenchantes ;\n"
     "- l'écart des moyennes du groupe à la baseline tout-online, qui est la dérive de fond "
     "du groupe et existe même si aucune règle n'est franchie.\n"
     "L'arbre ne retient que quelques métriques : une métrique absente des règles peut quand "
     "même dériver, la dérive de fond fait foi pour celles-là. Un groupe sans règle franchie "
     "mais dont les moyennes dérivent reste une dégradation réelle. Un groupe plat sur les "
     "deux signaux, avec un service `offline` ou `degraded`, signale une panne applicative "
     "et non une saturation de ressource."),
    ("human",
     "État des services : {service_state}\n"
     "{n_events} relevés anormaux sur {n_samples} dans cet état — sévérité {severity}\n"
     "Relevés anormaux (timestamps) :\n{period}\n\n"
     "Règles apprises sur l'ensemble de la période (arbre de profondeur 2) :\n{model_rules}\n\n"
     "Seuils franchis par ce groupe (récurrence et amplitude) :\n{metrics}\n\n"
     "Dérive de fond (moyenne du groupe vs baseline tout-online) :\n{baseline_delta}\n\n"
     "Plages observées hors dégradation (baseline tout-online) :\n{baseline_ranges}\n\n"
     "Utilise `{cluster_id}` comme `id`."),
])


def cluster_to_vars(c: dict):
    deltas = sorted(c["baseline_delta"].items(), key=lambda kv: -abs(kv[1]))
    return {
        "cluster_id": c["id"],
        "service_state": ", ".join(f"{s}: {v}" for s, v in c["service_state"].items()),
        "n_events": c["n_events"],
        "n_samples": c["n_samples"],
        "severity": c["severity"],
        "period": "\n".join(f"  - {t:%Y-%m-%d %H:%M}" for t in c["timestamps"])
                  or "  aucun relevé anormal",
        "metrics": "\n".join(
            f"  - `{m['rule']}` : {m['n_events']}/{c['n_events']} relevés, "
            f"valeurs {m['value_min']}→{m['value_max']}, "
            f"écart au seuil {m['mean_deviation']:+.0f}% en moyenne "
            f"(max {m['max_deviation']:+.0f}%)"
            for m in c["metrics"]
        ) or "  Aucune règle franchie : l'arbre ne classe aucun relevé de ce groupe en incident.",
        "baseline_delta": "\n".join(
            f"  - {metric}: {delta:+.0f}%" for metric, delta in deltas if abs(delta) >= 5
        ) or "  Aucune dérive : toutes les métriques à moins de 5 % de la baseline.",
        "baseline_ranges": c["baseline_ranges"],
        "model_rules": c["model_rules"],
    }


cluster_chain = (
    RunnableLambda(cluster_to_vars)
    | cluster_prompt
    | llm.with_structured_output(Recommendation, method="function_calling")
).with_retry(stop_after_attempt=3)


def generate_recommendations(clusters: list[dict]):
    """Une recommandation par état des services : quelques appels, contre un appel
    par relevé anormal. Les groupes sont disjoints par construction — un état des
    services = une cause racine —, il n'y a donc pas de doublons à consolider."""
    if not clusters:
        return []
    return cluster_chain.batch(clusters, config={"max_concurrency": 4})


# ---------- rapport ----------

def compute_service_status_summary(raw_status_df):
    """Classe chaque service selon l'état le plus dégradé observé sur la période."""
    summary = {"online": [], "degraded": [], "offline": []}
    for service in SERVICES:
        worst_status = max(raw_status_df[service], key=STATUS_RANK.get)
        summary[worst_status].append(service)
    return summary


def format_report_anomaly(a: dict):
    """Anomalie au format de sortie attendu ; le relevé concerné passe dans la description."""
    return {
        "metric": a["metric"],
        "value": a["value"],
        "threshold": a["threshold"],
        "severity": a["severity"],
        "description": f"[{a['timestamp']:%Y-%m-%d %H:%M}] {a['description']}",
    }


def make_report(dataset, raw_status_df):
    tree = fit_incident_tree(dataset)
    anomalies = detect_anomalies(dataset, tree)
    # recommandations générées par état des services, pas par relevé
    summary, profile, deltas, stats = summarize_by_service_state(dataset, raw_status_df, anomalies)
    clusters = build_clusters(summary, profile, deltas, stats, describe_rules(tree))
    recommendations = generate_recommendations(clusters)

    report = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "insights": compute_insights(dataset),
        "anomalies": [format_report_anomaly(a) for a in anomalies],
        "recommendations": [rec.model_dump() for rec in recommendations],
        "service_status_summary": compute_service_status_summary(raw_status_df),
    }
    return json.loads(json.dumps(report, default=str))
