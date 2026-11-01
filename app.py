"""Stage 4: Plotly Dash dashboard for air traffic delay patterns."""
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from src import charts as ch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CAUSE_LABELS = {
    "AIR_SYSTEM_DELAY": "Air system", "SECURITY_DELAY": "Security", "AIRLINE_DELAY": "Airline",
    "LATE_AIRCRAFT_DELAY": "Late aircraft", "WEATHER_DELAY": "Weather",
}
CARD = {"background": "#111827", "border": "1px solid #1f2937", "borderRadius": "10px",
        "height": "100%"}

# ---------------------------------------------------------------- load once
airports = pd.read_parquet(OUT / "airport_stats.parquet")
routes_all = pd.read_parquet(OUT / "route_stats.parquet")
airline_stats = pd.read_parquet(OUT / "airline_stats.parquet")
cause_stats = pd.read_parquet(OUT / "cause_stats.parquet")
hour_day_all = pd.read_parquet(OUT / "hour_day_matrix.parquet")
monthly_all = pd.read_parquet(OUT / "monthly_stats.parquet")
graph_metrics = pd.read_csv(OUT / "graph_metrics.csv")
graph_edges = pd.read_csv(OUT / "graph_edges.csv")

AIRLINE_NAMES = dict(zip(airline_stats["AIRLINE"], airline_stats["airline_name"]))
AP_INFO = airports.set_index("ORIGIN_AIRPORT")[["lat", "lon", "airport_name", "city", "state"]]

fact = pd.read_parquet(OUT / "fact.parquet")
for c in ["AIRLINE", "airline_name", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "day_name"]:
    fact[c] = fact[c].astype("category")

FULL_N = len(fact)
_full_group = fact.groupby("ORIGIN_AIRPORT", observed=True)["ARRIVAL_DELAY"]
FULL_AVG_DELAY = float(fact["ARRIVAL_DELAY"].mean())
FULL_DELAY_RATE = float(fact["delayed"].mean())
WORST_AP = _full_group.mean().idxmax()
BUSIEST_AP = fact["ORIGIN_AIRPORT"].value_counts().idxmax()
WORST_HOUR = int(fact.groupby("hour", observed=True)["ARRIVAL_DELAY"].mean().idxmax())
TOP_CAUSE = (cause_stats[cause_stats["scope"] == "overall"].set_index("cause")["minutes"]
             .rename(index=CAUSE_LABELS).idxmax())
TOP_HUB = graph_metrics.iloc[0]

STATIC_FIGS = {
    "net_comm": ch.network_graph(graph_metrics, graph_edges, 60, "betweenness", "community"),
    "net_delay": ch.network_graph(graph_metrics, graph_edges, 60, "betweenness", "delay"),
    "centr_betweenness": ch.centrality_bar(graph_metrics, "betweenness"),
    "centr_pagerank": ch.centrality_bar(graph_metrics, "pagerank"),
    "centr_degree": ch.centrality_bar(graph_metrics, "degree"),
}


def insights_scatter() -> go.Figure:
    m = graph_metrics.dropna(subset=["lat", "lon"])
    fig = go.Figure(go.Scatter(
        x=m["betweenness"], y=m["avg_delay"], mode="markers",
        marker=dict(size=6 + 22 * (m["traffic"] / m["traffic"].max()).fillna(0),
                    color=m["community"], colorscale="Turbo", showscale=False,
                    line=dict(width=0.5, color="#0b1220"), opacity=0.85),
        customdata=m[["airport", "traffic", "pagerank"]],
        hovertemplate=("<b>%{customdata[0]}</b><br>betweenness: %{x:.3f}<br>"
                       "avg delay: %{y:.1f} min<br>traffic: %{customdata[1]:,}<br>"
                       "pagerank: %{customdata[2]:.4f}<extra></extra>")))
    fig.update_layout(title="Hub-ness (betweenness) vs average delay",
                      xaxis_title="Betweenness centrality", yaxis_title="Avg arrival delay (min)",
                      height=480)
    return ch._style(fig)


STATIC_FIGS["scatter"] = insights_scatter()

# ------------------------------------------------------------------ helpers
def empty_fig(msg: str = "No flights match the current filters") -> go.Figure:
    return ch._empty(msg)


def filter_fact(airline_codes, month_range, origin):
    f = fact
    if airline_codes:
        f = f[f["AIRLINE"].isin(airline_codes)]
    if month_range and len(month_range) == 2:
        f = f[f["MONTH"].between(month_range[0], month_range[1])]
    if origin:
        f = f[f["ORIGIN_AIRPORT"] == origin]
    return f


def adapt_threshold(n: int) -> int:
    return max(30, min(500, n // 1000))


def filtered_outputs(airline_codes, month_range, origin, cause, top_n):
    f = filter_fact(airline_codes, month_range, origin)
    if f.empty:
        fig = empty_fig()
        return ("—", "—", "—", "—", "—", [], fig, fig, fig, fig, fig, fig, fig, fig)

    thr = adapt_threshold(len(f))
    f = f.assign(month_name=f["MONTH"].map(lambda m: MONTHS[m - 1]))

    ap = (f.groupby("ORIGIN_AIRPORT", observed=True)
          .agg(flights=("ARRIVAL_DELAY", "size"), avg_delay=("ARRIVAL_DELAY", "mean"),
               delay_rate=("delayed", "mean")).reset_index()
          .merge(AP_INFO, left_on="ORIGIN_AIRPORT", right_index=True, how="left"))
    rt = (f.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], observed=True)
          .agg(flights=("ARRIVAL_DELAY", "size"), avg_delay=("ARRIVAL_DELAY", "mean"),
               delay_rate=("delayed", "mean"))
          .reset_index())
    hd = (f.groupby(["hour", "day_name"], observed=True)["ARRIVAL_DELAY"].mean()
          .reset_index(name="avg_delay"))

    mo = (f.groupby("month_name", observed=True)
          .agg(flights=("ARRIVAL_DELAY", "size"), avg_delay=("ARRIVAL_DELAY", "mean"),
               delay_rate=("delayed", "mean")).reset_index())
    mo["month_name"] = pd.Categorical(mo["month_name"], categories=MONTHS, ordered=True)
    mo = mo.sort_values("month_name").round({"avg_delay": 2, "delay_rate": 4})

    al = (f.groupby(["AIRLINE", "airline_name"], observed=True)
          .agg(avg_delay=("ARRIVAL_DELAY", "mean"), delay_rate=("delayed", "mean"))
          .reset_index().round({"avg_delay": 2, "delay_rate": 4}))

    cs = cause_stats.copy()
    if airline_codes:
        names = [AIRLINE_NAMES.get(a, a) for a in airline_codes]
        cs = cs[(cs["scope"] != "airline") | (cs["scope_value"].isin(names))]
    if month_range and len(month_range) == 2 and (month_range[0] > 1 or month_range[1] < 12):
        sel = MONTHS[month_range[0] - 1: month_range[1]]
        cs = cs[(cs["scope"] != "month") | (cs["scope_value"].isin(sel))]
    if cause:
        cs = cs[cs["cause"] == cause]

    worst = ap[ap["flights"] >= thr]
    worst_ap_code = worst.loc[worst["avg_delay"].idxmax(), "ORIGIN_AIRPORT"] if len(worst) else "—"
    busiest_code = ap.loc[ap["flights"].idxmax(), "ORIGIN_AIRPORT"]

    geo = ch.geo_map(ap, rt, top_n, positions=AP_INFO[["lat", "lon"]])
    heat = ch.heatmap_hour_day(hd)
    trend = ch.monthly_trend(mo)
    air_bar = ch.airline_bar(al)
    c_air = ch.causes_stacked(cs[cs["scope"] == "airline"], "airline") if len(cs[cs["scope"] == "airline"]) else empty_fig()
    c_mon = ch.causes_stacked(cs[cs["scope"] == "month"], "month") if len(cs[cs["scope"] == "month"]) else empty_fig()
    worst_f = ch.worst_bars(ap, rt, min_flights=thr)

    pct_delayed = f"{float(f['delayed'].mean()) * 100:.1f}%"
    kpis = [f"{len(f):,}", pct_delayed, f"{float(f['ARRIVAL_DELAY'].mean()):.1f} min", worst_ap_code, busiest_code]

    insights = [
        html.Li([html.Strong("Worst airport: "),
                 f"{worst_ap_code} — {AP_INFO.loc[worst_ap_code, 'airport_name'] if worst_ap_code in AP_INFO.index else ''} "
                 f"({float(worst['avg_delay'].max()):.1f} min avg)" if len(worst) else "n/a"]),
        html.Li([html.Strong("Worst hour: "), f"{WORST_HOUR}:00 departures (full-year pattern)"]),
        html.Li([html.Strong("Top delay cause: "), f"{TOP_CAUSE} (most total delay minutes)"]),
        html.Li([html.Strong("Most central hub: "), f"{TOP_HUB['airport']} — {TOP_HUB['airport_name']} "
                 f"(betweenness {TOP_HUB['betweenness']:.3f})"]),
        html.Li([html.Strong("Communities found: "),
                 f"{graph_metrics['community'].nunique()} route-network clusters (Louvain)"]),
    ]
    return (*kpis, insights, geo, heat, trend, air_bar, c_air, c_mon, worst_f, trend)


# ------------------------------------------------------------------- layout
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "US Air Traffic Delay Patterns"

app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
      body { background-color: #0b1220 !important; }
      .card { background-color: #111827 !important; border: 1px solid #1f2937 !important;
              border-radius: 10px !important; }
      .kpi-value { font-size: 1.45rem; font-weight: 700; color: #38bdf8; }
      .kpi-label { font-size: .7rem; color: #9ca3af; text-transform: uppercase;
                   letter-spacing: .06em; }
      .insights li { margin-bottom: .45rem; color: #d1d5db; }
      .sidebar .form-label { color: #9ca3af; font-size: .78rem; text-transform: uppercase;
                              letter-spacing: .05em; margin-bottom: .25rem; }
      .nav-link.active { background-color: #111827 !important; color: #38bdf8 !important; }
      h4.app-title { color: #f9fafb; font-weight: 700; letter-spacing: .02em; }
      .subtitle { color: #6b7280; font-size: .85rem; }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>"""


def kpi_card(label: str, placeholder: str) -> dbc.Col:
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(placeholder, id=f"kpi-{label}", className="kpi-value"),
        html.Div(label, className="kpi-label"),
    ]), style=CARD), width=True, className="px-1")


def sidebar() -> dbc.Col:
    airline_opts = [{"label": f"{n} ({c})", "value": c}
                    for c, n in sorted(AIRLINE_NAMES.items(), key=lambda kv: kv[1])]
    return dbc.Col([
        html.Div([
            html.Label("Airline"),
            dcc.Dropdown(id="flt-airline", options=airline_opts, multi=True,
                         placeholder="All airlines"),
        ], className="mb-3"),
        html.Div([
            html.Label("Month range"),
            dcc.RangeSlider(id="flt-month", min=1, max=12, step=1, value=[1, 12],
                            marks={i: m for i, m in enumerate(MONTHS, start=1)},
                            allowCross=False, tooltip={"placement": "bottom"}),
        ], className="mb-3"),
        html.Div([
            html.Label("Origin airport"),
            dcc.Dropdown(id="flt-origin",
                         options=[{"label": f"{r.ORIGIN_AIRPORT} — {r.airport_name}",
                                   "value": r.ORIGIN_AIRPORT}
                                  for r in airports.sort_values("flights", ascending=False).itertuples()],
                         placeholder="All airports"),
        ], className="mb-3"),
        html.Div([
            html.Label("Delay cause"),
            dcc.Dropdown(id="flt-cause", clearable=True, placeholder="All causes",
                         options=[{"label": v, "value": k} for k, v in CAUSE_LABELS.items()]),
        ], className="mb-3"),
        html.Div([
            html.Label(f"Top-N routes on map"),
            dcc.Slider(id="flt-topn", min=10, max=100, step=5, value=40,
                       marks={10: "10", 50: "50", 100: "100"}),
        ], className="mb-3"),
        html.Hr(style={"borderColor": "#1f2937"}),
        html.Div([
            html.Label("Network node color"),
            dcc.RadioItems(id="net-color", options=[{"label": " Community", "value": "community"},
                                                    {"label": " Avg delay", "value": "delay"}],
                           value="community", inputClassName="form-check-input",
                           labelClassName="text-secondary"),
        ], className="mb-3"),
        html.Div([
            html.Label("Centrality metric"),
            dcc.Dropdown(id="cent-metric", value="betweenness", clearable=False,
                         options=[{"label": "Betweenness", "value": "betweenness"},
                                  {"label": "PageRank", "value": "pagerank"},
                                  {"label": "Degree", "value": "degree"}]),
        ], className="mb-3"),
        html.Div(id="filter-status", className="text-secondary small"),
    ], width=12, lg=2, className="sidebar")


tab_overview = dbc.Tab(label="Overview", tab_id="overview", labelClassName="text-light", children=[
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Key Insights", className="kpi-label"),
                                       html.Ul(id="insights-list", className="insights mb-0")]),
                         style=CARD), width=12, lg=4, className="mb-3"),
        dbc.Col(dcc.Graph(id="fig-monthly"), width=12, lg=8, className="mb-3"),
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id="fig-worst"), width=12)]),
])

tab_map = dbc.Tab(label="Map & Network", tab_id="map", labelClassName="text-light", children=[
    dbc.Row([dbc.Col(dcc.Graph(id="fig-geo"), width=12)]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-network"), width=12, xl=7),
        dbc.Col(dcc.Graph(id="fig-centrality"), width=12, xl=5),
    ]),
])

tab_time = dbc.Tab(label="Time Patterns", tab_id="time", labelClassName="text-light", children=[
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-heat"), width=12, xl=7),
        dbc.Col(dcc.Graph(id="fig-monthly-2"), width=12, xl=5),
    ]),
])

tab_airlines = dbc.Tab(label="Airlines & Causes", tab_id="airlines", labelClassName="text-light", children=[
    dbc.Row([dbc.Col(dcc.Graph(id="fig-airline"), width=12)]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-causes-airline"), width=12, xl=6),
        dbc.Col(dcc.Graph(id="fig-causes-month"), width=12, xl=6),
    ]),
])

tab_graph = dbc.Tab(label="Graph Insights", tab_id="graph", labelClassName="text-light", children=[
    dbc.Row([dbc.Col(dcc.Graph(id="fig-scatter", figure=STATIC_FIGS["scatter"]), width=12)]),
    html.Div([
        html.H6("How to read this", className="kpi-label mb-2"),
        html.Ul([
            html.Li("Nodes are airports positioned geographically; edges are routes with ≥100 flights."),
            html.Li("Betweenness ≈ how often an airport sits on shortest connection paths — hubs whose failure disrupts the network."),
            html.Li("PageRank ≈ share of connectivity concentrated at an airport (weight = flight count)."),
            html.Li("Communities (Louvain) ≈ regional route clusters (e.g. West Coast, Southeast)."),
            html.Li("Bubble size = yearly traffic; bubble color = Louvain community."),
        ], className="insights text-secondary small"),
    ], className="px-3 pb-3"),
])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("US Air Traffic Delay Patterns", className="app-title mb-0"),
            html.Div("2015 flights · graph view of the national route network", className="subtitle"),
        ], width=12, lg=8),
        dbc.Col(html.Div(id="clock", className="text-secondary small text-lg-end"), width=12, lg=4),
    ], className="py-3"),
    dbc.Row([
        sidebar(),
        dbc.Col([
            dbc.Row([
                kpi_card("Total flights", "…"), kpi_card("% delayed", "…"),
                kpi_card("Avg arrival delay", "…"), kpi_card("Worst airport", "…"),
                kpi_card("Busiest hub", "…"),
            ], className="g-2 mb-3"),
            dbc.Tabs([tab_overview, tab_map, tab_time, tab_airlines, tab_graph],
                     id="tabs", active_tab="overview", className="mb-3"),
        ], width=12, lg=10),
    ]),
], fluid=True, style={"maxWidth": "1600px"})


@app.callback(
    Output("kpi-Total flights", "children"), Output("kpi-% delayed", "children"),
    Output("kpi-Avg arrival delay", "children"), Output("kpi-Worst airport", "children"),
    Output("kpi-Busiest hub", "children"), Output("insights-list", "children"),
    Output("fig-geo", "figure"), Output("fig-heat", "figure"),
    Output("fig-monthly", "figure"), Output("fig-airline", "figure"),
    Output("fig-causes-airline", "figure"), Output("fig-causes-month", "figure"),
    Output("fig-worst", "figure"),
    Output("fig-monthly-2", "figure"),
    Output("filter-status", "children"),
    Input("flt-airline", "value"), Input("flt-month", "value"),
    Input("flt-origin", "value"), Input("flt-cause", "value"), Input("flt-topn", "value"),
    prevent_initial_call=False,
)
def update_filters(airline_codes, month_range, origin, cause, top_n):
    top_n = top_n or 40
    out = filtered_outputs(airline_codes, month_range, origin, cause, int(top_n))
    status = f"Filters: {out[0]} flights"
    if origin:
        status += f" · origin {origin}"
    if airline_codes:
        status += f" · {len(airline_codes)} airline(s)"
    if cause:
        status += f" · cause {CAUSE_LABELS.get(cause, cause)}"
    return (*out, status)


@app.callback(Output("fig-network", "figure"), Input("net-color", "value"))
def update_network(color_by):
    return STATIC_FIGS["net_comm"] if color_by == "community" else STATIC_FIGS["net_delay"]


@app.callback(Output("fig-centrality", "figure"), Input("cent-metric", "value"))
def update_centrality(metric):
    return STATIC_FIGS[f"centr_{metric}"]


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
