"""Plotly Dash front-end for the hotel cancellation API.

Form mirrors the FastAPI input schema. A predict button POSTs to the API and
renders a gauge + label. Two preset buttons load known cancel/stay rows so the
demo doesn't require filling 28 fields by hand.
"""
from __future__ import annotations

import os
from typing import Any

import dash
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, dcc, html, no_update

from hotels.ui.samples import LIKELY_CANCEL, LIKELY_STAY

API_BASE = os.environ.get("HOTELS_API_BASE", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.environ.get("HOTELS_API_TIMEOUT", "10"))

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MEALS = ["BB", "FB", "HB", "SC", "Undefined"]
MARKET_SEGMENTS = [
    "Direct", "Corporate", "Online TA", "Offline TA/TO",
    "Complementary", "Groups", "Undefined", "Aviation",
]
DISTRIBUTION_CHANNELS = ["Direct", "Corporate", "TA/TO", "Undefined", "GDS"]
CUSTOMER_TYPES = ["Transient", "Contract", "Transient-Party", "Group"]
ROOM_TYPES = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
COMMON_COUNTRIES = [
    "PRT", "GBR", "FRA", "ESP", "DEU", "IRL", "ITA", "BEL", "NLD", "USA",
    "BRA", "CHE", "AUT", "POL", "CHN", "RUS", "ROU", "NOR", "SWE", "Unknown",
]


def _opts(values: list[str]) -> list[dict[str, str]]:
    return [{"label": v, "value": v} for v in values]


# Form fields are declared in id/component pairs so the callback can rebuild
# the payload from State without listing every id by hand.
FIELD_IDS = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "meal", "country",
    "market_segment", "distribution_channel", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "reserved_room_type", "assigned_room_type", "booking_changes",
    "deposit_type", "agent", "company", "days_in_waiting_list",
    "customer_type", "adr", "required_car_parking_spaces",
    "total_of_special_requests",
]


def number_input(id_: str, value: float | int | None, **kwargs: Any) -> dcc.Input:
    return dcc.Input(id=id_, type="number", value=value, debounce=True, **kwargs)


def dropdown(id_: str, options: list[str], value: str | None, **kwargs: Any) -> dcc.Dropdown:
    return dcc.Dropdown(id=id_, options=_opts(options), value=value, clearable=False, **kwargs)


def card(title: str, children: list) -> html.Div:
    return html.Div(
        children=[html.H4(title, className="card-title"), *children],
        className="card",
    )


def labeled(label: str, component) -> html.Div:
    return html.Div([html.Label(label), component], className="field")


def build_form(initial: dict) -> html.Div:
    return html.Div(
        [
            card("Booking", [
                labeled("Hotel", dropdown("hotel", ["City Hotel", "Resort Hotel"], initial["hotel"])),
                labeled("Deposit type", dropdown("deposit_type",
                    ["No Deposit", "Non Refund", "Refundable"], initial["deposit_type"])),
                labeled("Lead time (days)", number_input("lead_time", initial["lead_time"], min=0, max=800, step=1)),
                labeled("Average daily rate (adr)", number_input("adr", initial["adr"], min=0, step=1)),
                labeled("Customer type", dropdown("customer_type", CUSTOMER_TYPES, initial["customer_type"])),
            ]),
            card("Arrival date", [
                labeled("Year", number_input("arrival_date_year", initial["arrival_date_year"], min=2014, max=2030, step=1)),
                labeled("Month", dropdown("arrival_date_month", MONTHS, initial["arrival_date_month"])),
                labeled("Week of year", number_input("arrival_date_week_number", initial["arrival_date_week_number"], min=1, max=53, step=1)),
                labeled("Day of month", number_input("arrival_date_day_of_month", initial["arrival_date_day_of_month"], min=1, max=31, step=1)),
            ]),
            card("Stay & guests", [
                labeled("Weekend nights", number_input("stays_in_weekend_nights", initial["stays_in_weekend_nights"], min=0, max=20, step=1)),
                labeled("Week nights", number_input("stays_in_week_nights", initial["stays_in_week_nights"], min=0, max=40, step=1)),
                labeled("Adults", number_input("adults", initial["adults"], min=0, max=10, step=1)),
                labeled("Children", number_input("children", initial["children"], min=0, max=10, step=1)),
                labeled("Babies", number_input("babies", initial["babies"], min=0, max=10, step=1)),
                labeled("Meal", dropdown("meal", MEALS, initial["meal"])),
            ]),
            card("Origin & channel", [
                labeled("Country", dropdown("country", COMMON_COUNTRIES, initial["country"] or "Unknown")),
                labeled("Market segment", dropdown("market_segment", MARKET_SEGMENTS, initial["market_segment"])),
                labeled("Distribution channel", dropdown("distribution_channel", DISTRIBUTION_CHANNELS, initial["distribution_channel"])),
                labeled("Agent ID (optional)", number_input("agent", initial["agent"], min=0, step=1)),
                labeled("Company ID (optional)", number_input("company", initial["company"], min=0, step=1)),
            ]),
            card("History", [
                labeled("Is repeated guest?", dropdown("is_repeated_guest", ["0", "1"], str(initial["is_repeated_guest"]))),
                labeled("Previous cancellations", number_input("previous_cancellations", initial["previous_cancellations"], min=0, step=1)),
                labeled("Previous successful stays", number_input("previous_bookings_not_canceled", initial["previous_bookings_not_canceled"], min=0, step=1)),
                labeled("Booking changes", number_input("booking_changes", initial["booking_changes"], min=0, step=1)),
                labeled("Days in waiting list", number_input("days_in_waiting_list", initial["days_in_waiting_list"], min=0, step=1)),
            ]),
            card("Room & extras", [
                labeled("Reserved room type", dropdown("reserved_room_type", ROOM_TYPES, initial["reserved_room_type"])),
                labeled("Assigned room type", dropdown("assigned_room_type", ROOM_TYPES, initial["assigned_room_type"])),
                labeled("Parking spaces", number_input("required_car_parking_spaces", initial["required_car_parking_spaces"], min=0, max=5, step=1)),
                labeled("Special requests", number_input("total_of_special_requests", initial["total_of_special_requests"], min=0, max=10, step=1)),
            ]),
        ],
        className="form-grid",
    )


def make_gauge(prob: float, label: int, threshold: float) -> go.Figure:
    color = "#d9534f" if label == 1 else "#5cb85c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        number={"valueformat": ".2%"},
        delta={"reference": threshold, "valueformat": ".2%"},
        title={"text": "Probability of cancellation"},
        gauge={
            "axis": {"range": [0, 1], "tickformat": ".0%"},
            "bar": {"color": color},
            "steps": [
                {"range": [0, threshold], "color": "#eef5ee"},
                {"range": [threshold, 1], "color": "#fbeeee"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.8,
                "value": threshold,
            },
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def health_badge(payload: dict | None) -> html.Span:
    if not payload or not payload.get("model_loaded"):
        return html.Span("model not loaded", className="badge badge-error")
    return html.Span(
        f"model v{payload.get('model_version')} ({payload.get('model_kind')})",
        className="badge badge-ok",
    )


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0; background: #f7f7f9; color: #222; }
.header { padding: 18px 28px; background: white; border-bottom: 1px solid #e3e3e8;
          display: flex; align-items: center; gap: 14px; }
.header h2 { margin: 0; font-weight: 600; }
.badge { padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; }
.badge-ok { background: #e4f5e4; color: #2c662d; }
.badge-error { background: #f9e1e1; color: #842029; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
@media (max-width: 900px) { .form-grid { grid-template-columns: 1fr; } }
.card { background: white; border: 1px solid #e3e3e8; border-radius: 10px;
        padding: 14px 16px; }
.card-title { margin: 0 0 10px 0; font-size: 14px; text-transform: uppercase;
              letter-spacing: 0.04em; color: #666; }
.field { margin-bottom: 10px; }
.field label { display: block; font-size: 12px; color: #555; margin-bottom: 3px; }
.field input, .field .Select-control { width: 100%; }
.actions { display: flex; gap: 10px; margin: 18px 0; align-items: center; flex-wrap: wrap; }
button { padding: 8px 16px; border-radius: 6px; border: 1px solid #ccc; background: white;
         cursor: pointer; font-size: 14px; }
button.primary { background: #2c5dc6; color: white; border-color: #2c5dc6; }
button.primary:hover { background: #244ea6; }
.result { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 900px) { .result { grid-template-columns: 1fr; } }
.error { color: #842029; background: #f9e1e1; padding: 10px; border-radius: 6px; }
.label-display { font-size: 24px; font-weight: 600; }
.label-cancel { color: #d9534f; }
.label-stay { color: #2c662d; }
"""


def serve_layout() -> html.Div:
    return html.Div([
        html.Div(
            className="header",
            children=[
                html.H2("Hotel cancellation predictor"),
                html.Div(id="health-badge"),
                dcc.Interval(id="health-tick", interval=15_000, n_intervals=0),
            ],
        ),
        html.Div(className="container", children=[
            html.Div(className="actions", children=[
                html.Button("Predict", id="predict-btn", className="primary", n_clicks=0),
                html.Button("Load cancel sample", id="load-cancel", n_clicks=0),
                html.Button("Load stay sample", id="load-stay", n_clicks=0),
                html.Span(id="api-status", style={"marginLeft": "auto", "color": "#666", "fontSize": "12px"}),
            ]),
            html.Div(id="result", className="result", children=[
                html.Div(id="result-text"),
                dcc.Graph(id="result-gauge", config={"displayModeBar": False}),
            ]),
            html.Hr(style={"margin": "24px 0", "border": "0", "borderTop": "1px solid #e3e3e8"}),
            build_form(LIKELY_STAY),
        ]),
    ])


app = dash.Dash(
    __name__,
    title="Hotel cancellation predictor",
    update_title=None,
)
app.index_string = (
    "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>"
    "{%favicon%}{%css%}<style>" + CSS + "</style></head><body>"
    "{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"
)
app.layout = serve_layout


@app.callback(
    Output("health-badge", "children"),
    Input("health-tick", "n_intervals"),
)
def refresh_health(_n):
    try:
        r = requests.get(f"{API_BASE}/healthz", timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return health_badge(r.json())
    except requests.RequestException:
        pass
    return health_badge(None)


@app.callback(
    [Output(f, "value") for f in FIELD_IDS],
    [Input("load-cancel", "n_clicks"), Input("load-stay", "n_clicks")],
    prevent_initial_call=True,
)
def load_sample(_c1, _c2):
    trigger = dash.ctx.triggered_id
    sample = LIKELY_CANCEL if trigger == "load-cancel" else LIKELY_STAY
    return [
        str(sample[f]) if f == "is_repeated_guest" else sample[f]
        for f in FIELD_IDS
    ]


@app.callback(
    [Output("result-text", "children"),
     Output("result-gauge", "figure"),
     Output("api-status", "children")],
    Input("predict-btn", "n_clicks"),
    [State(f, "value") for f in FIELD_IDS],
    prevent_initial_call=True,
)
def predict(_n, *values):
    payload = dict(zip(FIELD_IDS, values, strict=True))
    payload["is_repeated_guest"] = int(payload.get("is_repeated_guest") or 0)

    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return html.Div(f"Request failed: {exc}", className="error"), no_update, ""

    if r.status_code != 200:
        return html.Div(f"API {r.status_code}: {r.text}", className="error"), no_update, ""

    body = r.json()
    prob = body["probability_cancelled"]
    label = body["label"]
    threshold = body["threshold"]

    label_text = "Predicted: CANCEL" if label == 1 else "Predicted: STAY"
    label_class = "label-display " + ("label-cancel" if label == 1 else "label-stay")
    text = html.Div([
        html.Div(label_text, className=label_class),
        html.Div(f"Probability of cancellation: {prob:.2%}",
                 style={"marginTop": "8px", "fontSize": "16px"}),
        html.Div(f"Decision threshold: {threshold:.2f}",
                 style={"color": "#777", "fontSize": "13px"}),
    ])
    status = f"OK ({API_BASE})"
    return text, make_gauge(prob, label, threshold), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
