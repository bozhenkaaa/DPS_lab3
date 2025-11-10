from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List

import numpy as np
from scipy.integrate import solve_ivp

import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash.dash_table import DataTable
import plotly.graph_objects as go

# --- Model for 2D system ---
@dataclass
class PhasePlaneModel:
    label: str
    dx: Callable[[float, float], float]
    dy: Callable[[float, float], float]
    jacob: Callable[[float, float], np.ndarray]
    fixed_pts: List[Tuple[float, float]]
    xrange: Tuple[float, float]
    yrange: Tuple[float, float]
    note: str

def classify_point(J: np.ndarray) -> str:
    eigs = np.linalg.eigvals(J)
    real, imag = np.real(eigs), np.imag(eigs)
    if np.allclose(imag, 0):
        s = np.sign(real)
        if s[0] * s[1] < 0:
            return "Saddle"
        if s[0] > 0 and s[1] > 0:
            return "Unstable node"
        if s[0] < 0 and s[1] < 0:
            return "Stable node"
        return "Degenerate"
    return "Center" if np.isclose(real[0], 0) else ("Unstable focus" if real[0] > 0 else "Stable focus")

# --- Example systems ---
modelA = PhasePlaneModel(
    label="System A: dx/dt = -y, dy/dt = x - 3x^2",
    dx=lambda x, y: -y,
    dy=lambda x, y: x - 3 * x**2,
    jacob=lambda x, y: np.array([[0.0, -1.0], [1.0 - 6.0 * x, 0.0]]),
    fixed_pts=[(0.0, 0.0), (1.0 / 3.0, 0.0)],
    xrange=(-1.5, 2.0),
    yrange=(-2.0, 2.0),
    note="(0,0): center; (1/3,0): saddle"
)

modelB = PhasePlaneModel(
    label="System B: dx/dt = -2xy, dy/dt = x^2 + y^2 - 1",
    dx=lambda x, y: -2 * x * y,
    dy=lambda x, y: x**2 + y**2 - 1.0,
    jacob=lambda x, y: np.array([[-2.0 * y, -2.0 * x], [2.0 * x, 2.0 * y]]),
    fixed_pts=[(1.0, 0.0), (-1.0, 0.0), (0.0,
1.0), (0.0, -1.0)],
    xrange=(-2.0, 2.0),
    yrange=(-2.0, 2.0),
    note="(±1,0), (0,±1): saddle/center"
)

MODELS: Dict[str, PhasePlaneModel] = {m.label: m for m in (modelA, modelB)}

# --- Nullclines (with new colors) ---
def find_nullclines(x, y, z) -> List[np.ndarray]:
    nx, ny = len(x), len(y)
    segments = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            z00, z10 = z[j, i], z[j, i+1]
            z01, z11 = z[j+1, i], z[j+1, i+1]
            mask = (z00 > 0) + 2*(z10 > 0) + 4*(z11 > 0) + 8*(z01 > 0)
            if mask in (0, 15):
                continue
            def interp(a, b, va, vb):
                t = va / (va - vb + 1e-16)
                return a + t * (b - a)
            pts = []
            if (z00 > 0) != (z10 > 0):
                pts.append((interp(x[i], x[i+1], z00, z10), y[j]))
            if (z10 > 0) != (z11 > 0):
                pts.append((x[i+1], interp(y[j], y[j+1], z10, z11)))
            if (z11 > 0) != (z01 > 0):
                pts.append((interp(x[i+1], x[i], z11, z01), y[j+1]))
            if (z01 > 0) != (z00 > 0):
                pts.append((x[i], interp(y[j+1], y[j], z01, z00)))
            if len(pts) >= 2:
                segments.append(np.array(pts[:2]))
    return segments

def make_nullclines(model: PhasePlaneModel, xlim, ylim, n=220):
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    X, Y = np.meshgrid(xs, ys)
    F = model.dx(X, Y)
    G = model.dy(X, Y)
    f_lines = find_nullclines(xs, ys, F)
    g_lines = find_nullclines(xs, ys, G)
    def mk_trace(lines, label, dash=None, color=None):
        if not lines:
            return go.Scattergl(x=[], y=[], mode="lines", name=label)
        xpts, ypts = [], []
        for seg in lines:
            xpts += [seg[0, 0], seg[1, 0], None]
            ypts += [seg[0, 1], seg[1, 1], None]
        return go.Scattergl(
            x=xpts, y=ypts, mode="lines",
            line=dict(width=2, dash=dash, color=color) if dash or color else dict(width=2),
            name=label, hoverinfo="skip"
        )
    return [
        mk_trace(f_lines, "dx/dt=0", color="#e74c3c"),
        mk_trace(g_lines, "dy/dt=0", dash="dot", color="#2980b9")
    ]

# --- Streamlines (orange) ---
def plot_streams(model: PhasePlaneModel, xlim, ylim, grid=14, t1=8.0, t2=8.0, extra: List[Tuple[float, float]] = None):
    x0 = np.linspace(xlim[0], xlim[1], grid)
    y0 = np.linspace(ylim[0], ylim[1], grid)
    seeds = [(xi, yi) for xi in x0 for yi in y0]
    if extra:
        seeds.extend(extra)
    def rhs(t, z):
        x, y = z
        return [model.dx(x, y), model.dy(x, y)]
    def in_bounds(t, z):
        x, y = z
        pad = 0.1 * max(xlim[1]-xlim[0], ylim[1]-ylim[0])
        return float((xlim[0]-pad <= x <= xlim[1]+pad) and (ylim[0]-pad <= y <= ylim[1]+pad))
    in_bounds.terminal = True
    in_bounds.direction = -1
    curves = []
    for (xi, yi) in seeds:
        if any(np.hypot(xi - ex, yi - ey) < 1e-2 for ex, ey in model.fixed_pts):
            continue
        sol_f = solve_ivp(rhs, (0, t1), [xi, yi], events=in_bounds, max_step=0.08, rtol=1e-5, atol=1e-8)
        sol_b = solve_ivp(rhs, (0, -t2), [xi, yi], events=in_bounds, max_step=0.08, rtol=1e-5, atol=1e-8)
        x = np.hstack([sol_b.y[0][::-1], sol_f.y[0]])
        y = np.hstack([sol_b.y[1][::-1], sol_f.y[1]])
        curves.append((x, y))
    return curves

# --- Vector field (green) ---
def plot_vector_field(model: PhasePlaneModel, xlim, ylim, density=18):
    xs = np.linspace(*xlim, density)
    ys = np.linspace(*ylim, density)
    XX, YY = np.meshgrid(xs, ys)
    U = model.dx(XX, YY)
    V = model.dy(XX, YY)
    S = np.sqrt(U**2 + V**2) + 1e-9
    U /= S
    V /= S
    scale = 0.07 * (xlim[1] - xlim[0])
    qx, qy = [], []
    for i in range(XX.size):
        x0, y0 = XX.ravel()[i], YY.ravel()[i]
        x1, y1 = x0 + scale * U.ravel()[i], y0 + scale * V.ravel()[i]
        qx += [x0, x1,
 None]
        qy += [y0, y1, None]
    return go.Scattergl(x=qx, y=qy, mode="lines", line=dict(width=1.2, color="#27ae60"), name="Vector field", hoverinfo="skip")

# --- Main plot builder ---
def make_figure(
    model: PhasePlaneModel,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    show_field=True, show_null=True, show_stream=True,
    density=18, grid=14, tspan=8.0,
    extra: List[Tuple[float, float]] = None,
    dark=False
) -> go.Figure:
    fig = go.Figure()
    if show_field:
        fig.add_trace(plot_vector_field(model, xlim, ylim, density=density))
    if show_null:
        for t in make_nullclines(model, xlim, ylim):
            fig.add_trace(t)
    if show_stream:
        for (x, y) in plot_streams(model, xlim, ylim, grid, tspan, tspan, extra):
            fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", line=dict(width=1, color="#f39c12"), showlegend=False, hoverinfo="skip"))
    for (ex, ey) in model.fixed_pts:
        typ = classify_point(model.jacob(ex, ey))
        fig.add_trace(go.Scattergl(x=[ex], y=[ey], mode="markers+text",
                                   marker=dict(size=12, symbol="diamond", color="#8e44ad"),
                                   text=[typ], textposition="top center",
                                   name=f"Fixed ({ex:.3g},{ey:.3g})"))
    bg = "#222831" if dark else "#f7f7f7"
    gridc = "#393e46" if dark else "#cccccc"
    fontc = "#eeeeee" if dark else "#222"
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(range=xlim, zeroline=True, gridcolor=gridc),
        yaxis=dict(range=ylim, zeroline=True, gridcolor=gridc, scaleanchor="x", scaleratio=1),
        plot_bgcolor=bg, paper_bgcolor=bg, font=dict(color=fontc),
        legend=dict(bgcolor="rgba(0,0,0,0)" if dark else "rgba(255,255,255,0.7)", borderwidth=0),
    )
    return fig

def eig_table(model: PhasePlaneModel):
    rows = []
    for (ex, ey) in model.fixed_pts:
        J = model.jacob(ex, ey)
        vals = np.linalg.eigvals(J)
        rows.append({"x*": f"{ex:.6g}", "y*": f"{ey:.6g}", "λ1": f"{vals[0]:.6g}", "λ2": f"{vals[1]:.6g}",
                     "Type": classify_point(J)})
    return rows

external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/lux/bootstrap.min.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Phase Portraits Playground"
server = app.server

model_names = list(MODELS.keys())

def card(title, content):
    return html.Div(className="card shadow mb-3", children=[
        html.Div(className="card-header fw-bold", children=title),
        html.Div(className="card-body", children=content),
    ])

app.layout = html.Div(className="container-xl py-3", children=[
    html.Div(className="d-flex align-items-center justify-content-between mb-3", children=[
        html.H3("Phase Portraits Interactive", className="m-0"),
        html.Div(children=[
            html.Label("Choose system:", className="me-2"),
            dcc.Dropdown(options=[{"label": n, "value": n} for n in model_names],
                         value=model_names[0], id="sys", clearable=False, style={"minWidth": 360})
        ])
    ]),
    html.Div(className="row g-3", children=[
        html.Div(className="col-12 col-xl-8", children=[
            card("Plot", [
                html.Div(className="d-flex gap-2 flex-wrap mb-2", children=[
                    html.Button("Save as PNG", id="btn-export", className="btn btn-outline-primary btn-sm"),
                    dcc.Download(id="download-fig"),
                    html.Div(className="form-check form-switch", children=[
                        dcc.Checklist(id="theme-dark", options=[{"label": "Dark mode", "value": "dark"}],
                                      value=[], inputClassName="form-check-input", labelClassName="form-check-label")
                    ]),
                ]),
                dcc.Graph(id="fig", clear_on_unhover=True, config={"displayModeBar": True}),
                html.Div(className="text-muted small", children="Click on the plot to add a custom initial point.")
            ])
        ]),
        html.Div(className="col-12 col-xl-4", children=[
            card("Domain & Settings", [
                html.Div(className="row g-2", children=[
                    html.Div(className="col-6", children=[html.Label("x min"),
                                                          dcc.Input(id="x-min", type="number",
                                                                    value=MODELS[model_names[0]].xrange[0],
                                                                    className="form-control")]),
                    html.Div(className="col-6", children=[html.Label("x max"),
                                                          dcc.Input(id="x-max", type="number",
                                                                    value=MODELS[model_names[0]].xrange[1],
                                                                    className="form-control")]),
                    html.Div(className="col-6", children=[html.Label("y min"),
                                                          dcc.Input(id="y-min", type="number",
                                                                    value=MODELS[model_names[0]].yrange[0],
                                                                    className="form-control")]),
                    html.Div(className="col-6", children=[html.Label("y max"),
                                                          dcc.Input(id="y-max", type="number",
                                                                    value=MODELS[model_names[0]].yrange[1],

                                                                    className="form-control")]),
                ]),
                html.Hr(),
                html.Label("Vector field density"),
                dcc.Slider(10, 35, 1, value=18, id="density"),
                html.Label("Seed grid size"),
                dcc.Slider(6, 24, 1, value=14, id="seeds-grid"),
                html.Label("Integration time"),
                dcc.Slider(2, 20, 0.5, value=8.0, id="tspan"),
                html.Hr(),
                html.Label("Layers"),
                dcc.Checklist(
                    id="layers",
                    options=[{"label": "Vector field", "value": "vec"},
                             {"label": "Nullclines", "value": "null"},
                             {"label": "Streamlines", "value": "stream"}],
                    value=["vec", "null", "stream"],
                    inputClassName="form-check-input me-2",
                    labelClassName="form-check-label d-block"
                ),
                html.Hr(),
                html.Label("Fixed point for linearization"),
                dcc.Dropdown(id="eq-choose", clearable=True, placeholder="Select fixed point"),
                html.Div(id="lin-info", className="mt-2 small")
            ]),
            card("Custom initial points", [
                dcc.Store(id="seeds-store", data=[]),
                html.Div(id="seed-list", className="mb-2"),
                html.Div(className="d-flex gap-2", children=[
                    html.Button("Clear points", id="btn-clear-seeds", className="btn btn-sm btn-outline-danger"),
                ])
            ]),
            card("Eigenvalues at fixed points", [
                DataTable(
                    id="eig-table",
                    columns=[{"name": c, "id": c} for c in ["x*", "y*", "λ1", "λ2", "Type"]],
                    data=[], style_table={"overflowX": "auto"},
                    style_cell={"padding": "6px", "fontFamily": "monospace"},
                    style_header={"fontWeight": "600"}
                )
            ]),
        ])
    ])
])

@app.callback(
    Output("eq-choose", "options"),
    Output("x-min", "value"),
    Output("x-max", "value"),
    Output("y-min", "value"),
    Output("y-max", "value"),
    Output("eig-table", "data"),
    Input("sys", "value")
)
def on_model_change(model_label):
    m = MODELS[model_label]
    eq_opts = [{"label": f"({ex:.3g}, {ey:.3g})", "value": f"{ex},{ey}"} for (ex, ey) in m.fixed_pts]
    return eq_opts, m.xrange[0], m.xrange[1], m.yrange[0], m.yrange[1], eig_table(m)

@app.callback(
    Output("seeds-store", "data", allow_duplicate=True),
    Output("seed-list", "children"),
    Input("fig", "clickData"),
    Input("btn-clear-seeds", "n_clicks"),
    State("seeds-store", "data"),
    prevent_initial_call=True
)
def handle_seed_click(click, clear_clicks, seeds):
    triggered = [t["prop_id"] for t in callback_context.triggered]
    seeds = list(seeds or [])
    if "btn-clear-seeds.n_clicks" in triggered:
        seeds = []
    elif click and "points" in click and click["points"]:
        p = click["points"][0]
        seeds.append((float(p["x"]), float(p["y"])))
    items = [html.Code(f"({x:.3g}, {y:.3g})") for x, y in seeds] or [html.Span("— none —", className="text-muted")]
    return seeds, html.Div(["Current points: "] + items)

@app.callback(
    Output("fig", "figure"),
    Output("lin-info", "children"),
    Input("sys", "value"),
    Input("x-min", "value"), Input("x-max", "value"),
    Input("y-min", "value"), Input("y-max", "value"),
    Input("layers", "value"),
    Input("density", "value"),
    Input("seeds-grid", "value"),
    Input("tspan", "value"),
    Input("eq-choose", "value"),
    Input("seeds-store", "data"),
    Input("theme-dark", "value"),
)
def update_plot(model_label, xmin, xmax, ymin, ymax, layers,
                density, grid, tspan, eq_value, seeds, theme_val):
    m = MODELS[model_label]
    xlim = (xmin, xmax)
    ylim = (ymin, ymax)
    fig = make_figure(
        m, xlim, ylim,
        show_field=("vec" in layers),
        show_null=("null" in layers),
        show_stream=("stream" in layers),
        density=density, grid=grid, tspan=tspan,
        extra=seeds or [],
        dark=("dark" in (theme_val or []))
    )
    fig.update_layout(title=model_label)
    if not eq_value:
        return fig, ""
    ex, ey = map(float, eq_value.split(","))
    J = m.jacob(ex, ey)
    vals = np.linalg.eigvals(J)
    typ = classify_point(J)
    info = html.Div([
        html.Div(f"Fixed point: ({ex:.6g}, {ey:.6g})"),
        html.Div(f"Jacobian: [[{J[0,0]:.6g}, {J[0,1]:.6g}], [{J[1,0]:.6g}, {J[1,1]:.6g}]]"),
        html.Div(f"λ1 = {vals[0]:.6g},  λ2 = {vals[1]:.6g}  →  {typ}"),
    ])
    return fig, info

@app.callback(
    Output("download-fig", "data"),
    Input("btn-export", "n_clicks"),
    State("fig", "figure"),
    prevent_initial_call=True
)
def export_png(n, fig_dict):
    fig = go.Figure(fig_dict)
    return dcc.send_bytes(fig.to_image(format="png", scale=2), "phase_portrait.png")

if __name__ == "__main__":
    app.run(debug=True)