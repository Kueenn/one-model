import base64
import io

import numpy as np
import pandas as pd
import cvxpy as cp

import dash
from dash import (
    Dash, 
    dcc, 
    html, 
    dash_table, 
    Input, 
    Output, 
    State,
)

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

matplotlib.use("Agg")


initial_data = {
    "DMU": [
        1,
        2,
        3, 
        4, 
        5, 
        6, 
        7, 
        8, 
        9, 
        10, 
        11, 
        12,
    ],

    "University": [
        "Carleton U.", 
        "McGill U.", 
        "McMaster U.", 
        "Queen’s U.", 
        "Simon Fraser U.",
        "U. of Alberta", 
        "U. of British Columbia", 
        "U. of New Brunswick", 
        "U. of Toronto", 
        "U. of Victoria", 
        "U. of Waterloo", 
        "U. of Western Ontario",
    ],

    "Faculty": [
        30, 
        66, 
        51, 
        56, 
        47, 
        71, 
        76, 
        32, 
        85, 
        23, 
        38, 
        70,
    ],

    "Citation": [
        227.85, 
        771.13, 
        897.37, 
        660.87, 
        251.4, 
        682.53, 
        1860.12, 
        158.83, 
        1341.55, 
        98.5, 
        207.83, 
        978.02,
    ],

    "Paper": [
        41.95, 
        112.26, 
        201.05, 
        90.52, 
        78.03, 
        133.52, 
        286.78, 
        64.83, 
        208.69, 
        32.92, 
        63.33, 
        88.05,
    ],
}

initial_df = pd.DataFrame(initial_data)

DARKBLUE = "#1c2b4d"
WHITE = "#ffffff"
GREY = "#dbdada"
GREEN_HIGHLIGHT = "#e0ffe0"
YELLOW_HIGHLIGHT = "#fdffe0"
RED_HIGHLIGHT = "#ffe0e0"

FONT = "Georgia"
ANOTHER_FONT = "Courier New"

table_style = {
                "overflowX": "auto",
                "borderRadius": "30px",
                "border": "1px solid #ccc",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.4)",
            }

app = Dash(
    __name__,
    assets_folder="assets",
    external_stylesheets=[],
)

app.title = "One-Model DEA Input Congestion Dashboard"

app.layout = html.Div([
    html.H1("One-Model DEA Input Congestion Dashboard", 
            style={
                "textAlign": "center",
                "fontFamily": FONT,
            },
        ),

    html.Div([
        html.H3("Input Data Table",
                style={
                    "fontWeight": "bold",
                    "fontSize": "24px",
                    "fontFamily": FONT,
                },
            ),
        dash_table.DataTable(
            id="input-table",
            data=initial_df.to_dict("records"),
            columns=[
                {
                    "name": col, 
                    "id": col,
                    "editable": (col != "DMU"),
                } 
                for col in initial_df.columns
            ],
            style_header={
                "backgroundColor": DARKBLUE,
                "fontColor": WHITE,
                "color": WHITE,
                "fontWeight": "bold",
            },
            style_cell={
                "textAlign": "center",
                "fontWeight": "bold",
                "fontFamily": ANOTHER_FONT,
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'University'},
                    'width': '250px'
                },
            ],
            editable=True,
            row_deletable=False,
            style_table=table_style,
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": GREY,
                },
                {
                    "if": {"column_editable": False},
                    "backgroundColor": DARKBLUE,
                    "color": WHITE,
                    "fontWeight": "bold",
                },
            ],
            page_size=12,
        ),
        html.Div([
            html.Div([
                html.Button(
                    "🔁 Update DEA", 
                    id="run-dea", 
                    n_clicks=0, 
                    className="custom-button",
                )
            ], 
            style={
                "display": "inline-block", 
                "marginRight": "10px"
            },
            ),

            html.Div([
                dcc.Upload(
                    id="upload-excel",
                    children=html.Button(
                        "📤 Import Excel", 
                        className="custom-button",
                    ),
                    accept=".xlsx",
                    multiple=False,
                )
            ], 
            style={
                "display": "inline-block", 
                "marginRight": "10px"
            },
            ),

            html.Div([
                html.Button(
                    "📥 Export Excel", 
                    id="download-input-btn", 
                    className="custom-button",
                ),
                dcc.Download(id="excel-input-download")
            ], 
            style={
                "display": "inline-block"
            },
            )
        ], 
        style={
            "marginTop": "20px", 
            "marginBottom": "20px", 
            "marginLeft": "20px"
        },
    ),

    ],
    style={
        "marginBottom": "30px",
        }
    ),

    html.Hr(),
    html.Div([
        dcc.Dropdown(
            options=[
                {"label": u, "value": u}
                for u in initial_df["University"]
            ],
            id="university-dropdown",
            multi=True,
            value=initial_df["University"].tolist(),
            placeholder="Select universities...",
            style={
                "width": "100%",
                "fontFamily": ANOTHER_FONT,
                "fontWeight": "bold",
                "borderRadius": "5px",
                "border": "1px solid #ccc",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
            },
        )
        ], 
        style={
            "display": "flex",
            "justifyContent": "center",
            "maxWidth": "1200px",
            "width": "100%",
            "margin": "0 auto",
            "marginTop": "20px",
            "marginBottom": "30px",
        }
    ),

    dcc.Checklist(
        options=[
            {
                "label": "ٍEfficiency Score", 
                "value": "Efficiency Score",
            },
            {
                "label": "Congestion", 
                "value": "Congestion",
            },
            {
                "label": "Extra Faculty Needed", 
                "value": "Extra Faculty Needed",
            },
            {
                "label": "Citation Slack", 
                "value": "Citation Slack",
            },
            {
                "label": "Paper Slack", 
                "value": "Paper Slack",
            },
        ],
        value=[
            "Efficiency Score", 
            "Congestion", 
            "Extra Faculty Needed", 
            "Citation Slack",
        ],
        id="metric-checklist",
        inline=True,
        style={
                "textAlign": "center",
        },
        labelStyle={
                "display": "inline-block",
                "borderRadius": "40px",
                "border": "1px solid #ccc",
                "padding": "10px",
                "marginRight": "20px",
                "backgroundColor": GREY,
                "fontFamily": FONT,
        },
    ),

    html.Div(id="metric-plots"),

    html.Hr(),
    dcc.Tabs([
        dcc.Tab(
            label="Parallel Coordinates Plot", 
            children=[
                html.Div([
                    dcc.Graph(id="parallel-coord")
                ],
                style={
                    "marginTop": "40px",
                }
                ),
            ],
        ),

        dcc.Tab(label="Correlation Heatmap", 
            children=[
                html.Div([
                    html.Img(
                        id="corr-heatmap", 
                        style={
                            "maxWidth": "1200px", 
                            "width": "100%",
                        }
                    ),
                    ], 
                    style={
                        "textAlign": "center", 
                        "padding": "20px",
                    },
                ),
            ],
        ),
    ], 
    style={
            "paddingTop": "20px",
            "fontFamily": FONT,
        }
    ),

    html.H3(
        "Results Table",
        style={
            "fontWeight": "bold",
            "fontSize": "24px",
            "fontFamily": FONT,
        }
    ),

    dcc.RadioItems(
        options=[
            {
                "label": "DMU", 
                "value": "DMU",
            },
            {
                "label": "Efficiency Score", 
                "value": "Efficiency Score",
            },
            {
                "label": "Congestion", 
                "value": "Congestion",
            },
            {
                "label": "Extra Faculty Needed", 
                "value": "Extra Faculty Needed",
            },
            {
                "label": "Citation Slack", 
                "value": "Citation Slack",
            },
            {
                "label": "Paper Slack", 
                "value": "Paper Slack",
            },
        ],
        value="Efficiency Score",
        id="order-option",
        inline=True,
        style={
            "paddingBottom": "20px",
            "textAlign": "center",
        },
        labelStyle={
            "display": "inline-block",
            "paddingRight": "20px",
            "fontFamily": FONT,
            "borderRadius": "40px",
            "border": "1px solid #ccc",
            "padding": "10px",
            "marginRight": "20px",
            "backgroundColor": GREY,
        },
    ),

    dash_table.DataTable(
        id="results-table",
        columns=[],
        data=[],
        style_table={
            **table_style,
            "marginBottom": "10px",
        },
        page_size=12,
        style_header={
            "backgroundColor": DARKBLUE,
            "fontColor": WHITE,
            "color": WHITE,
            "fontWeight": "bold",
        },
        style_cell={
            "textAlign": "center",
            "fontWeight": "bold",
            "fontFamily": ANOTHER_FONT,
        },
    ),

    html.Div([
        html.Button(
            "📥 Export Excel", 
            id="download-excel-btn", 
            style={
                "marginBottom": "100px", 
                "marginLeft": "20px", 
            },
            className="custom-button",
        ),
        dcc.Download(id="excel-download"),
    ], 
    style={
        "marginTop": "20px",
    },
    ),

    dcc.Store(id="dea-data"),
],

style={
        "paddingTop": "50px",
        "paddingLeft": "200px",
        "paddingRight": "200px",
},
)

@app.callback(
    Output("results-table", "columns"),
    Output("results-table", "data"),
    Output("dea-data", "data"),
    Input("run-dea", "n_clicks"),
    Input("order-option", "value"),
    State("input-table", "data"),
)
def run_dea(_, order, table_data):
    df = pd.DataFrame(table_data)
    X = df[["Faculty"]].T.values
    Y = df[["Citation", "Paper"]].T.values

    results = []
    for i in range(X.shape[1]):
        res = dea_one_model(X, Y, i)
        results.append(
            {
                "University": df.loc[i, "University"],
                "phi": res["phi"],
                "Congestion": res["s_c"][0],
                "Extra Faculty Needed": res["s_plus_i2"][0],
                "Citation Slack": res["s_plus_r"][0],
                "Paper Slack": res["s_plus_r"][1],
            }
        )

    result_df = pd.DataFrame(results)

    result_df["phi"] = (
        result_df["phi"]
            .astype(np.float64)
    )

    result_df["Congestion"] = (
        result_df["Congestion"]
            .astype(np.int32)
    )

    result_df["Extra Faculty Needed"] = (
        result_df["Extra Faculty Needed"]
            .astype(np.int32)
    )

    result_df["Citation Slack"] = (
        result_df["Citation Slack"]
            .astype(np.float64)
            .round(2)
    )

    result_df["Paper Slack"] = (
        result_df["Paper Slack"]
            .astype(np.float64)
            .round(2)
    )

    result_df["phi"] = (
        (result_df["phi"].max() - result_df["phi"] + 1)
        .round(2)
    )

    result_df.rename(
        columns={
            "phi": "Efficiency Score"
        }, 
        inplace=True,
    )

    result_df.index += 1

    result_df.index.name = "DMU"

    result_df.reset_index(inplace=True)

    result_df.sort_values(
        by=order, 
        ascending=(order == "DMU"), 
        inplace=True
    )

    columns = [
        {
            "name": col, 
            "id": col,
        } 
        for col in result_df.columns
    ]

    return (
        columns, 
        result_df.to_dict("records"),
        result_df.to_dict("records"),
    )


@app.callback(
    Output("input-table", "data"),
    Output("input-table", "columns"),
    Input("upload-excel", "contents"),
    State("upload-excel", "filename"),
    prevent_initial_call=True
)
def update_table_from_excel(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if filename.endswith(".xlsx"):
            df_uploaded = pd.read_excel(io.BytesIO(decoded))
        else:
            return dash.no_update, dash.no_update
        
    except Exception as e:
        print("Error reading file:", e)
        return dash.no_update, dash.no_update

    return (
        df_uploaded.to_dict("records"), 
        [
            {
                "name": col, 
                "id": col, 
                "editable": col != "DMU"
            } 
            for col in df_uploaded.columns
        ]
    )


@app.callback(
    Output("university-dropdown", "options"),
    Output("university-dropdown", "value"),
    Input("run-dea", "n_clicks"),
    State("input-table", "data"),
)
def update_uni_dropdown(_, table_data):
    df = pd.DataFrame(table_data)

    return (
        [
            {"label": u, "value": u}
            for u in df["University"]
        ], 
        df["University"].to_list()
    )


@app.callback(
    Output("results-table", "style_data_conditional"),
    Input("results-table", "data"),
)
def highlight_results_rows(data):
    good_score = int(pd.DataFrame(data)["Efficiency Score"].max())

    green_query, yellow_query, red_query = (
        f"{{Efficiency Score}} >= {good_score}", 
        f"{{Efficiency Score}} >= {good_score - 1} && {{Efficiency Score}} < {good_score}", 
        f"{{Efficiency Score}} < {good_score - 1}",
    )

    return [
            {
                "if": {"filter_query": green_query}, 
                "backgroundColor": GREEN_HIGHLIGHT,
            },
            {
                "if": {"filter_query": yellow_query},
                "backgroundColor": YELLOW_HIGHLIGHT,
            },
            {
                "if": {"filter_query": red_query}, 
                "backgroundColor": RED_HIGHLIGHT,
            },
    ]


@app.callback(
    Output("excel-input-download", "data"),
    Input("download-input-btn", "n_clicks"),
    State("input-table", "data"),
    prevent_initial_call=True
)
def download_excel_input(_, data):
    return dcc.send_data_frame(
        pd.DataFrame(data).to_excel, 
        "input_data.xlsx",
        index=False,
    )


@app.callback(
    Output("metric-plots", "children"),
    Input("university-dropdown", "value"),
    Input("metric-checklist", "value"),
    Input("dea-data", "data"),
)
def update_metric_charts(selected_uni, selected_metrics, data):
    if not (data and selected_uni and selected_metrics):
        return []

    df = pd.DataFrame(data)

    df.sort_values(
        by="Efficiency Score", 
        ascending=False, 
        inplace=True
    )

    df = df[df["University"].isin(selected_uni)]

    plots = []
    for metric in selected_metrics:
        fig = px.bar(
            df,
            x="University",
            y=metric,
            color="University",
            color_discrete_sequence=px.colors.sequential.Turbo_r,
        )

        plots.append(
            html.Div(
                children=dcc.Graph(figure=fig),
                style={
                    "width": "48%",
                    "display": "inline-block",
                    "verticalAlign": "top",
                    "padding": "1%"
                }
            )
        )

    return plots


@app.callback(
    Output("parallel-coord", "figure"),
    Input("dea-data", "data")
)
def update_parallel_coord(data):
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)

    metrics = [
        "Efficiency Score", 
        "Congestion", 
        "Extra Faculty Needed", 
        "Citation Slack", 
        "Paper Slack",
    ]

    df_scaled = df.copy()

    df_scaled[metrics] = MinMaxScaler().fit_transform(df[metrics])

    fig = px.parallel_coordinates(
        df_scaled,
        dimensions=metrics,
        color="Efficiency Score",
        labels={m: m for m in metrics},
        color_continuous_scale=px.colors.sequential.Turbo,
    )

    return fig


@app.callback(
    Output("corr-heatmap", "src"),
    Input("dea-data", "data")
)
def update_heatmap_image(data):
    if not data:
        return None
    
    df = pd.DataFrame(data)

    metrics = [
        "Efficiency Score", 
        "Congestion", 
        "Extra Faculty Needed", 
        "Citation Slack", 
        "Paper Slack"
    ]
    
    corr = df[metrics].corr()

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        corr, 
        annot=True, 
        cmap="coolwarm", 
        center=0, 
        ax=ax
    )

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.title("Correlation Matrix")

    buf = io.BytesIO()
    plt.tight_layout()

    fig.savefig(buf, format="png")
    plt.close(fig)

    encoded = (
        base64.b64encode(buf.getbuffer())
            .decode("utf-8")
    )

    return f"data:image/png;base64,{encoded}"


@app.callback(
    Output("excel-download", "data"),
    Input("download-excel-btn", "n_clicks"),
    State("dea-data", "data"),
    prevent_initial_call=True,
)
def download_excel(_, data):
    return dcc.send_data_frame(
        pd.DataFrame(data).to_excel, 
        "dea_results.xlsx",
        index=False,
    )


def dea_one_model(X, Y, dmu_index, epsilon=1e-4):
    m, n = X.shape
    s = Y.shape[0]
    
    x0 = X[:, dmu_index]
    y0 = Y[:, dmu_index]

    phi = cp.Variable()
    lambdas = cp.Variable(n)
    s_c = cp.Variable(m, nonneg=True)
    s_plus_i2 = cp.Variable(m, nonneg=True)
    s_plus_r = cp.Variable(s, nonneg=True)

    constraints = []
    for i in range(m):
        constraints.append(x0[i] == X[i, :] @ lambdas + s_c[i] - s_plus_i2[i])

    for r in range(s):
        constraints.append(0 == Y[r, :] @ lambdas - phi * y0[r] - s_plus_r[r])

    constraints.append(cp.sum(lambdas) == 1)
    constraints.append(lambdas >= 0)

    obj = cp.Maximize(phi + epsilon * (-cp.sum(s_c) + cp.sum(s_plus_r) - cp.sum(s_plus_i2)))
    prob = cp.Problem(obj, constraints)
    
    prob.solve()

    return {
        "phi": phi.value,
        "lambdas": lambdas.value,
        "s_c": s_c.value,
        "s_plus_i2": s_plus_i2.value,
        "s_plus_r": s_plus_r.value
    }


server = app.server

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)