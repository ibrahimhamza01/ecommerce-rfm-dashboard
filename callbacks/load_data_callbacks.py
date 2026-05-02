import os
import pandas as pd

from dash import Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate


def load_dataset_from_path(file_path: str, nrows=None) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path, nrows=nrows)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    return df


def get_dataset_shape(file_path: str):
    df = pd.read_csv(file_path)
    return df.shape, df.columns.tolist()


def get_date_bounds(file_path: str):
    df = pd.read_csv(file_path, usecols=["InvoiceDate"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    if df.empty:
        return None, None

    min_date = df["InvoiceDate"].min().date().isoformat()
    max_date = df["InvoiceDate"].max().date().isoformat()
    return min_date, max_date


def prepare_preview_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    preview_df = df.head(10).copy()

    for col in preview_df.columns:
        if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
            preview_df[col] = preview_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    preview_df = preview_df.where(pd.notnull(preview_df), None)
    preview_df.columns = [str(col) for col in preview_df.columns]
    return preview_df


def register_callbacks(app):
    @app.callback(
        [
            Output("raw-data-store", "data"),
            Output("current-data-store", "data"),
            Output("metadata-store", "data"),
            Output("load-status-badge", "children"),
            Output("load-status-badge", "color"),
            Output("load-data-alert", "children"),
            Output("load-data-alert", "color"),
            Output("rows-display", "value"),
            Output("cols-display", "value"),
            Output("preview-country-filter", "options"),
            Output("preview-country-filter", "value"),
            Output("preview-quantity-type", "value"),
            Output("preview-date-filter", "min_date_allowed"),
            Output("preview-date-filter", "max_date_allowed"),
            Output("preview-date-filter", "start_date"),
            Output("preview-date-filter", "end_date"),
            Output("preview-date-filter", "initial_visible_month"),
        ],
        [
            Input("load-data-btn", "n_clicks"),
            Input("reset-data-btn", "n_clicks"),
        ],
        [
            State("dataset-selector", "value"),
            State("raw-data-store", "data"),
        ],
        prevent_initial_call=False,
    )
    def handle_load_reset(load_clicks, reset_clicks, dataset_path, raw_store):
        ctx = callback_context

        if not ctx.triggered:
            return (
                None,
                None,
                None,
                "Awaiting load...",
                "warning",
                "No dataset loaded yet.",
                "info",
                "0",
                "0",
                [],
                [],
                "both",
                None,
                None,
                None,
                None,
                None,
            )

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        try:
            if trigger_id == "load-data-btn":
                (rows, cols), columns = get_dataset_shape(dataset_path)

                df_preview_source = load_dataset_from_path(dataset_path)
                country_options = []
                selected_countries = []

                if "Country" in df_preview_source.columns:
                    countries = sorted(
                        df_preview_source["Country"].dropna().astype(str).unique().tolist()
                    )
                    country_options = [{"label": c, "value": c} for c in countries]
                    selected_countries = countries.copy()

                min_date, max_date = get_date_bounds(dataset_path)

                raw_store_data = {"file_path": dataset_path}
                current_store_data = {"file_path": dataset_path}
                metadata = {
                    "file_path": dataset_path,
                    "rows": int(rows),
                    "cols": int(cols),
                    "columns": columns,
                    "min_date": min_date,
                    "max_date": max_date,
                }

                return (
                    raw_store_data,
                    current_store_data,
                    metadata,
                    "Loaded",
                    "success",
                    f"Dataset loaded successfully from: {dataset_path}",
                    "success",
                    str(rows),
                    str(cols),
                    country_options,
                    selected_countries,
                    "both",
                    min_date,
                    max_date,
                    min_date,
                    max_date,
                    min_date,
                )

            elif trigger_id == "reset-data-btn":
                if raw_store is None or "file_path" not in raw_store:
                    raise PreventUpdate

                dataset_path = raw_store["file_path"]
                (rows, cols), columns = get_dataset_shape(dataset_path)

                df_preview_source = load_dataset_from_path(dataset_path)
                country_options = []
                selected_countries = []

                if "Country" in df_preview_source.columns:
                    countries = sorted(
                        df_preview_source["Country"].dropna().astype(str).unique().tolist()
                    )
                    country_options = [{"label": c, "value": c} for c in countries]
                    selected_countries = countries.copy()

                min_date, max_date = get_date_bounds(dataset_path)

                current_store_data = {"file_path": dataset_path}
                metadata = {
                    "file_path": dataset_path,
                    "rows": int(rows),
                    "cols": int(cols),
                    "columns": columns,
                    "min_date": min_date,
                    "max_date": max_date,
                }

                return (
                    raw_store,
                    current_store_data,
                    metadata,
                    "Reset",
                    "secondary",
                    "Current dataset reset to the raw loaded dataset.",
                    "secondary",
                    str(rows),
                    str(cols),
                    country_options,
                    selected_countries,
                    "both",
                    min_date,
                    max_date,
                    min_date,
                    max_date,
                    min_date,
                )

            raise PreventUpdate

        except Exception as e:
            return (
                no_update,
                no_update,
                no_update,
                "Error",
                "danger",
                f"Failed in Load Data tab: {str(e)}",
                "danger",
                "0",
                "0",
                [],
                [],
                "both",
                None,
                None,
                None,
                None,
                None,
            )

    @app.callback(
        [
            Output("data-preview-table", "data"),
            Output("data-preview-table", "columns"),
        ],
        [
            Input("current-data-store", "data"),
            Input("preview-country-filter", "value"),
            Input("preview-quantity-type", "value"),
            Input("preview-date-filter", "start_date"),
            Input("preview-date-filter", "end_date"),
        ],
        prevent_initial_call=False,
    )
    def update_preview_table(
        current_store,
        selected_countries,
        quantity_type,
        start_date,
        end_date,
    ):
        if current_store is None or "file_path" not in current_store:
            return [], []

        file_path = current_store["file_path"]
        df = load_dataset_from_path(file_path)

        if "Country" in df.columns and selected_countries:
            df = df[df["Country"].astype(str).isin(selected_countries)].copy()

        if "Quantity" in df.columns:
            quantity_numeric = pd.to_numeric(df["Quantity"], errors="coerce")

            if quantity_type == "orders":
                df = df[quantity_numeric > 0].copy()
            elif quantity_type == "returns":
                df = df[quantity_numeric < 0].copy()

        if "InvoiceDate" in df.columns:
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

            if start_date:
                df = df[df["InvoiceDate"] >= pd.to_datetime(start_date)].copy()

            if end_date:
                df = df[df["InvoiceDate"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)].copy()

        preview_df = prepare_preview_dataframe(df)

        return (
            preview_df.to_dict("records"),
            [{"name": str(col), "id": str(col)} for col in preview_df.columns],
        )