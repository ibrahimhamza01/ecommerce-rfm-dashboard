import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate


CACHE_DIR = "cache"
CURRENT_CLEANED_FILE = os.path.join(CACHE_DIR, "current_cleaned_dataset.csv")


def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    return df


def save_dataset(df: pd.DataFrame, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_to_save = df.copy()

    if "InvoiceDate" in df_to_save.columns:
        df_to_save["InvoiceDate"] = pd.to_datetime(
            df_to_save["InvoiceDate"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

    df_to_save.to_csv(file_path, index=False)


def get_missing_column_options(file_path: str):
    df = load_dataset(file_path)
    missing_counts = df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
    options = [
        {"label": f"{col} ({int(missing_counts[col]):,} missing)", "value": col}
        for col in cols_with_missing
    ]
    value = cols_with_missing[0] if cols_with_missing else None
    return options, value


def humanize_method(method: str) -> str:
    mapping = {
        "keep": "Keep Missing Values",
        "drop_rows": "Drop Rows with Missing Values",
        "ffill": "Forward Fill",
        "bfill": "Backward Fill",
        "mean": "Fill with Mean",
        "median": "Fill with Median",
        "mode": "Fill with Mode",
        "constant": "Fill with Constant",
        "flag_missing": "Add Missing Flag Column",
    }
    return mapping.get(method, method)


def apply_missing_treatment(df: pd.DataFrame, column: str, method: str, constant_value: str):
    if column is None or column not in df.columns:
        return df, f"Skipped missing-value step because column '{column}' was not available."

    result_df = df.copy()
    steps = []

    before_missing = int(result_df[column].isna().sum())

    if method == "keep":
        steps.append(f"Kept missing values in '{column}' unchanged.")

    elif method == "drop_rows":
        before_rows = len(result_df)
        result_df = result_df[result_df[column].notna()].copy()
        removed = before_rows - len(result_df)
        steps.append(f"Dropped rows with missing '{column}': {removed:,}")

    elif method == "ffill":
        result_df[column] = result_df[column].ffill()
        steps.append(f"Applied forward fill to '{column}'.")

    elif method == "bfill":
        result_df[column] = result_df[column].bfill()
        steps.append(f"Applied backward fill to '{column}'.")

    elif method == "mean":
        if pd.api.types.is_numeric_dtype(result_df[column]):
            fill_value = result_df[column].mean()
            result_df[column] = result_df[column].fillna(fill_value)
            steps.append(f"Filled missing '{column}' with mean: {fill_value:.2f}")
        else:
            steps.append(f"Skipped mean fill for '{column}' because it is not numeric.")

    elif method == "median":
        if pd.api.types.is_numeric_dtype(result_df[column]):
            fill_value = result_df[column].median()
            result_df[column] = result_df[column].fillna(fill_value)
            steps.append(f"Filled missing '{column}' with median: {fill_value:.2f}")
        else:
            steps.append(f"Skipped median fill for '{column}' because it is not numeric.")

    elif method == "mode":
        mode_series = result_df[column].mode(dropna=True)
        if not mode_series.empty:
            fill_value = mode_series.iloc[0]
            result_df[column] = result_df[column].fillna(fill_value)
            steps.append(f"Filled missing '{column}' with mode: {fill_value}")
        else:
            steps.append(f"Skipped mode fill for '{column}' because no mode was available.")

    elif method == "constant":
        result_df[column] = result_df[column].fillna(constant_value)
        steps.append(f"Filled missing '{column}' with constant value: {constant_value}")

    elif method == "flag_missing":
        flag_col = f"{column}_MissingFlag"
        result_df[flag_col] = result_df[column].isna().astype(int)
        steps.append(f"Added missing flag column: '{flag_col}'")

    after_missing = int(result_df[column].isna().sum()) if column in result_df.columns else 0
    steps.append(f"Missing values in '{column}': {before_missing:,} → {after_missing:,}")

    return result_df, " ".join(steps)


def apply_duplicate_handling(df: pd.DataFrame, method: str):
    result_df = df.copy()

    if method == "remove_exact":
        before_rows = len(result_df)
        result_df = result_df.drop_duplicates().copy()
        removed = before_rows - len(result_df)
        return result_df, f"Removed exact duplicate rows: {removed:,}"

    return result_df, "Kept duplicate rows unchanged."


def apply_admin_filter(df: pd.DataFrame, method: str):
    result_df = df.copy()

    if method == "remove_admin" and "Description" in result_df.columns:
        before_rows = len(result_df)

        desc = result_df["Description"].astype(str).str.lower()
        admin_mask = (
            desc.str.contains("manual", na=False)
            | desc.str.contains("bad debt", na=False)
            | desc.str.contains("adjust", na=False)
        )

        result_df = result_df[~admin_mask].copy()
        removed = before_rows - len(result_df)
        return result_df, f"Removed administrative rows: {removed:,}"

    return result_df, "Kept administrative rows unchanged."


def apply_feature_engineering(df: pd.DataFrame, features: list):
    result_df = df.copy()
    steps = []
    features = features or []

    if "linetotal" in features:
        if "Quantity" in result_df.columns and "Price" in result_df.columns:
            result_df["LineTotal"] = (
                pd.to_numeric(result_df["Quantity"], errors="coerce")
                * pd.to_numeric(result_df["Price"], errors="coerce")
            )
            steps.append("Created LineTotal = Quantity × Price")

    if "transaction_status" in features:
        if "Invoice" in result_df.columns and "Quantity" in result_df.columns:
            invoice_str = result_df["Invoice"].astype(str)
            qty = pd.to_numeric(result_df["Quantity"], errors="coerce")
            result_df["TransactionStatus"] = "Completed"
            result_df.loc[invoice_str.str.startswith("C", na=False) | (qty < 0), "TransactionStatus"] = "Cancelled"
            steps.append("Created TransactionStatus")

    if "purchase_quarter" in features:
        if "InvoiceDate" in result_df.columns:
            result_df["InvoiceDate"] = pd.to_datetime(result_df["InvoiceDate"], errors="coerce")
            result_df["PurchaseQuarter"] = result_df["InvoiceDate"].dt.to_period("Q").astype(str)
            steps.append("Created PurchaseQuarter")

    if "price_category" in features:
        if "Price" in result_df.columns:
            price_numeric = pd.to_numeric(result_df["Price"], errors="coerce")
            result_df["PriceCategory"] = pd.cut(
                price_numeric,
                bins=[-float("inf"), 0, 2, 5, 10, float("inf")],
                labels=["Negative/Zero", "Low", "Medium", "High", "Premium"],
            )
            result_df["PriceCategory"] = result_df["PriceCategory"].astype(str)
            steps.append("Created PriceCategory")

    if not steps:
        steps.append("No feature engineering selected.")

    return result_df, steps


def build_comparison_chart(before_rows, after_rows, before_total_missing, after_total_missing):
    fig = go.Figure()
    fig.add_bar(name="Before", x=["Rows", "Total Missing Values"], y=[before_rows, before_total_missing])
    fig.add_bar(name="After", x=["Rows", "Total Missing Values"], y=[after_rows, after_total_missing])
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title="Before vs After Cleaning",
        yaxis_title="Count",
        legend_title="Dataset State",
    )
    return fig


def build_transaction_status_chart(df: pd.DataFrame):
    if "TransactionStatus" not in df.columns:
        return go.Figure().update_layout(template="plotly_white", title="TransactionStatus not available")

    counts = df["TransactionStatus"].fillna("Unknown").value_counts().reset_index()
    counts.columns = ["TransactionStatus", "Count"]
    fig = px.bar(counts, x="TransactionStatus", y="Count", text="Count", template="plotly_white")
    fig.update_traces(textposition="outside")
    fig.update_layout(title="Transaction Status Distribution")
    return fig


def build_purchase_quarter_chart(df: pd.DataFrame):
    if "PurchaseQuarter" not in df.columns:
        return go.Figure().update_layout(template="plotly_white", title="PurchaseQuarter not available")

    counts = df["PurchaseQuarter"].fillna("Unknown").value_counts().sort_index().reset_index()
    counts.columns = ["PurchaseQuarter", "Count"]
    fig = px.bar(counts, x="PurchaseQuarter", y="Count", text="Count", template="plotly_white")
    fig.update_traces(textposition="outside")
    fig.update_layout(title="Purchase Quarter Distribution", xaxis_tickangle=-45)
    return fig


def build_price_category_chart(df: pd.DataFrame):
    if "PriceCategory" not in df.columns:
        return go.Figure().update_layout(template="plotly_white", title="PriceCategory not available")

    counts = df["PriceCategory"].fillna("Unknown").value_counts().reset_index()
    counts.columns = ["PriceCategory", "Count"]
    fig = px.bar(counts, x="PriceCategory", y="Count", text="Count", template="plotly_white")
    fig.update_traces(textposition="outside")
    fig.update_layout(title="Price Category Distribution")
    return fig


def register_callbacks(app):
    @app.callback(
        [
            Output("clean-missing-column", "options"),
            Output("clean-missing-column", "value"),
        ],
        [
            Input("raw-data-store", "data"),
            Input("current-data-store", "data"),
        ],
        prevent_initial_call=False,
    )
    def populate_missing_columns(raw_store, current_store):
        file_path = None

        if raw_store and "file_path" in raw_store:
            file_path = raw_store["file_path"]
        elif current_store and "file_path" in current_store:
            file_path = current_store["file_path"]

        if file_path is None:
            return [], None

        try:
            return get_missing_column_options(file_path)
        except Exception:
            return [], None

    @app.callback(
        [
            Output("missing-rules-store", "data"),
            Output("missing-rules-table", "data"),
        ],
        [
            Input("add-missing-rule-btn", "n_clicks"),
            Input("clear-missing-rules-btn", "n_clicks"),
        ],
        [
            State("clean-missing-column", "value"),
            State("clean-missing-select", "value"),
            State("clean-constant-value", "value"),
            State("missing-rules-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def manage_missing_rules(add_clicks, clear_clicks, column, method, constant_value, existing_rules):
        from dash import callback_context

        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        existing_rules = existing_rules or []

        if trigger_id == "clear-missing-rules-btn":
            return [], []

        if trigger_id == "add-missing-rule-btn":
            if column is None or method is None:
                raise PreventUpdate

            new_rule = {
                "column": column,
                "method": humanize_method(method),
                "method_key": method,
                "constant_value": constant_value if constant_value is not None else "",
            }

            updated_rules = existing_rules + [new_rule]
            table_data = [
                {
                    "column": rule["column"],
                    "method": rule["method"],
                    "constant_value": rule["constant_value"],
                }
                for rule in updated_rules
            ]
            return updated_rules, table_data

        raise PreventUpdate

    @app.callback(
        [
            Output("current-data-store", "data", allow_duplicate=True),
            Output("metadata-store", "data", allow_duplicate=True),
            Output("cleaning-alert", "children"),
            Output("cleaning-alert", "color"),
            Output("cleaning-summary-text", "children"),
            Output("cleaning-comparison-chart", "figure"),
            Output("transaction-status-chart", "figure"),
            Output("purchase-quarter-chart", "figure"),
            Output("price-category-chart", "figure"),
        ],
        Input("apply-cleaning-btn", "n_clicks"),
        [
            State("raw-data-store", "data"),
            State("missing-rules-store", "data"),
            State("clean-duplicate-select", "value"),
            State("clean-admin-select", "value"),
            State("feature-engineering-checklist", "value"),
            State("metadata-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def apply_cleaning(
        n_clicks,
        raw_store,
        missing_rules,
        duplicate_rule,
        admin_rule,
        feature_flags,
        metadata,
    ):
        if not n_clicks:
            raise PreventUpdate

        if raw_store is None or "file_path" not in raw_store:
            raise PreventUpdate

        source_file = raw_store["file_path"]
        df = load_dataset(source_file)

        before_rows = len(df)
        before_total_missing = int(df.isna().sum().sum())

        steps_list = []
        working_df = df.copy()

        for rule in (missing_rules or []):
            working_df, step = apply_missing_treatment(
                df=working_df,
                column=rule["column"],
                method=rule["method_key"],
                constant_value=rule.get("constant_value", ""),
            )
            steps_list.append(step)

        if not missing_rules:
            steps_list.append("No missing-value rules were applied.")

        working_df, duplicate_step = apply_duplicate_handling(
            df=working_df,
            method=duplicate_rule,
        )
        steps_list.append(duplicate_step)

        working_df, admin_step = apply_admin_filter(
            df=working_df,
            method=admin_rule,
        )
        steps_list.append(admin_step)

        working_df, feature_steps = apply_feature_engineering(
            df=working_df,
            features=feature_flags,
        )
        steps_list.extend(feature_steps)

        after_rows = len(working_df)
        after_total_missing = int(working_df.isna().sum().sum())

        save_dataset(working_df, CURRENT_CLEANED_FILE)

        updated_current_store = {"file_path": CURRENT_CLEANED_FILE}

        updated_metadata = metadata.copy() if metadata else {}
        updated_metadata["current_file_path"] = CURRENT_CLEANED_FILE
        updated_metadata["current_rows"] = int(after_rows)
        updated_metadata["cleaning_applied"] = True
        updated_metadata["feature_engineering_applied"] = feature_flags or []

        summary_children = html.Div(
            [
                html.P(f"Rows before cleaning: {before_rows:,}"),
                html.P(f"Rows after cleaning: {after_rows:,}"),
                html.P(f"Total missing values before cleaning: {before_total_missing:,}"),
                html.P(f"Total missing values after cleaning: {after_total_missing:,}"),
                html.Hr(),
                html.Ul([html.Li(step) for step in steps_list]),
            ]
        )

        comparison_fig = build_comparison_chart(
            before_rows=before_rows,
            after_rows=after_rows,
            before_total_missing=before_total_missing,
            after_total_missing=after_total_missing,
        )

        transaction_fig = build_transaction_status_chart(working_df)
        quarter_fig = build_purchase_quarter_chart(working_df)
        price_category_fig = build_price_category_chart(working_df)

        return (
            updated_current_store,
            updated_metadata,
            "Cleaning applied successfully. The cleaned dataset is now the active dataset for the next tabs.",
            "success",
            summary_children,
            comparison_fig,
            transaction_fig,
            quarter_fig,
            price_category_fig,
        )