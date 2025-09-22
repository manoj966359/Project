import pandas as pd

def generate_pivot_table(df, index_cols, column_cols, agg_dict, show_pct):
    if not agg_dict:
        return pd.DataFrame({"Error": ["Select at least one value and aggregation function"]})

    pivot = pd.pivot_table(
        df,
        index=index_cols if index_cols else None,
        columns=column_cols if column_cols else None,
        values=list(agg_dict.keys()),
        aggfunc=agg_dict,
        fill_value=0,
    )
    pivot = pivot.reset_index()

    value_cols = pivot.select_dtypes(include="number").columns
    percent_df = pivot.copy()

    if show_pct == "Row %":
        percent_df[value_cols] = percent_df[value_cols].div(percent_df[value_cols].sum(axis=1), axis=0).fillna(0) * 100
    elif show_pct == "Column %":
        percent_df[value_cols] = percent_df[value_cols].div(percent_df[value_cols].sum(axis=0), axis=1).fillna(0) * 100
    elif show_pct == "Overall %":
        total = percent_df[value_cols].values.sum()
        percent_df[value_cols] = percent_df[value_cols].div(total).fillna(0) * 100
    else:
        percent_df[value_cols] = 0

    for col in value_cols:
        pivot[f"{col} (%)"] = percent_df[col].round(2)

    pivot.columns = [
        " ".join([str(c) for c in col if c and str(c) != ""]) if isinstance(col, tuple) else str(col)
        for col in pivot.columns
    ]
    return pivot
