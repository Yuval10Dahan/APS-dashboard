import streamlit as st
st.set_page_config(page_title="APS Disruption Time Results", layout="wide", initial_sidebar_state="expanded")

import os
import io
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
from PIL import Image

# =========================================
# CONFIG
# =========================================
DB_FILENAME = "APS_data_base2.db"
MAIN_TABLE = 'Disruption Time Measurement'
LOGO_FILENAME = "Packetlight Logo.png"

DB_PATH = os.path.join(os.path.dirname(__file__), DB_FILENAME)
engine = create_engine(f"sqlite:///{DB_PATH}")

DISPLAY_COLUMNS_MAP = {
    "_rowid_": "ID",
    "Product Name": "Product Name",
    "Protection Type": "Protection Type",
    "SoftWare Version": "Software Version",
    "System Mode": "System Mode",
    "Uplink Service Type": "Uplink Service Type",
    "Client Service Type": "Client Service Type",
    "Transceiver PN": "Transceiver PN",
    "Transceiver FW": "Transceiver FW",
    "Time Stamp": "Date & Time",
    "Number": "Sample Number",
    "W2P Measurement": "W2P (ms)",
    "P2W Measurement": "P2W (ms)",
}

# The columns that define a "configuration" (combinations)
CONFIG_COLS = [
    "Product Name",
    "Protection Type",
    "SoftWare Version",
    "System Mode",
    "Uplink Service Type",
    "Client Service Type",
    "Transceiver PN",
    "Transceiver FW",
    "Time Stamp",
]

# Auto->Log heuristic (since we removed the slider option)
AUTO_LOG_RATIO_THRESHOLD = 200  # max/median >= this => use log

# =========================================
# HELPERS
# =========================================
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_sql(f'SELECT rowid as _rowid_, * FROM "{MAIN_TABLE}"', engine)

    # Normalize timestamp column
    if "Time Stamp" in df.columns:
        parsed = pd.to_datetime(df["Time Stamp"], errors="coerce", dayfirst=True)
        df["Time Stamp"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
        df.loc[parsed.isna(), "Time Stamp"] = df.loc[parsed.isna(), "Time Stamp"].astype(str)

    # Convert measurements to numeric
    for c in ["W2P Measurement", "P2W Measurement"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure "Number" numeric
    if "Number" in df.columns:
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce")

    desired_order = [
        "_rowid_",
        "Product Name",
        "Protection Type",
        "SoftWare Version",
        "System Mode",
        "Uplink Service Type",
        "Client Service Type",
        "Transceiver PN",
        "Transceiver FW",
        "Time Stamp",
        "Number",
        "W2P Measurement",
        "P2W Measurement",
    ]
    df = df[[c for c in desired_order if c in df.columns] + [c for c in df.columns if c not in desired_order]]
    return df


def build_column_config_for_autowidth(df: pd.DataFrame, min_px=90, max_px=380, px_per_char=7):
    cfg = {}
    for col in df.columns:
        s = df[col].astype(str).fillna("")
        max_len = max([len(str(col))] + s.map(len).tolist())
        width_px = int(max_len * px_per_char + 24)
        width_px = max(min_px, min(max_px, width_px))
        cfg[col] = st.column_config.Column(width=width_px)
    return cfg


def sidebar_filters(df: pd.DataFrame):
    """
    Sidebar filters:
      - "Base filters" affect BOTH Summary + Full table
      - W2P/P2W filters affect ONLY the Full table (records)
    """
    with st.sidebar:
        st.subheader("Contact: Yuval Dahan")
        st.header("🔍 Filters")

        filtered_options_df = df.copy()

        def multisel(col, label):
            nonlocal filtered_options_df
            selected = []
            if col in filtered_options_df.columns:
                selected = st.multiselect(label, sorted(filtered_options_df[col].dropna().unique()))
                if selected:
                    filtered_options_df = filtered_options_df[filtered_options_df[col].isin(selected)]
            return selected

        # ---- Base filters (shared) ----
        selected_product = multisel("Product Name", "Product Name")
        selected_protection = multisel("Protection Type", "Protection Type")
        selected_sw = multisel("SoftWare Version", "Software Version")
        selected_mode = multisel("System Mode", "System Mode")
        selected_uplink = multisel("Uplink Service Type", "Uplink Service Type")
        selected_client = multisel("Client Service Type", "Client Service Type")
        selected_transceiver_pn = multisel("Transceiver PN", "Transceiver PN")
        selected_transceiver_fw = multisel("Transceiver FW", "Transceiver FW")

        selected_timestamp = []
        if "Time Stamp" in filtered_options_df.columns:
            ts_options = sorted(filtered_options_df["Time Stamp"].dropna().unique(), reverse=True)
            selected_timestamp = st.multiselect("Date & Time", ts_options)
            if selected_timestamp:
                filtered_options_df = filtered_options_df[filtered_options_df["Time Stamp"].isin(selected_timestamp)]

        # ---- Measurements filters (records-only) ----
        st.header("⏱️ W2P Filter (Only Full table)")
        w2p_filter_type = st.radio("Filter W2P:", ["Show All", "Above", "Below"], horizontal=True, key="w2p_radio")
        w2p_threshold = st.number_input("W2P Threshold", min_value=0.0, step=0.1, key="w2p_thr")

        st.header("⏱️ P2W Filter (Only Full table)")
        p2w_filter_type = st.radio("Filter P2W:", ["Show All", "Above", "Below"], horizontal=True, key="p2w_radio")
        p2w_threshold = st.number_input("P2W Threshold", min_value=0.0, step=0.1, key="p2w_thr")

        st.header("🧩 Columns to Display (Only Full table)")
        st.caption("Toggle columns on/off for the FULL table view:")
        display_df_preview = df.rename(columns=DISPLAY_COLUMNS_MAP)
        checkbox_columns = {col: st.checkbox(col, value=True) for col in display_df_preview.columns}
        selected_columns = [col for col, show in checkbox_columns.items() if show]

    base_filters = {
        "selected_product": selected_product,
        "selected_protection": selected_protection,
        "selected_sw": selected_sw,
        "selected_mode": selected_mode,
        "selected_uplink": selected_uplink,
        "selected_client": selected_client,
        "selected_transceiver_pn": selected_transceiver_pn,
        "selected_transceiver_fw": selected_transceiver_fw,
        "selected_timestamp": selected_timestamp,
    }

    measurement_filters = {
        "w2p_filter_type": w2p_filter_type,
        "w2p_threshold": w2p_threshold,
        "p2w_filter_type": p2w_filter_type,
        "p2w_threshold": p2w_threshold,
    }

    return base_filters, measurement_filters, selected_columns


def apply_base_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    """
    Applies ONLY the "shared" filters (no W2P/P2W thresholds here).
    Used for BOTH Summary and Full table.
    """
    out = df.copy()

    def apply_in(col, values):
        nonlocal out
        if values and col in out.columns:
            out = out[out[col].isin(values)]

    apply_in("Product Name", f["selected_product"])
    apply_in("Protection Type", f["selected_protection"])
    apply_in("SoftWare Version", f["selected_sw"])
    apply_in("System Mode", f["selected_mode"])
    apply_in("Uplink Service Type", f["selected_uplink"])
    apply_in("Client Service Type", f["selected_client"])
    apply_in("Transceiver PN", f["selected_transceiver_pn"])
    apply_in("Transceiver FW", f["selected_transceiver_fw"])
    apply_in("Time Stamp", f["selected_timestamp"])

    return out


def apply_measurement_filters_records_only(df: pd.DataFrame, mf: dict) -> pd.DataFrame:
    """
    Applies ONLY W2P/P2W Above/Below thresholds.
    This should apply ONLY to the Full table.
    """
    out = df.copy()

    if "W2P Measurement" in out.columns:
        if mf["w2p_filter_type"] == "Above":
            out = out[out["W2P Measurement"] > mf["w2p_threshold"]]
        elif mf["w2p_filter_type"] == "Below":
            out = out[out["W2P Measurement"] < mf["w2p_threshold"]]

    if "P2W Measurement" in out.columns:
        if mf["p2w_filter_type"] == "Above":
            out = out[out["P2W Measurement"] > mf["p2w_threshold"]]
        elif mf["p2w_filter_type"] == "Below":
            out = out[out["P2W Measurement"] < mf["p2w_threshold"]]

    return out


def calc_distribution(series: pd.Series) -> dict:
    """
    Buckets:
      <=50ms, >50ms
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    total = int(len(s))
    if total == 0:
        return {
            "Below/Equal 50mSec [%]": 0.0,
            "Above 50mSec [%]": 0.0,
            "Total Number of Measurements": 0,
        }

    below_50 = (s <= 50).sum()
    above_50 = (s > 50).sum()

    return {
        "Below/Equal 50mSec [%]": (below_50 / total) * 100.0,
        "Above 50mSec [%]": (above_50 / total) * 100.0,
        "Total Number of Measurements": total,
    }


def build_summary_table(filtered_df_original_names: pd.DataFrame) -> pd.DataFrame:
    """
    filtered_df_original_names: must contain original DB column names.
    NOTE: This MUST be built from the BASE-filtered dataframe (NO W2P/P2W thresholds).
    """
    cols_present = [c for c in CONFIG_COLS if c in filtered_df_original_names.columns]
    if not cols_present:
        return pd.DataFrame()

    grouped = filtered_df_original_names.groupby(cols_present, dropna=False)

    rows = []
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(cols_present, key))

        w2p_dist = calc_distribution(g.get("W2P Measurement"))
        p2w_dist = calc_distribution(g.get("P2W Measurement"))

        row.update({
            "W2P Below/Equal 50ms [%]": w2p_dist["Below/Equal 50mSec [%]"],
            "W2P Above 50ms [%]": w2p_dist["Above 50mSec [%]"],
            "P2W Below/Equal 50ms [%]": p2w_dist["Below/Equal 50mSec [%]"],
            "P2W Above 50ms [%]": p2w_dist["Above 50mSec [%]"],
            "Total Number of Measurements": int(len(g)),
        })
        rows.append(row)

    out = pd.DataFrame(rows)

    pct_cols = [c for c in out.columns if c.endswith("[%]")]
    out[pct_cols] = out[pct_cols].round(4)

    if "Time Stamp" in out.columns:
        out = out.sort_values("Time Stamp", ascending=False)

    return out


def df_to_excel_bytes(df: pd.DataFrame, sheet_name="Sheet1", logo_path: str | None = None,
                      title: str | None = None) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        start_row = 0
        if logo_path and os.path.exists(logo_path):
            start_row = 5

        df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        if logo_path and os.path.exists(logo_path):
            worksheet.insert_image("A1", logo_path, {"x_scale": 0.5, "y_scale": 0.5})

        if title:
            title_format = workbook.add_format({"bold": True, "font_size": 16, "align": "left", "valign": "vcenter"})
            worksheet.write("A4", title, title_format)

        header_format = workbook.add_format({
            "bold": True, "align": "center", "valign": "vcenter",
            "bg_color": "#D9E1F2", "border": 1
        })
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(start_row, col_num, value, header_format)

        cell_format = workbook.add_format({"align": "center", "valign": "vcenter", "border": 1})
        for r in range(len(df)):
            for c in range(len(df.columns)):
                worksheet.write(start_row + 1 + r, c, df.iloc[r, c], cell_format)

        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)

        worksheet.freeze_panes(start_row + 1, 0)

    output.seek(0)
    return output.getvalue()


def render_graph_by_combination_id(
    base_filtered_original_df: pd.DataFrame,
    summary_df_original: pd.DataFrame,
    id_col: str = "Combination ID",
):
    """
    Graph is based on BASE-FILTERED dataset (no W2P/P2W thresholds),
    because the combination itself is defined by base filters.

    Graph display options:
      - Y-axis scale: Auto / Log

    UI change requested:
      - Make the "Enter Combination ID" input small (roughly label-sized),
        not a huge full-width input.
    """
    st.divider()
    st.subheader("📈 Generate Graph")

    if summary_df_original.empty:
        st.info("No combinations available to plot.")
        return

    max_id = int(summary_df_original[id_col].max()) if id_col in summary_df_original.columns else 1
    max_id = max(1, max_id)

    # ---- Compact input row (label + small input + spacer) ----
    c1, c2, c3 = st.columns([1.6, 1.0, 8.0])  # c2 is intentionally small
    with c1:
        st.markdown("**Enter Combination ID**")
    with c2:
        comb_id = st.number_input(
            label="",
            min_value=1,
            max_value=max_id,
            value=1,
            step=1,
            key="comb_id_input",
            label_visibility="collapsed",
        )
    with c3:
        st.empty()

    # with st.expander("Graph display options", expanded=True):
    #     scale_mode = st.radio("Y-axis scale", ["Auto", "Log"], horizontal=True, key="y_scale_mode")
    with st.popover("Graph display options"):
        scale_mode = st.radio("Y-axis scale", ["Auto", "Log"], horizontal=True, key="y_scale_mode")

    if st.button("📊 Generate Graph", key="btn_graph_by_id"):
        if id_col not in summary_df_original.columns:
            st.error("Internal error: Summary table does not include Combination ID.")
            return

        row = summary_df_original.loc[summary_df_original[id_col] == int(comb_id)]
        if row.empty:
            st.error(f"Combination ID {comb_id} not found.")
            return

        cfg_cols_present = [c for c in CONFIG_COLS if c in base_filtered_original_df.columns and c in row.columns]
        if not cfg_cols_present:
            st.error("Missing configuration columns for filtering the combination.")
            return

        mask = pd.Series(True, index=base_filtered_original_df.index)
        for c in cfg_cols_present:
            v = row.iloc[0][c]
            if pd.isna(v):
                mask &= base_filtered_original_df[c].isna()
            else:
                mask &= (base_filtered_original_df[c] == v)

        plot_df = base_filtered_original_df.loc[mask].copy()

        required_cols = {"Number", "W2P Measurement", "P2W Measurement"}
        miss_cols = [c for c in required_cols if c not in plot_df.columns]
        if miss_cols:
            st.error(f"Missing required columns: {miss_cols}")
            return

        plot_df = plot_df.dropna(subset=["Number"]).sort_values("Number")
        if plot_df.empty:
            st.warning("No samples to plot for this Combination ID (after filters).")
            return

        # Title
        details_parts = []
        for c in ["Product Name", "Protection Type", "SoftWare Version", "System Mode",
                  "Uplink Service Type", "Client Service Type", "Transceiver PN", "Transceiver FW", "Time Stamp"]:
            if c in row.columns:
                val = row.iloc[0][c]
                if pd.isna(val):
                    continue
                details_parts.append(str(val))
        details = " | ".join(details_parts[:4])
        title_prefix = f"Combination ID {comb_id} - Graph"
        if details:
            title_prefix += f"<br><sup>{details}</sup>"

        # Prepare y
        w2p = pd.to_numeric(plot_df["W2P Measurement"], errors="coerce")
        p2w = pd.to_numeric(plot_df["P2W Measurement"], errors="coerce")
        y_all = pd.concat([w2p, p2w]).dropna()

        if y_all.empty:
            st.warning("No valid measurements to plot.")
            return

        # Scale decision
        if scale_mode == "Log":
            use_log = True
        else:
            med = float(y_all.median()) if len(y_all) else 0.0
            mx = float(y_all.max())
            ratio = (mx / med) if med and med > 0 else float("inf")
            use_log = ratio >= AUTO_LOG_RATIO_THRESHOLD

        if use_log and scale_mode == "Auto":
            st.info("Y-axis switched to **log scale** automatically (large max/median spread).")

        # Linear y-range (only if not log)
        w2p_plot = w2p.copy()
        p2w_plot = p2w.copy()

        y_min = float(pd.concat([w2p_plot, p2w_plot]).min())
        y_max = float(pd.concat([w2p_plot, p2w_plot]).max())
        pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        y_range = [y_min - pad, y_max + pad]

        # W2P plot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=plot_df["Number"], y=w2p_plot, mode="lines", name="W2P (ms)",
            line=dict(width=2),
            connectgaps=True
        ))
        fig1.update_layout(
            title=dict(text=f"{title_prefix}<br><sup>W2P</sup>", x=0.5),
            xaxis=dict(title="Cycle / Sample Number", tickangle=90, nticks=35, showgrid=False),
            yaxis=dict(title="Disruption Time (mSec)", showgrid=True),
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
            height=420,
            margin=dict(l=60, r=30, t=80, b=80),
        )
        fig1.update_xaxes(rangeslider_visible=False)
        fig1.update_yaxes(type="log" if use_log else "linear")
        if not use_log:
            fig1.update_yaxes(range=y_range)
        st.plotly_chart(fig1, use_container_width=True)

        st.divider()

        # P2W plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=plot_df["Number"], y=p2w_plot, mode="lines", name="P2W (ms)",
            line=dict(width=2),
            connectgaps=True
        ))
        fig2.update_layout(
            title=dict(text=f"{title_prefix}<br><sup>P2W</sup>", x=0.5),
            xaxis=dict(title="Cycle / Sample Number", tickangle=90, nticks=35, showgrid=False),
            yaxis=dict(title="Disruption Time (mSec)", showgrid=True),
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
            height=420,
            margin=dict(l=60, r=30, t=80, b=80),
        )
        fig2.update_xaxes(rangeslider_visible=False)
        fig2.update_yaxes(type="log" if use_log else "linear")
        if not use_log:
            fig2.update_yaxes(range=y_range)
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Show samples used"):
            st.dataframe(plot_df[["Number", "W2P Measurement", "P2W Measurement"]], use_container_width=True)


def render_records_section(
    summary_display_df: pd.DataFrame,          # display names, BASE-filtered
    records_display_df: pd.DataFrame,          # display names, BASE+measurement-filtered
    selected_columns: list[str],
    logo_path: str
):
    """
    Summary tab uses BASE-filtered data ONLY (no W2P/P2W).
    Full tab uses RECORDS-filtered data (W2P/P2W apply).
    """
    st.divider()

    base_original_df = summary_display_df.rename(columns={v: k for k, v in DISPLAY_COLUMNS_MAP.items()})
    summary_df = build_summary_table(base_original_df)

    if not summary_df.empty:
        summary_df = summary_df.reset_index(drop=True)
        summary_df.insert(0, "Combination ID", range(1, len(summary_df) + 1))

    combinations_count = int(len(summary_df))

    tab_summary, tab_full = st.tabs(["Summary by Configuration", "Show Measurements Full Table"])

    with tab_summary:
        st.subheader(f"Showing {combinations_count} Combinations")

        if summary_df.empty:
            st.info("No summary available (missing configuration columns).")
        else:
            shown = summary_df.rename(columns={
                "SoftWare Version": "Software Version",
                "Time Stamp": "Date & Time",
            })
            cfg = build_column_config_for_autowidth(shown)
            st.dataframe(shown, use_container_width=True, hide_index=True, column_config=cfg)

            comb_excel = df_to_excel_bytes(
                shown,
                sheet_name="Combinations",
                logo_path=logo_path,
                title="PacketLight APS Disruption Time Results (Combinations)"
            )
            st.download_button(
                "Download Combinations Results - Excel File",
                data=comb_excel,
                file_name="aps_combinations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_combinations",
            )

    with tab_full:
        st.subheader(f"Showing {len(records_display_df)} Records")

        if len(records_display_df) == 0:
            st.info("No records to display.")
        else:
            table_df = records_display_df[[c for c in selected_columns if c in records_display_df.columns]].copy()
            col_cfg = build_column_config_for_autowidth(table_df)

            st.data_editor(
                table_df,
                use_container_width=True,
                hide_index=False,
                disabled=True,
                column_config=col_cfg
            )

            rec_excel = df_to_excel_bytes(
                table_df,
                sheet_name="APS Results",
                logo_path=logo_path,
                title="PacketLight APS Disruption Time Results"
            )
            st.download_button(
                "Download Filtered Results - Excel File",
                data=rec_excel,
                file_name="aps_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_records",
            )

    return base_original_df, summary_df


# =========================================
# MAIN
# =========================================
df = load_data()

# Header
logo_path = os.path.join(os.path.dirname(__file__), LOGO_FILENAME)
if os.path.exists(logo_path):
    st.image(Image.open(logo_path), width=250)

st.title("PacketLight - APS Disruption Time Results")
st.subheader("(W2P / P2W Disruption Time Measurements)")

# Sidebar + filters
base_filters, measurement_filters, selected_columns = sidebar_filters(df)

# 1) Base filtered df (for SUMMARY + also base for full-table)
base_filtered_df = apply_base_filters(df, base_filters)

# 2) Records df = base + W2P/P2W thresholds (for FULL TABLE ONLY)
records_filtered_df = apply_measurement_filters_records_only(base_filtered_df, measurement_filters)

# Rename for display
summary_display_df = base_filtered_df.rename(columns=DISPLAY_COLUMNS_MAP)
records_display_df = records_filtered_df.rename(columns=DISPLAY_COLUMNS_MAP)

selected_columns = [c for c in selected_columns if c in records_display_df.columns]

# Records section (tabs)
base_original_df_for_graph, summary_df_original = render_records_section(
    summary_display_df=summary_display_df,
    records_display_df=records_display_df,
    selected_columns=selected_columns,
    logo_path=logo_path
)

# Graph by Combination ID (based on BASE-filtered df, not on W2P/P2W thresholds)
render_graph_by_combination_id(base_original_df_for_graph, summary_df_original, id_col="Combination ID")