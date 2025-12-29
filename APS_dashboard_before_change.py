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


# =========================================
# HELPERS
# =========================================
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_sql(f'SELECT rowid as _rowid_, * FROM "{MAIN_TABLE}"', engine)

    if "Time Stamp" in df.columns:
        parsed = pd.to_datetime(df["Time Stamp"], errors="coerce", dayfirst=True)
        df["Time Stamp"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
        df.loc[parsed.isna(), "Time Stamp"] = df.loc[parsed.isna(), "Time Stamp"].astype(str)

    for c in ["W2P Measurement", "P2W Measurement"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
    Builds the sidebar UI and returns:
      - filters dict (raw selections)
      - selected_columns (renamed columns selected for table)
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

        st.header("🆔 Filter by Sample Number")
        number_input = st.text_input("Enter sample numbers (comma-separated)", value="")
        number_list = []
        if number_input.strip():
            number_list = [int(x.strip()) for x in number_input.split(",") if x.strip().isdigit()]

        st.header("⏱️ W2P Filter")
        w2p_filter_type = st.radio("Filter W2P:", ["Show All", "Above", "Below"], horizontal=True, key="w2p_radio")
        w2p_threshold = st.number_input("W2P Threshold", min_value=0.0, step=0.1, key="w2p_thr")

        st.header("⏱️ P2W Filter")
        p2w_filter_type = st.radio("Filter P2W:", ["Show All", "Above", "Below"], horizontal=True, key="p2w_radio")
        p2w_threshold = st.number_input("P2W Threshold", min_value=0.0, step=0.1, key="p2w_thr")

        st.header("🧩 Columns to Display")
        st.caption("Toggle columns on/off to display in the table:")

        display_df_preview = df.rename(columns=DISPLAY_COLUMNS_MAP)
        checkbox_columns = {col: st.checkbox(col, value=True) for col in display_df_preview.columns}
        selected_columns = [col for col, show in checkbox_columns.items() if show]

    filters = {
        "selected_product": selected_product,
        "selected_protection": selected_protection,
        "selected_sw": selected_sw,
        "selected_mode": selected_mode,
        "selected_uplink": selected_uplink,
        "selected_client": selected_client,
        "selected_transceiver_pn": selected_transceiver_pn,
        "selected_transceiver_fw": selected_transceiver_fw,
        "selected_timestamp": selected_timestamp,
        "number_list": number_list,
        "w2p_filter_type": w2p_filter_type,
        "w2p_threshold": w2p_threshold,
        "p2w_filter_type": p2w_filter_type,
        "p2w_threshold": p2w_threshold,
    }
    return filters, selected_columns


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
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

    if f["number_list"] and "Number" in out.columns:
        out = out[out["Number"].isin(f["number_list"])]

    if "W2P Measurement" in out.columns:
        if f["w2p_filter_type"] == "Above":
            out = out[out["W2P Measurement"] > f["w2p_threshold"]]
        elif f["w2p_filter_type"] == "Below":
            out = out[out["W2P Measurement"] < f["w2p_threshold"]]

    if "P2W Measurement" in out.columns:
        if f["p2w_filter_type"] == "Above":
            out = out[out["P2W Measurement"] > f["p2w_threshold"]]
        elif f["p2w_filter_type"] == "Below":
            out = out[out["P2W Measurement"] < f["p2w_threshold"]]

    return out


def render_config_graph(df: pd.DataFrame):
    st.divider()
    st.subheader("📈 Configuration Graph (Excel-like)")

    base_graph_df = df.copy()

    c1, c2 = st.columns(2)
    with c1:
        g_product = st.selectbox("Product Name (required)", [""] + sorted(base_graph_df["Product Name"].dropna().unique())) if "Product Name" in base_graph_df.columns else ""
        if g_product:
            base_graph_df = base_graph_df[base_graph_df["Product Name"] == g_product]

        g_protection = st.selectbox("Protection Type (required)", [""] + sorted(base_graph_df["Protection Type"].dropna().unique())) if "Protection Type" in base_graph_df.columns else ""
        if g_protection:
            base_graph_df = base_graph_df[base_graph_df["Protection Type"] == g_protection]

    with c2:
        g_sw = st.selectbox("Software Version (required)", [""] + sorted(base_graph_df["SoftWare Version"].dropna().unique())) if "SoftWare Version" in base_graph_df.columns else ""
        if g_sw:
            base_graph_df = base_graph_df[base_graph_df["SoftWare Version"] == g_sw]

        g_ts = st.selectbox("Date & Time (required)", [""] + sorted(base_graph_df["Time Stamp"].dropna().unique(), reverse=True)) if "Time Stamp" in base_graph_df.columns else ""
        if g_ts:
            base_graph_df = base_graph_df[base_graph_df["Time Stamp"] == g_ts]

    can_plot = all([g_product, g_protection, g_sw, g_ts])

    with st.expander("Graph options"):
        show_markers = st.checkbox("Show markers", value=False)
        use_log_y = st.checkbox("Log scale (Y)", value=False)
        y_pad_pct = st.slider("Y padding (%)", min_value=0, max_value=30, value=5, step=1)

    if not can_plot:
        st.info("Select ALL required fields: Product Name, Protection Type, Software Version, Date & Time.")
        return

    required = {"Number", "W2P Measurement", "P2W Measurement"}
    missing = [c for c in required if c not in base_graph_df.columns]
    if missing:
        st.error(f"Missing required columns for graph: {missing}")
        return

    plot_df = base_graph_df.dropna(subset=["Number"]).sort_values("Number")
    w2p_y: pd.Series = plot_df["W2P Measurement"]
    p2w_y: pd.Series = plot_df["P2W Measurement"]

    y_min = pd.concat([w2p_y, p2w_y]).min()
    y_max = pd.concat([w2p_y, p2w_y]).max()
    if pd.isna(y_min) or pd.isna(y_max):
        st.warning("No valid measurements to plot for this configuration.")
        return

    pad = (y_max - y_min) * (y_pad_pct / 100.0) if y_max > y_min else 1
    y_range = [y_min - pad, y_max + pad]
    mode = "lines+markers" if show_markers else "lines"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["Number"], y=w2p_y, mode=mode, name="W2P (ms)",
        line=dict(width=2),
        marker=dict(size=4) if show_markers else None,
        connectgaps=True
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["Number"], y=p2w_y, mode=mode, name="P2W (ms)",
        line=dict(width=2),
        marker=dict(size=4) if show_markers else None,
        connectgaps=True
    ))

    fig.update_layout(
        title=dict(
            text=f"Disruption Protection-Working<br><sup>{g_product} | {g_protection} | {g_sw} | {g_ts}</sup>",
            x=0.5
        ),
        xaxis=dict(
            title="Cycle / Sample Number",
            showgrid=False,
            ticks="outside",
            tickangle=90,
            nticks=35
        ),
        yaxis=dict(
            title="Disruption Time (mSec)",
            showgrid=True,
            gridwidth=1
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=90, b=80),
    )

    fig.update_yaxes(showgrid=True, zeroline=False)
    fig.update_xaxes(showgrid=False)

    if use_log_y:
        fig.update_yaxes(type="log")
    else:
        fig.update_yaxes(range=y_range)

    fig.update_xaxes(rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show samples used for the graph"):
        st.dataframe(plot_df[["Number", "W2P Measurement", "P2W Measurement"]], use_container_width=True)


def render_results_table(display_df: pd.DataFrame, selected_columns: list[str]):
    st.divider()
    st.subheader(f"Showing {len(display_df)} Records")

    table_df = display_df[selected_columns].copy()
    col_cfg = build_column_config_for_autowidth(table_df)

    st.data_editor(
        table_df,
        use_container_width=True,
        hide_index=False,
        disabled=True,
        column_config=col_cfg
    )


def build_excel_bytes(display_df: pd.DataFrame, selected_columns: list[str], logo_path: str) -> bytes:
    export_df = display_df[selected_columns]
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        sheet_name = "APS Results"
        export_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=5)

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        if os.path.exists(logo_path):
            worksheet.insert_image("A1", logo_path, {"x_scale": 0.5, "y_scale": 0.5})

        title_format = workbook.add_format({"bold": True, "font_size": 16, "align": "left", "valign": "vcenter"})
        worksheet.write("A4", "PacketLight APS Disruption Time Results", title_format)

        header_format = workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#D9E1F2", "border": 1})
        for col_num, value in enumerate(export_df.columns.values):
            worksheet.write(5, col_num, value, header_format)

        cell_format = workbook.add_format({"align": "center", "valign": "vcenter", "border": 1})
        for row in range(len(export_df)):
            for col in range(len(export_df.columns)):
                worksheet.write(row + 6, col, export_df.iloc[row, col], cell_format)

        for i, col in enumerate(export_df.columns):
            max_len = max(export_df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)

        worksheet.freeze_panes(6, 0)

    output.seek(0)
    return output.getvalue()


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

# Sidebar + filtering
filters, selected_columns = sidebar_filters(df)
filtered_df = apply_filters(df, filters)

display_df = filtered_df.rename(columns=DISPLAY_COLUMNS_MAP)
selected_columns = [c for c in selected_columns if c in display_df.columns]

# Graph
render_config_graph(df)

# Table
render_results_table(display_df, selected_columns)

# Export
excel_bytes = build_excel_bytes(display_df, selected_columns, logo_path)
st.download_button(
    "Download Filtered Results - Excel File",
    data=excel_bytes,
    file_name="aps_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)