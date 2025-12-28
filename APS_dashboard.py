import streamlit as st
st.set_page_config(page_title="APS Disruption Time Results", layout="wide", initial_sidebar_state="expanded")

import os
import io
import pandas as pd
from sqlalchemy import create_engine
from PIL import Image

# =========================================
# CONFIG
# =========================================
DB_FILENAME = "APS_data_base2.db"               
MAIN_TABLE = 'Disruption Time Measurement'     
LOGO_FILENAME = "Packetlight Logo.png"         

# --- DB Connection ---
DB_PATH = os.path.join(os.path.dirname(__file__), DB_FILENAME)
# DB_PATH = r"G:\Python\PacketLight Automation\Test_Cases\General Tests\APS Tests\APS_data_base2.db"
engine = create_engine(f"sqlite:///{DB_PATH}")

@st.cache_data
def load_data():
    # Read all columns from the main table (+ rowid for uniqueness)
    df = pd.read_sql(f'SELECT rowid as _rowid_, * FROM "{MAIN_TABLE}"', engine)

    # Normalize timestamp column (your screenshot shows "Time Stamp")
    if "Time Stamp" in df.columns:
        # Example: "16:48:51 10-08-2025" (time then date)
        parsed = pd.to_datetime(df["Time Stamp"], errors="coerce", dayfirst=True)
        if parsed.notna().any():
            df["Time Stamp"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")

    # Convert measurements to numeric
    for c in ["W2P Measurement", "P2W Measurement"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure "Number" is numeric
    if "Number" in df.columns:
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce")

    # Order columns nicely (keep only those that exist)
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
    """
    Estimate a good column width (in px) based on the longest string in each column.
    """
    cfg = {}
    for col in df.columns:
        # compute max length among header + values
        s = df[col].astype(str).fillna("")
        max_len = max([len(str(col))] + s.map(len).tolist())
        width_px = int(max_len * px_per_char + 24)  # +padding
        width_px = max(min_px, min(max_px, width_px))
        cfg[col] = st.column_config.Column(width=width_px)
    return cfg

df = load_data()

# =========================================
# HEADER
# =========================================
logo_path = os.path.join(os.path.dirname(__file__), LOGO_FILENAME)
if os.path.exists(logo_path):
    st.image(Image.open(logo_path), width=250)

st.title("PacketLight - APS Disruption Time Results")
st.subheader("(W2P / P2W Disruption Time Measurements)")

# =========================================
# DISPLAY COLUMN NAMES
# =========================================
display_columns_map = {
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
# SIDEBAR FILTERS (interdependent)
# =========================================
with st.sidebar:
    st.subheader("Contact: Yuval Dahan")
    st.header("🔍 Filters")

    filtered_options_df = df.copy()

    # Product Name
    selected_product = []
    if "Product Name" in filtered_options_df.columns:
        selected_product = st.multiselect(
            "Product Name",
            sorted(filtered_options_df["Product Name"].dropna().unique())
        )
        if selected_product:
            filtered_options_df = filtered_options_df[filtered_options_df["Product Name"].isin(selected_product)]

    # Protection Type
    selected_protection = []
    if "Protection Type" in filtered_options_df.columns:
        selected_protection = st.multiselect(
            "Protection Type",
            sorted(filtered_options_df["Protection Type"].dropna().unique())
        )
        if selected_protection:
            filtered_options_df = filtered_options_df[filtered_options_df["Protection Type"].isin(selected_protection)]

    # Software Version
    selected_sw = []
    if "SoftWare Version" in filtered_options_df.columns:
        selected_sw = st.multiselect(
            "Software Version",
            sorted(filtered_options_df["SoftWare Version"].dropna().unique())
        )
        if selected_sw:
            filtered_options_df = filtered_options_df[filtered_options_df["SoftWare Version"].isin(selected_sw)]

    # System Mode
    selected_mode = []
    if "System Mode" in filtered_options_df.columns:
        selected_mode = st.multiselect(
            "System Mode",
            sorted(filtered_options_df["System Mode"].dropna().unique())
        )
        if selected_mode:
            filtered_options_df = filtered_options_df[filtered_options_df["System Mode"].isin(selected_mode)]

    # Uplink Service Type
    selected_uplink = []
    if "Uplink Service Type" in filtered_options_df.columns:
        selected_uplink = st.multiselect(
            "Uplink Service Type",
            sorted(filtered_options_df["Uplink Service Type"].dropna().unique())
        )
        if selected_uplink:
            filtered_options_df = filtered_options_df[filtered_options_df["Uplink Service Type"].isin(selected_uplink)]

    # Client Service Type
    selected_client = []
    if "Client Service Type" in filtered_options_df.columns:
        selected_client = st.multiselect(
            "Client Service Type",
            sorted(filtered_options_df["Client Service Type"].dropna().unique())
        )
        if selected_client:
            filtered_options_df = filtered_options_df[filtered_options_df["Client Service Type"].isin(selected_client)]

    # Transceiver PN
    selected_transceiver_pn = []
    if "Transceiver PN" in filtered_options_df.columns:
        selected_transceiver_pn = st.multiselect(
            "Transceiver PN",
            sorted(filtered_options_df["Transceiver PN"].dropna().unique())
        )
        if selected_transceiver_pn:
            filtered_options_df = filtered_options_df[filtered_options_df["Transceiver PN"].isin(selected_transceiver_pn)]

    # Transceiver FW
    selected_transceiver_fw = []
    if "Transceiver FW" in filtered_options_df.columns:
        selected_transceiver_fw = st.multiselect(
            "Transceiver FW",
            sorted(filtered_options_df["Transceiver FW"].dropna().unique())
        )
        if selected_transceiver_fw:
            filtered_options_df = filtered_options_df[filtered_options_df["Transceiver FW"].isin(selected_transceiver_fw)]

    # --- Filter by sample number ---
    st.header("🆔 Filter by Sample Number")
    number_input = st.text_input("Enter sample numbers (comma-separated)", value="")
    number_list = []
    if number_input.strip():
        number_list = [int(x.strip()) for x in number_input.split(",") if x.strip().isdigit()]

    # --- Measurement filters ---
    st.header("⏱️ W2P Filter")
    w2p_filter_type = st.radio("Filter W2P:", ["Show All", "Above", "Below"], horizontal=True, key="w2p_radio")
    w2p_threshold = st.number_input("W2P Threshold", min_value=0.0, step=0.1, key="w2p_thr")

    st.header("⏱️ P2W Filter")
    p2w_filter_type = st.radio("Filter P2W:", ["Show All", "Above", "Below"], horizontal=True, key="p2w_radio")
    p2w_threshold = st.number_input("P2W Threshold", min_value=0.0, step=0.1, key="p2w_thr")

    # --- Column toggles ---
    st.header("🧩 Columns to Display")
    st.caption("Toggle columns on/off to display in the table:")

    display_df_preview = df.rename(columns=display_columns_map)
    checkbox_columns = {}
    for col in display_df_preview.columns:
        checkbox_columns[col] = st.checkbox(col, value=True)
    selected_columns = [col for col, show in checkbox_columns.items() if show]

# =========================================
# APPLY FILTERS
# =========================================
filtered_df = df.copy()

if selected_product and "Product Name" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Product Name"].isin(selected_product)]
if selected_protection and "Protection Type" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Protection Type"].isin(selected_protection)]
if selected_sw and "SoftWare Version" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["SoftWare Version"].isin(selected_sw)]
if selected_mode and "System Mode" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["System Mode"].isin(selected_mode)]
if selected_uplink and "Uplink Service Type" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Uplink Service Type"].isin(selected_uplink)]
if selected_client and "Client Service Type" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Client Service Type"].isin(selected_client)]
if selected_transceiver_pn and "Transceiver PN" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Transceiver PN"].isin(selected_transceiver_pn)]
if selected_transceiver_fw and "Transceiver FW" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Transceiver FW"].isin(selected_transceiver_fw)]

if number_list and "Number" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Number"].isin(number_list)]

# W2P filter
if "W2P Measurement" in filtered_df.columns:
    if w2p_filter_type == "Above":
        filtered_df = filtered_df[filtered_df["W2P Measurement"] > w2p_threshold]
    elif w2p_filter_type == "Below":
        filtered_df = filtered_df[filtered_df["W2P Measurement"] < w2p_threshold]

# P2W filter
if "P2W Measurement" in filtered_df.columns:
    if p2w_filter_type == "Above":
        filtered_df = filtered_df[filtered_df["P2W Measurement"] > p2w_threshold]
    elif p2w_filter_type == "Below":
        filtered_df = filtered_df[filtered_df["P2W Measurement"] < p2w_threshold]

# Rename columns for display
display_df = filtered_df.rename(columns=display_columns_map)

# Ensure selected columns exist (in case user toggled something odd)
selected_columns = [c for c in selected_columns if c in display_df.columns]

# =========================================
# DISPLAY RESULTS
# =========================================
st.subheader(f"Showing {len(display_df)} Records")
# st.dataframe(display_df[selected_columns], use_container_width=True)
table_df = display_df[selected_columns].copy()

col_cfg = build_column_config_for_autowidth(table_df)

st.data_editor(
    table_df,
    use_container_width=True,
    hide_index=False,
    disabled=True,          # read-only
    column_config=col_cfg
)


# =========================================
# DOWNLOAD EXCEL 
# =========================================
export_df = display_df[selected_columns]
output = io.BytesIO()

with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    sheet_name = "APS Results"
    export_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=5)

    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    # Insert PacketLight Logo
    if os.path.exists(logo_path):
        worksheet.insert_image("A1", logo_path, {"x_scale": 0.5, "y_scale": 0.5})

    # Title
    title_format = workbook.add_format({
        "bold": True,
        "font_size": 16,
        "align": "left",
        "valign": "vcenter"
    })
    worksheet.write("A4", "PacketLight APS Disruption Time Results", title_format)

    # Header formatting
    header_format = workbook.add_format({
        "bold": True,
        "align": "center",
        "valign": "vcenter",
        "bg_color": "#D9E1F2",
        "border": 1
    })
    for col_num, value in enumerate(export_df.columns.values):
        worksheet.write(5, col_num, value, header_format)

    # Cell formatting
    cell_format = workbook.add_format({
        "align": "center",
        "valign": "vcenter",
        "border": 1
    })
    for row in range(len(export_df)):
        for col in range(len(export_df.columns)):
            worksheet.write(row + 6, col, export_df.iloc[row, col], cell_format)

    # Auto-fit column widths
    for i, col in enumerate(export_df.columns):
        max_len = max(export_df[col].astype(str).map(len).max(), len(col)) + 2
        worksheet.set_column(i, i, max_len)

    # Freeze header
    worksheet.freeze_panes(6, 0)

output.seek(0)

st.download_button(
    "Download Filtered Results - Excel File",
    data=output,
    file_name="aps_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)