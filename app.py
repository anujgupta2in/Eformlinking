import streamlit as st
import pandas as pd
import re
import io
import altair as alt

st.set_page_config(page_title="PMS Live Vessel Checker", page_icon="🚢", layout="wide")

def highlight_count_gt_1(val):
    if isinstance(val, (int, float)) and val > 1:
        return 'background-color: #ffcccb; font-weight: bold'
    return ''

def style_eg_summary(df):
    return df.style.applymap(highlight_count_gt_1, subset=['EG_Count'])

def style_job_check(df):
    if 'Job_Linked' in df.columns:
        return df.style.applymap(highlight_count_gt_1, subset=['Job_Linked'])
    return df

EXCLUDE_OOM_VESSELS = True

def norm_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.upper()

def is_live(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"1", "yes", "y", "true", "live", "pms live", "active"}

def join_unique(series) -> str:
    vals = [str(v).strip() for v in series.dropna().unique() if str(v).strip() != ""]
    return " | ".join(vals)

def detect_critical(job_df: pd.DataFrame) -> pd.Series:
    possible_cols = [c for c in job_df.columns if str(c).strip().lower() in {
        "critical", "is critical", "critical flag", "critical_flag", "job critical"
    }]
    if possible_cols:
        col = possible_cols[0]
        def to_bool(x):
            if pd.isna(x): return False
            s = str(x).strip().lower()
            return s in {"1", "yes", "y", "true", "critical"}
        return job_df[col].apply(to_bool)

    fields = [c for c in ["Job Action", "Tag(s)", "Job Title", "Description"] if c in job_df.columns]
    def heuristic(row):
        text = " ".join([str(row.get(c, "")) for c in fields]).lower()
        return "critical" in text
    return job_df.apply(heuristic, axis=1)

def build_pms_live(jibe_file) -> pd.DataFrame:
    raw = pd.read_excel(jibe_file, sheet_name="Sheet1", header=1)
    
    new_cols = raw.iloc[0].tolist()
    df = raw.iloc[1:].copy()
    df.columns = [str(c).strip() for c in new_cols]

    pms_col = "PMS" if "PMS" in df.columns else next((c for c in df.columns if "pms" in c.lower()), None)
    if not pms_col:
        raise ValueError("Could not find PMS column in JiBe List file.")

    df["PMS_live"] = df[pms_col].apply(is_live)

    cols_keep = [c for c in ["Fleet", "Group Owner", "Vessel Name", "JiBe Go-Live Date", pms_col] if c in df.columns]
    pms_live = df.loc[df["PMS_live"], cols_keep].copy()
    pms_live = pms_live.rename(columns={pms_col: "PMS"})

    if EXCLUDE_OOM_VESSELS and "Vessel Name" in pms_live.columns:
        pms_live = pms_live[~pms_live["Vessel Name"].astype(str).str.contains(r"\(OOM\)", na=False)].copy()

    if "JiBe Go-Live Date" in pms_live.columns:
        pms_live["JiBe Go-Live Date"] = pd.to_datetime(pms_live["JiBe Go-Live Date"], errors="coerce").dt.date

    pms_live["Vessel_norm"] = pms_live["Vessel Name"].apply(norm_name)
    return pms_live.sort_values(["Fleet", "Vessel Name"], na_position="last")

def detect_machinery_type(emg_file, filename=None):
    emg_file.seek(0)
    eg = pd.read_csv(emg_file)
    emg_file.seek(0)
    
    machinery_map = {
        'MAC': ('Main Air Compressor', 'MAC'),
        'EG': ('Emergency Generator', 'EG'),
        'ME': ('Main Engine', 'ME'),
        'AE': ('Auxiliary Engine', 'AE'),
        'AB': ('Auxiliary Boiler', 'AB'),
        'MB': ('Main Boiler', 'MB'),
        'OWS': ('Oily Water Separator', 'OWS'),
        'IG': ('Inert Gas', 'IG'),
        'BWTS': ('Ballast Water Treatment System', 'BWTS'),
    }
    
    if filename:
        fn_upper = filename.upper()
        for code, (full_name, short) in machinery_map.items():
            if code in fn_upper.replace('_', ' ').replace('-', ' ').split():
                return full_name, short
            words = fn_upper.replace('_', ' ').replace('-', ' ')
            if f"_{code}_" in f"_{fn_upper}_" or fn_upper.startswith(f"{code}_") or fn_upper.endswith(f"_{code}"):
                return full_name, short
    
    if 'Machinery' in eg.columns:
        machinery_values = eg['Machinery'].dropna().astype(str).str.strip()
        if not machinery_values.empty:
            most_common = machinery_values.mode()
            if not most_common.empty:
                machinery_name = most_common.iloc[0]
                words = machinery_name.upper().replace(' ', '')
                for code, (full_name, short) in machinery_map.items():
                    if code in words or full_name.upper().replace(' ', '') in words:
                        return machinery_name, code
                short_code = ''.join([w[0] for w in machinery_name.split() if w])[:3].upper()
                return machinery_name, short_code
    
    return "Emergency Generator", "EG"

def build_emggen_final(emg_file) -> pd.DataFrame:
    eg = pd.read_csv(emg_file)

    required = ["Assigned to Vessels", "Machinery", "Maker", "Model", "Particulars"]
    missing = [c for c in required if c not in eg.columns]
    if missing:
        raise ValueError(f"Machinery List CSV missing columns: {missing}")

    eg_out = eg[required].copy()
    eg_out = eg_out.dropna(subset=["Assigned to Vessels"])
    eg_out = eg_out[eg_out["Assigned to Vessels"].astype(str).str.strip() != ""]

    eg_out["Vessel Name"] = eg_out["Assigned to Vessels"].astype(str).str.split(",")
    eg_out = eg_out.explode("Vessel Name")
    eg_out["Vessel Name"] = eg_out["Vessel Name"].astype(str).str.strip()

    eg_out = eg_out.rename(columns={"Machinery": "Machinery Name"})
    eg_out = eg_out[["Vessel Name", "Machinery Name", "Maker", "Model", "Particulars"]].drop_duplicates()

    eg_out["Vessel_norm"] = eg_out["Vessel Name"].apply(norm_name)
    return eg_out.reset_index(drop=True)

def check_emggen_coverage(pms_live: pd.DataFrame, eg_final: pd.DataFrame, machinery_name: str, machinery_short: str):
    eg_vessels = set(eg_final["Vessel_norm"].unique())
    pms_live = pms_live.copy()
    has_col = f"Has {machinery_name}"
    count_col = f"{machinery_short}_Count"
    pms_live[has_col] = pms_live["Vessel_norm"].apply(lambda x: x in eg_vessels)

    missing_eg = pms_live.loc[~pms_live[has_col],
                              ["Fleet", "Group Owner", "Vessel Name", "JiBe Go-Live Date"]].copy()

    details = pms_live.merge(
        eg_final[["Vessel_norm", "Machinery Name", "Maker", "Model", "Particulars"]],
        on="Vessel_norm",
        how="left"
    )

    summary = details.groupby(["Fleet", "Group Owner", "Vessel Name", "JiBe Go-Live Date", "PMS"], dropna=False).agg(
        **{count_col: ("Maker", lambda s: s.notna().sum())},
        Makers=("Maker", join_unique),
        Models=("Model", join_unique),
        Particulars=("Particulars", join_unique),
    ).reset_index()
    summary[has_col] = summary[count_col].gt(0)

    return summary, details.drop(columns=["Vessel_norm"]), missing_eg, count_col

def detect_job_code(job_file, filename=None):
    job = pd.read_csv(job_file)
    job_file.seek(0)
    
    if filename:
        import re
        match = re.search(r'(\d{3,5})', filename)
        if match:
            potential_code = match.group(1)
            if 'Job Code' in job.columns:
                if potential_code in job['Job Code'].astype(str).str.strip().values:
                    return potential_code
    
    if 'Job Code' in job.columns:
        job_codes = job['Job Code'].astype(str).str.strip()
        job_codes = job_codes[job_codes != '']
        if not job_codes.empty:
            return job_codes.mode().iloc[0] if not job_codes.mode().empty else job_codes.iloc[0]
    
    return None

def check_job_code(pms_live: pd.DataFrame, job_file, job_code: str):
    job = pd.read_csv(job_file)

    required_cols = {"Vessel", "Job Code", "Frequency", "Frequency Type"}
    missing = required_cols - set(job.columns)
    if missing:
        raise ValueError(f"Job List CSV missing required columns: {missing}")

    job["Vessel_norm"] = job["Vessel"].apply(norm_name)

    job_codes = job["Job Code"]
    job_filtered = job[job_codes.astype(str).str.strip().eq(str(job_code))].copy()

    job_filtered["Is Critical"] = detect_critical(job_filtered)

    agg = job_filtered.groupby("Vessel_norm").agg(
        Job_Linked=("Job Code", "count"),
        Frequency=("Frequency", join_unique),
        Frequency_Type=("Frequency Type", join_unique),
        Critical_Flag=("Is Critical", "any"),
    ).reset_index()

    base = pms_live[["Fleet", "Group Owner", "Vessel Name", "JiBe Go-Live Date", "PMS", "Vessel_norm"]].copy()
    check = base.merge(agg, on="Vessel_norm", how="left")
    check[f"Job {job_code} Linked"] = check["Job_Linked"].fillna(0).astype(int) > 0

    missing_job = check.loc[~check[f"Job {job_code} Linked"],
                            ["Fleet", "Group Owner", "Vessel Name", "JiBe Go-Live Date", "PMS"]].copy()

    return check.drop(columns=["Vessel_norm"]), missing_job, job_code

def check_eform_availability(pms_live: pd.DataFrame, job_file):
    job = pd.read_csv(job_file)
    
    eform_col = None
    for col in job.columns:
        col_normalized = re.sub(r'[^a-z0-9]', '', col.lower())
        if 'eform' in col_normalized and 'link' in col_normalized:
            eform_col = col
            break
    
    if not eform_col:
        for col in job.columns:
            col_normalized = re.sub(r'[^a-z0-9]', '', col.lower())
            if 'eform' in col_normalized:
                eform_col = col
                break
    
    if not eform_col:
        return None, None, None
    
    job["Vessel_norm"] = job["Vessel"].apply(norm_name)
    
    def has_eform(x):
        if pd.isna(x):
            return False
        s = str(x).strip()
        if s == "" or s.lower() in {"na", "n/a", "none", "-", "nan", "null"}:
            return False
        if s.lower().startswith("http") or "/" in s or len(s) > 10:
            return True
        return True
    
    job["Has_Eform"] = job[eform_col].apply(has_eform)
    
    eform_agg = job.groupby("Vessel_norm").agg(
        Eform_Available=("Has_Eform", "any"),
        Eform_Links=(eform_col, join_unique)
    ).reset_index()
    
    base = pms_live[["Fleet", "Group Owner", "Vessel Name", "JiBe Go-Live Date", "PMS", "Vessel_norm"]].copy()
    eform_check = base.merge(eform_agg, on="Vessel_norm", how="left")
    eform_check["Eform_Available"] = eform_check["Eform_Available"].fillna(False)
    
    eform_available = eform_check.loc[eform_check["Eform_Available"],
                                       ["Fleet", "Group Owner", "Vessel Name", "JiBe Go-Live Date", "PMS", "Eform_Links"]].copy()
    
    eform_missing = eform_check.loc[~eform_check["Eform_Available"],
                                     ["Fleet", "Group Owner", "Vessel Name", "JiBe Go-Live Date", "PMS"]].copy()
    
    return eform_check.drop(columns=["Vessel_norm"]), eform_available, eform_missing

def create_excel_output(pms_live, eg_final, eg_summary, eg_details, eg_missing, job_check, job_missing, job_code="Job", machinery_short="EG", eform_check=None, eform_available=None, eform_missing=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pms_live.drop(columns=["Vessel_norm"], errors="ignore").to_excel(writer, index=False, sheet_name="PMS_Live")
        eg_final.drop(columns=["Vessel_norm"], errors="ignore").to_excel(writer, index=False, sheet_name=f"{machinery_short}_Final")
        eg_summary.to_excel(writer, index=False, sheet_name=f"{machinery_short}_Summary")
        eg_details.to_excel(writer, index=False, sheet_name=f"{machinery_short}_Details")
        eg_missing.to_excel(writer, index=False, sheet_name=f"{machinery_short}_Missing_on_PMS")
        if not job_check.empty:
            job_check.to_excel(writer, index=False, sheet_name=f"Job{job_code}_Check")
        if not job_missing.empty:
            job_missing.to_excel(writer, index=False, sheet_name=f"Job{job_code}_Missing")
        if eform_check is not None:
            eform_check.to_excel(writer, index=False, sheet_name="Eform_Check")
        if eform_available is not None:
            eform_available.to_excel(writer, index=False, sheet_name="Eform_Available")
        if eform_missing is not None:
            eform_missing.to_excel(writer, index=False, sheet_name="Eform_Missing")
    output.seek(0)
    return output

st.title("🚢 PMS Live Vessel Checker")
st.markdown("Upload your files to process and generate the final vessel checklist report.")

st.header("📁 Upload Files")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Machinery List")
    machinery_file = st.file_uploader(
        "Upload Machinery List (CSV)",
        type=["csv"],
        key="machinery",
        help="CSV file with columns: Assigned to Vessels, Machinery, Maker, Model, Particulars"
    )

with col2:
    st.subheader("JiBe Live Vessels")
    jibe_file = st.file_uploader(
        "Upload JiBe List (Excel)",
        type=["xlsx", "xls"],
        key="jibe",
        help="Excel file with PMS live status for vessels"
    )

with col3:
    st.subheader("Job List Vessels")
    job_file = st.file_uploader(
        "Upload Job List (CSV)",
        type=["csv"],
        key="job",
        help="CSV file with columns: Vessel, Job Code, Frequency, Frequency Type"
    )

st.divider()

all_files_uploaded = machinery_file and jibe_file and job_file

if all_files_uploaded:
    if st.button("🔄 Process Files", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing files..."):
                pms_live = build_pms_live(jibe_file)
                machinery_name, machinery_short = detect_machinery_type(machinery_file, machinery_file.name if hasattr(machinery_file, 'name') else None)
                machinery_file.seek(0)
                eg_final = build_emggen_final(machinery_file)
                eg_summary, eg_details, eg_missing, count_col = check_emggen_coverage(pms_live, eg_final, machinery_name, machinery_short)
                job_file.seek(0)
                detected_job_code = detect_job_code(job_file, job_file.name if hasattr(job_file, 'name') else None)
                job_file.seek(0)
                if detected_job_code:
                    job_check, job_missing, job_code = check_job_code(pms_live, job_file, detected_job_code)
                else:
                    job_check, job_missing, job_code = None, None, None
                job_file.seek(0)
                eform_check, eform_available, eform_missing = check_eform_availability(pms_live, job_file)
                
                st.session_state['processed'] = True
                st.session_state['pms_live'] = pms_live
                st.session_state['eg_final'] = eg_final
                st.session_state['eg_summary'] = eg_summary
                st.session_state['eg_details'] = eg_details
                st.session_state['eg_missing'] = eg_missing
                st.session_state['job_check'] = job_check
                st.session_state['job_missing'] = job_missing
                st.session_state['job_code'] = job_code
                st.session_state['eform_check'] = eform_check
                st.session_state['eform_available'] = eform_available
                st.session_state['eform_missing'] = eform_missing
                st.session_state['machinery_name'] = machinery_name
                st.session_state['machinery_short'] = machinery_short
                st.session_state['count_col'] = count_col
                
            st.success("Files processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.session_state['processed'] = False

if st.session_state.get('processed', False):
    st.header("📊 Results Summary")
    
    pms_live = st.session_state['pms_live']
    eg_missing = st.session_state['eg_missing']
    job_missing = st.session_state['job_missing']
    eg_summary = st.session_state['eg_summary']
    job_check = st.session_state['job_check']
    job_code = st.session_state.get('job_code', 'Job')
    eform_check = st.session_state.get('eform_check')
    eform_available = st.session_state.get('eform_available')
    eform_missing = st.session_state.get('eform_missing')
    machinery_name = st.session_state.get('machinery_name', 'Machinery')
    machinery_short = st.session_state.get('machinery_short', 'EG')
    count_col = st.session_state.get('count_col', 'EG_Count')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("PMS Live Vessels", len(pms_live))
    with col2:
        st.metric(f"Missing {machinery_name}", len(eg_missing))
    with col3:
        if job_missing is not None:
            st.metric(f"Missing Job {job_code}", len(job_missing))
        else:
            st.metric("Missing Job", "N/A")
    with col4:
        if eform_available is not None:
            st.metric("Eform Available", len(eform_available))
        else:
            st.metric("Eform Available", "N/A")
    
    st.divider()
    
    st.header("📈 Visual Insights")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.subheader(f"{machinery_name} Count by Vessel")
        if count_col in eg_summary.columns:
            eg_chart_data = eg_summary[['Vessel Name', count_col]].copy()
            eg_chart_data = eg_chart_data[eg_chart_data[count_col] > 0].sort_values(count_col, ascending=False).head(20)
            eg_chart_data = eg_chart_data.rename(columns={count_col: 'Count'})
            
            if not eg_chart_data.empty:
                eg_chart = alt.Chart(eg_chart_data).mark_bar().encode(
                    x=alt.X('Count:Q', title=f'{machinery_name} Count'),
                    y=alt.Y('Vessel Name:N', sort='-x', title='Vessel'),
                    color=alt.condition(
                        alt.datum.Count > 1,
                        alt.value('#ff6b6b'),
                        alt.value('#4dabf7')
                    ),
                    tooltip=['Vessel Name', 'Count']
                ).properties(height=400)
                st.altair_chart(eg_chart, use_container_width=True)
            else:
                st.info(f"No vessels with {machinery_name} found.")
        else:
            st.info(f"No {machinery_name} data available.")
    
    with viz_col2:
        st.subheader(f"Job {job_code} Count by Vessel")
        if job_check is not None:
            job_chart_data = job_check[['Vessel Name', 'Job_Linked']].copy()
            job_chart_data['Job_Linked'] = job_chart_data['Job_Linked'].fillna(0).astype(int)
            job_chart_data = job_chart_data[job_chart_data['Job_Linked'] > 0].sort_values('Job_Linked', ascending=False).head(20)
            
            if not job_chart_data.empty:
                job_chart = alt.Chart(job_chart_data).mark_bar().encode(
                    x=alt.X('Job_Linked:Q', title='Job Count'),
                    y=alt.Y('Vessel Name:N', sort='-x', title='Vessel'),
                    color=alt.condition(
                        alt.datum.Job_Linked > 1,
                        alt.value('#ff6b6b'),
                        alt.value('#51cf66')
                    ),
                    tooltip=['Vessel Name', 'Job_Linked']
                ).properties(height=400)
                st.altair_chart(job_chart, use_container_width=True)
            else:
                st.info(f"No vessels with Job {job_code} linked found.")
        else:
            st.info("No job data available.")
    
    st.subheader("Frequency Distribution (Combined)")
    
    if job_check is not None:
        freq_data = job_check[['Frequency', 'Frequency_Type']].copy()
        
        combined_freqs = []
        for idx, row in freq_data.iterrows():
            freq_val = str(row['Frequency']) if pd.notna(row['Frequency']) else ''
            freq_type_val = str(row['Frequency_Type']) if pd.notna(row['Frequency_Type']) else ''
            
            freq_items = [f.strip() for f in freq_val.split('|') if f.strip()]
            type_items = [t.strip() for t in freq_type_val.split('|') if t.strip()]
            
            if freq_items and type_items:
                for i in range(max(len(freq_items), len(type_items))):
                    freq = freq_items[i] if i < len(freq_items) else freq_items[-1]
                    ftype = type_items[i] if i < len(type_items) else type_items[-1]
                    combined = f"{freq} {ftype}"
                    combined_freqs.append(combined)
            elif freq_items:
                for f in freq_items:
                    combined_freqs.append(f)
            elif type_items:
                for t in type_items:
                    combined_freqs.append(t)
        
        if combined_freqs:
            freq_combined_df = pd.DataFrame({'Frequency': combined_freqs})
            freq_counts = freq_combined_df.groupby('Frequency').size().reset_index(name='Count')
            freq_counts = freq_counts.sort_values('Count', ascending=False)
            
            freq_chart = alt.Chart(freq_counts).mark_bar().encode(
                x=alt.X('Frequency:N', sort='-y', title='Frequency'),
                y=alt.Y('Count:Q', title='Count'),
                color=alt.condition(
                    alt.datum.Count > 1,
                    alt.value('#ff6b6b'),
                    alt.value('#845ef7')
                ),
                tooltip=['Frequency', 'Count']
            ).properties(height=300)
            st.altair_chart(freq_chart, use_container_width=True)
        else:
            st.info("No frequency data available.")
    else:
        st.info("No job data available for frequency analysis.")
    
    st.divider()
    
    st.header("📋 Data Preview")
    
    tab_names = [
        "PMS Live", f"{machinery_short} Final", f"{machinery_short} Summary", f"{machinery_short} Details", 
        f"{machinery_short} Missing on PMS"
    ]
    if job_check is not None:
        tab_names.extend([f"Job {job_code} Check", f"Job {job_code} Missing"])
    if eform_check is not None:
        tab_names.extend(["Eform Check", "Eform Available", "Eform Missing"])
    
    tabs = st.tabs(tab_names)
    
    tab_idx = 0
    with tabs[tab_idx]:
        st.dataframe(st.session_state['pms_live'].drop(columns=["Vessel_norm"], errors="ignore"), use_container_width=True)
    tab_idx += 1
    with tabs[tab_idx]:
        st.dataframe(st.session_state['eg_final'].drop(columns=["Vessel_norm"], errors="ignore"), use_container_width=True)
    tab_idx += 1
    with tabs[tab_idx]:
        eg_sum_display = st.session_state['eg_summary'].copy()
        if count_col in eg_sum_display.columns:
            styled_eg = eg_sum_display.style.applymap(
                highlight_count_gt_1, 
                subset=[count_col]
            )
            st.dataframe(styled_eg, use_container_width=True)
        else:
            st.dataframe(eg_sum_display, use_container_width=True)
    tab_idx += 1
    with tabs[tab_idx]:
        st.dataframe(st.session_state['eg_details'], use_container_width=True)
    tab_idx += 1
    with tabs[tab_idx]:
        st.dataframe(st.session_state['eg_missing'], use_container_width=True)
    tab_idx += 1
    
    if job_check is not None:
        with tabs[tab_idx]:
            job_check_display = job_check.copy()
            job_check_display['Job_Linked'] = job_check_display['Job_Linked'].fillna(0).astype(int)
            styled_job = job_check_display.style.applymap(
                highlight_count_gt_1,
                subset=['Job_Linked']
            )
            st.dataframe(styled_job, use_container_width=True)
        tab_idx += 1
        with tabs[tab_idx]:
            if job_missing is not None:
                st.dataframe(job_missing, use_container_width=True)
            else:
                st.info("No missing job data.")
        tab_idx += 1
    
    if eform_check is not None:
        with tabs[tab_idx]:
            st.dataframe(eform_check, use_container_width=True)
        tab_idx += 1
        with tabs[tab_idx]:
            if eform_available is not None and not eform_available.empty:
                st.dataframe(eform_available, use_container_width=True)
            else:
                st.info("No vessels with Eform available.")
        tab_idx += 1
        with tabs[tab_idx]:
            if eform_missing is not None and not eform_missing.empty:
                st.dataframe(eform_missing, use_container_width=True)
            else:
                st.info("All vessels have Eform available.")
    
    st.divider()
    
    st.header("📥 Download Results")
    
    excel_output = create_excel_output(
        st.session_state['pms_live'],
        st.session_state['eg_final'],
        st.session_state['eg_summary'],
        st.session_state['eg_details'],
        st.session_state['eg_missing'],
        job_check if job_check is not None else pd.DataFrame(),
        job_missing if job_missing is not None else pd.DataFrame(),
        job_code if job_code else "Job",
        machinery_short,
        eform_check,
        eform_available,
        eform_missing
    )
    
    st.download_button(
        label="📥 Download Complete Excel Report",
        data=excel_output,
        file_name="PMS_Live_Final_Checks.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True
    )

else:
    st.info("👆 Please upload all three files above to begin processing.")
