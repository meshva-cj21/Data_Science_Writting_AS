import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("merged_dataset.csv")

countries = df['Country_Name'].unique()
selected_country = st.sidebar.selectbox("Select a Country", countries)

country_data = df[df['Country_Name'] == selected_country]
# print(country_data)

# Title
st.title(f"Road Safety Profile: {selected_country}")
st.markdown("---")

# === SECTION 1: Road Safety Snapshot ===
st.header("\U0001F4CA Road Safety Snapshot")

# Basic metrics to display
metrics_to_display = [
    "Existence of a national road safety strategy",
    "Existence of a road safety lead agency",
    "Availability of funding for national road safety strategy",
    "Existence of national speed limits",
    "Existence of a national seat-belt law",
    "Existence of a national drink-driving law",
    "Existence of a national child-restraint law"
]

cols = st.columns(2)
for i, indicator in enumerate(metrics_to_display):
    val = country_data[country_data['Type'] == indicator]['Value'].values
    # print(val)
    if(len(val) == 0):
        val = "No data"
    else:
        val = val[0] if val[0] != "-" else "No data"
    cols[i % 2].metric(indicator, val)

st.markdown("---")

# SECTION 2: Seatbelt Wearing Rate & Helmet Law

st.subheader("🚗 Seat-belt Wearing Rate and Helmet Law")

# --- Seat-belt Wearing Rate ---
seatbelt_df = country_data[country_data['Type'] == 'Seat-belt wearing rate (%)']
seatbelt_df = seatbelt_df[['Data_Description', 'Value']].dropna()

st.markdown("**Seat-belt Wearing Rate (%):**")

if not seatbelt_df.empty:
    seatbelt_metrics = seatbelt_df.set_index('Data_Description')['Value'].to_dict()
    col1, col2 = st.columns(2)
    for i, (label, value) in enumerate(seatbelt_metrics.items()):
        col = col1 if i % 2 == 0 else col2
        try:
            value_float = float(value)
            col.metric(label=label, value=f"{value_float:.1f}%")
        except:
            col.metric(label=label, value="No data")
else:
    st.info("No seat-belt wearing data available for this country.")

st.markdown("---")

# --- Helmet Law Applicability ---
helmet_df = country_data[country_data['Type'].str.contains("helmet law", case=False, na=False)]
helmet_df = helmet_df[['Data_Description', 'Value']].dropna()
# print(helmet_df)

st.markdown("**Helmet Law Applicability:**")

if not helmet_df.empty:
    col1, col2 = st.columns(2)
    for i, row in helmet_df.iterrows():
        col = col1 if i % 2 == 0 else col2
        label = row['Data_Description']
        value = row['Value']
        if value.strip() in ['—', '-', 'Not restricted', 'No data', '', 'NaN']:
            value = "Not applicable / No data"
        col.metric(label=label, value=value)
else:
    st.info("No helmet law data available for this country.")


st.markdown("---")

# === SECTION 3: Road User Death Distribution ===
st.header("\U0001F480 Road User Death Distribution")
death_dist = country_data[country_data['Type'] == 
    "Distribution of road traffic deaths by type of road user (%)"]

if not death_dist.empty:
    pie_df = death_dist[['Data_Description', 'Value']].copy()
    pie_df.columns = ['Road User Type', 'Percentage']
    pie_df['Percentage'] = pie_df['Percentage'].str.replace('%', '').astype(float)
    pie_df['Percentage'] = pie_df['Percentage'] / 100
    pie_df = pie_df.sort_values(by='Percentage', ascending=False)
    
    # print(pie_df)
    fig = px.pie(pie_df, names='Road User Type', values='Percentage', title='Deaths by Road User Type')
    st.plotly_chart(fig)
else:
    st.info("No data available for death distribution.")

st.markdown("---")

# === SECTION 4: Speed & Alcohol Limits ===
st.header("\U0001F699 Speed & Alcohol Limits")
speed_limits = country_data[country_data['Type'] == "Maximum speed limits"]
bac_limits = country_data[country_data['Type'] == "Blood Alcohol Concentration (BAC) limit for drivers"]

st.subheader("Speed Limits")
for _, row in speed_limits.iterrows():
    st.write(f"**{row['Data_Description']}**: {row['Value'] if row['Value'] != '?' else 'No Data'} km/h")

st.subheader("Blood Alcohol Concentration (BAC) Limits")
for _, row in bac_limits.iterrows():
    st.write(f"**{row['Data_Description']}**: {row['Value'] if row['Value'] != '-' else 'No Data'}")

st.markdown("---")

# === SECTION 5: Estimated Deaths ===
st.header("\U0001F5E3 Estimated Road Traffic Deaths")
death_est = country_data[country_data['Type'].str.contains("Estimated number of road traffic deaths")]
death_rate = country_data[country_data['Type'].str.contains("death rate")]

col1, col2 = st.columns(2)
col1.metric("Estimated Deaths", death_est['Value'].values[0] if not death_est.empty else "No data")
col2.metric("Death Rate per 100,000", death_rate['Value'].values[0] if not death_rate.empty else "No data")

st.markdown("---")

st.caption("Data Source: WHO Road Safety Global Status Report")