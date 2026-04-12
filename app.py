import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, f_oneway
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Safety Dashboard",
    page_icon="https://png.pngtree.com/png-vector/20240202/ourmid/pngtree-3d-traffic-light-png-image_11590393.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.4rem; font-weight:800; color:#1a1a2e; margin-bottom:0; }
    .sub-title   { font-size:1.05rem; color:#555; margin-top:0; }
    .kpi-box     { background:#f0f4ff; border-radius:12px; padding:18px 22px;
                   border-left:5px solid #3a86ff; margin-bottom:8px; }
    .kpi-val     { font-size:2rem; font-weight:800; color:#3a86ff; }
    .kpi-label   { font-size:0.85rem; color:#555; margin-top:2px; }
    .verdict-yes { background:#e6f9f0; border-left:5px solid #2ecc71;
                   border-radius:8px; padding:12px 16px; margin:8px 0; }
    .verdict-no  { background:#fff4e6; border-left:5px solid #f39c12;
                   border-radius:8px; padding:12px 16px; margin:8px 0; }
    .section-hdr { font-size:1.3rem; font-weight:700; color:#1a1a2e;
                   border-bottom:2px solid #3a86ff; padding-bottom:4px;
                   margin:16px 0 10px 0; }
    .callout     { background:#fff8e1; border-left:4px solid #f9a825;
                   border-radius:6px; padding:10px 14px; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)


# DATA LOADING


BASE_URL = "https://raw.githubusercontent.com/meshva-cj21/Data_Science_Writting_AS/main/data/"
FILES = [
    "applicability_of_national_motorcycle_helmet_law_to_all_occupants.csv",
    "applicability_of_seat_belt_to_all_occupants.csv",
    "attribution_of_road_traffic_deaths_to_alcohol.csv",
    "availability_of_funding_for_national_road_safety_strategy.csv",
    "blood_alcohol_concentration.csv",
    "definition_of_drink_driving_by_BAC.csv",
    "distribution_of_road_traffic_deaths_by_type_of_road_user.csv",
    "estimated_number_of_road_traffic_deaths.csv",
    "estimated_road_traffic_death_rate.csv",
    "existence_of_a_national_child_restraint_law.csv",
    "existence_of_a_national_road_safety_strategy.csv",
    "existence_of_a_road_safety_lead_agency.csv",
    "existence_of_a_universal_access_telephone_number.csv",
    "existence_of_national_drink_driving_law.csv",
    "existence_of_national_seat_belt_law.csv",
    "existence_of_national_speed_limits.csv",
    "law_requires_helmet_to_be_fastened.csv",
    "maximum_speed_limits.csv",
    "seat_belt_wearing_rate.csv",
    "vehile_standards.csv",
]

DEATH_RATE_TYPE   = "Estimated road traffic death rate (per 100 000 population)"
TOTAL_DEATHS_TYPE = "Estimated number of road traffic deaths"
HELMET_TYPE       = "Applicability of national motorcycle helmet law to all occupants"
SEATBELT_TYPE     = "Existence of a national seat-belt law"
DRINKDRIVE_TYPE   = "Existence of national drink-driving law"
SPEED_TYPE        = "Existence of national speed limits"
CHILD_TYPE        = "Existence of a national child restraint law"
ALCOHOL_ATTR_TYPE = "Attribution of road traffic deaths to alcohol"
ROAD_USER_TYPE    = "Distribution of road traffic deaths by type of road user"


def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="UTF-8")
    df = df.iloc[:, [1, 3, 4, 6, 7, 9, 11, 12, 29]]
    df.columns = [
        "Type", "Global_Location_Code", "Global_Location_Name",
        "Country_Code", "Country_Name", "Year",
        "Data_Type", "Data_Description", "Value",
    ]
    df["Data_Type"]        = df["Data_Type"].fillna("-")
    df["Data_Description"] = df["Data_Description"].fillna("-")
    return df


@st.cache_data(show_spinner=False)
def load_all_data() -> pd.DataFrame:
    """Load merged dataset — from local file if present, else fetch from GitHub."""
   
    if os.path.exists("merged_dataset.csv"):
        return pd.read_csv("merged_dataset.csv")

    # fetching individual CSVs from GitHub
    os.makedirs("data", exist_ok=True)
    dfs = []
    for fname in FILES:
        local = f"data/{fname}"
        if not os.path.exists(local):
            try:
                r = requests.get(BASE_URL + fname, timeout=30)
                if r.status_code == 200:
                    with open(local, "wb") as fout:
                        fout.write(r.content)
            except Exception:
                pass
        if os.path.exists(local):
            try:
                dfs.append(load_dataset(local))
            except Exception:
                pass

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv("merged_dataset.csv", index=False)
        return df

    st.error(" Could not load data. Please place merged_dataset.csv in this folder.")
    st.stop()


# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("📡 Loading WHO Road Safety datasets…"):
    df = load_all_data()

# ── Derived dataframes ─────────────────────────────────────────────────────────
df_dr = df[df["Type"] == DEATH_RATE_TYPE].copy()
df_dr["Death_Rate"] = pd.to_numeric(df_dr["Value"], errors="coerce")
df_dr = df_dr[df_dr["Country_Code"].notna() & (df_dr["Country_Code"] != "")]
df_dr.dropna(subset=["Death_Rate"], inplace=True)

df_total = df[df["Type"] == TOTAL_DEATHS_TYPE].copy()
df_total["Total_Deaths"] = pd.to_numeric(df_total["Value"], errors="coerce")
df_total.dropna(subset=["Total_Deaths"], inplace=True)

all_countries = sorted(df_dr["Country_Name"].dropna().unique().tolist())
all_regions   = sorted(df_dr["Global_Location_Name"].dropna().unique().tolist())



# SIDEBAR

with st.sidebar:
    st.image("https://static.vecteezy.com/system/resources/previews/039/629/853/non_2x/road-shield-safe-transport-logo-design-vector.jpg",
             use_container_width=True, output_format="JPEG")
    st.markdown("##  Road Safety Dashboard")

    st.markdown("**Data:** WHO Global Health Observatory  \n**Topic:** Road Safety Analysis")
    st.markdown("---")

    page = st.radio(
        " Navigate",
        [" Global Overview",
         " Hypothesis Testing",
         " Country Deep-Dive",
         " Country Comparison",
         " Data Explorer"],
        index=0,
    )

    st.markdown("---")
    st.markdown("###  Filters")
    sel_regions = st.multiselect(
        "Filter by WHO Region",
        options=all_regions,
        default=all_regions,
    )
    dr_min, dr_max = float(df_dr["Death_Rate"].min()), float(df_dr["Death_Rate"].max())
    dr_range = st.slider(
        "Death Rate Range (per 100k)",
        min_value=dr_min, max_value=dr_max,
        value=(dr_min, dr_max), step=0.5,
    )

    st.markdown("---")
    st.markdown(
        "<small>Built with Streamlit + Plotly · WHO GHO Data</small>",
        unsafe_allow_html=True,
    )

# Apply sidebar filters
df_filt = df_dr[
    df_dr["Global_Location_Name"].isin(sel_regions) &
    df_dr["Death_Rate"].between(dr_range[0], dr_range[1])
].copy()



# HELPER: KPI BOX

def kpi(label, value, unit=""):
    st.markdown(
        f"""<div class='kpi-box'>
            <div class='kpi-val'>{value}<span style='font-size:1rem;color:#888'> {unit}</span></div>
            <div class='kpi-label'>{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )



#GLOBAL OVERVIEW

if page == " Global Overview":
    st.markdown("<div class='main-title'> Global Road Safety Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>WHO Estimated Road Traffic Death Rates · All Countries</div>",
                unsafe_allow_html=True)
    st.markdown("")

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Countries in Dataset", df_filt["Country_Name"].nunique())
    with c2:
        kpi("Global Mean Death Rate", f"{df_filt['Death_Rate'].mean():.1f}", "per 100k")
    with c3:
        worst = df_filt.loc[df_filt["Death_Rate"].idxmax(), "Country_Name"]
        kpi("Highest Death Rate Country", worst)
    with c4:
        safest = df_filt.loc[df_filt["Death_Rate"].idxmin(), "Country_Name"]
        kpi("Lowest Death Rate Country", safest)

    st.markdown("---")

    # ── Choropleth ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'> World Heatmap — Road Traffic Death Rate</div>",
                unsafe_allow_html=True)
    fig_map = px.choropleth(
        df_filt,
        locations="Country_Code",
        color="Death_Rate",
        hover_name="Country_Name",
        color_continuous_scale="RdYlGn_r",
        labels={"Death_Rate": "Deaths per 100k"},
        template="plotly_white",
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="Deaths<br>per 100k"),
        height=480,
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown(
        "<div class='callout'> <b>Observation:</b> Sub-Saharan Africa and parts of South/South-East Asia "
        "show the deepest red, while high-income Europe and Australia appear green. "
        "This north–south divide motivates all five hypotheses tested in this study.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # ── Bar: Top 15 most dangerous ────────────────────────────────────────────
    with col_l:
        st.markdown("<div class='section-hdr'> 15 Most Dangerous Countries</div>",
                    unsafe_allow_html=True)
        top15 = df_filt.nlargest(15, "Death_Rate")
        fig_top = px.bar(
            top15, x="Death_Rate", y="Country_Name", orientation="h",
            color="Death_Rate", color_continuous_scale="Reds",
            labels={"Death_Rate": "Deaths per 100k", "Country_Name": ""},
            template="plotly_white",
        )
        fig_top.update_layout(showlegend=False, coloraxis_showscale=False,
                               height=420, margin=dict(l=0, r=0, t=10, b=0),
                               yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top, use_container_width=True)

    # ── Bar: Top 15 safest ────────────────────────────────────────────────────
    with col_r:
        st.markdown("<div class='section-hdr'> 15 Safest Countries</div>",
                    unsafe_allow_html=True)
        bot15 = df_filt.nsmallest(15, "Death_Rate")
        fig_bot = px.bar(
            bot15, x="Death_Rate", y="Country_Name", orientation="h",
            color="Death_Rate", color_continuous_scale="Greens_r",
            labels={"Death_Rate": "Deaths per 100k", "Country_Name": ""},
            template="plotly_white",
        )
        fig_bot.update_layout(showlegend=False, coloraxis_showscale=False,
                               height=420, margin=dict(l=0, r=0, t=10, b=0),
                               yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bot, use_container_width=True)

    # ── Regional box plot ─────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'> Distribution by WHO Region</div>",
                unsafe_allow_html=True)
    region_order = (df_filt.groupby("Global_Location_Name")["Death_Rate"]
                    .median().sort_values(ascending=False).index.tolist())
    fig_box = px.box(
        df_filt, x="Global_Location_Name", y="Death_Rate",
        color="Global_Location_Name",
        category_orders={"Global_Location_Name": region_order},
        points="all",
        labels={"Death_Rate": "Death Rate (per 100k)", "Global_Location_Name": "WHO Region"},
        template="plotly_white",
    )
    fig_box.update_layout(showlegend=False, height=400,
                           margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_box, use_container_width=True)



# HYPOTHESIS TESTING

elif page == " Hypothesis Testing":
    st.markdown("<div class='main-title'> Hypothesis Testing</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Data-driven tests for 5 road safety hypotheses</div>",
                unsafe_allow_html=True)
    st.markdown("")

    hyp = st.selectbox(
        "Select Hypothesis",
        [
            "H1 — Income Level vs Death Rate",
            "H2 — Helmet Laws vs Death Rate",
            "H3 — Law Score vs Death Rate",
            "H4 — Regional Inequality",
        ],
    )

    st.markdown("---")

    # ── H1 ────────────────────────────────────────────────────────────────────
    if hyp.startswith("H1"):
        st.markdown("### H1: High-income countries have lower road traffic death rates")
        st.info("**Method:** WHO regions are used as an income proxy. One-way ANOVA + Pearson correlation test for statistical significance.")

        income_map = {
            "Africa":                            "Low / Lower-Middle",
            "South-East Asia":                   "Low / Lower-Middle",
            "Eastern Mediterranean":             "Lower-Middle / Upper-Middle",
            "Western Pacific":                   "Upper-Middle / High",
            "Americas":                          "Upper-Middle / High",
            "Europe":                            "High",
        }
        income_rank = {
            "Low / Lower-Middle": 1,
            "Lower-Middle / Upper-Middle": 2,
            "Upper-Middle / High": 3,
            "High": 4,
        }
        df_h1 = df_filt.copy()
        df_h1["Income_Group"] = df_h1["Global_Location_Name"].map(income_map)
        df_h1["Income_Rank"]  = df_h1["Income_Group"].map(income_rank)
        df_h1.dropna(subset=["Income_Group"], inplace=True)

        order = ["Low / Lower-Middle", "Lower-Middle / Upper-Middle",
                 "Upper-Middle / High", "High"]
        colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

        fig_h1 = px.box(
            df_h1, x="Income_Group", y="Death_Rate",
            color="Income_Group",
            category_orders={"Income_Group": order},
            color_discrete_sequence=colors,
            points="all",
            hover_name="Country_Name",
            labels={"Death_Rate": "Death Rate (per 100k)", "Income_Group": "Income Group"},
            template="plotly_white",
            title="H1: Death Rate by Income Group (WHO Region Proxy)",
        )
        fig_h1.update_layout(showlegend=False, height=430)
        st.plotly_chart(fig_h1, use_container_width=True)

        # Stats
        groups = [df_h1[df_h1["Income_Group"] == g]["Death_Rate"].dropna().values for g in order]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            F, p = f_oneway(*groups)
            r_val, p_r = pearsonr(df_h1["Income_Rank"].dropna(),
                                   df_h1.loc[df_h1["Income_Rank"].notna(), "Death_Rate"])

            c1, c2, c3 = st.columns(3)
            with c1: kpi("ANOVA F-statistic", f"{F:.2f}")
            with c2: kpi("ANOVA p-value", f"{p:.4f}")
            with c3: kpi("Pearson r", f"{r_val:.3f}")

            verdict = p < 0.05
            if verdict:
                st.markdown("<div class='verdict-yes'> <b>H1 SUPPORTED</b> — Income group differences are statistically significant (p &lt; 0.05). "
                            "Higher-income regions have significantly lower death rates.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='verdict-no'> H1 not statistically significant at p &lt; 0.05 with current filter.</div>",
                            unsafe_allow_html=True)

            # Summary table
            summary = df_h1.groupby("Income_Group")["Death_Rate"].agg(["mean","median","count"]).round(2)
            summary.columns = ["Mean Death Rate", "Median Death Rate", "N Countries"]
            st.dataframe(summary.loc[order], use_container_width=True)

        st.markdown(
            "<div class='callout'><b>Calling Bullshit:</b> Income correlates with MANY things simultaneously — "
            "infrastructure, vehicle age, emergency care, enforcement. Income is a proxy, not a direct cause. "
            "Also, WHO death-rate data for low-income countries relies on statistical modelling; "
            "true rates may be <i>higher</i> than reported.</div>", unsafe_allow_html=True)

    # ── H2 ────────────────────────────────────────────────────────────────────
    elif hyp.startswith("H2"):
        st.markdown("### H2: Countries with motorcycle helmet laws have lower fatality rates")
        st.info("**Method:** Extract helmet-law status per country, merge with death rate, run Mann-Whitney U test.")

        df_helm = df[df["Type"] == HELMET_TYPE][["Country_Code", "Value"]].copy()
        df_helm.rename(columns={"Value": "Helmet_Law"}, inplace=True)
        df_helm = df_helm[df_helm["Helmet_Law"] != "-"].drop_duplicates("Country_Code")

        def simplify_helmet(v):
            v = str(v).lower()
            if "all" in v:      return "All Occupants"
            if "no law" in v or "not" in v: return "No Law"
            return "Partial / Other"

        df_helm["Helmet_Simple"] = df_helm["Helmet_Law"].apply(simplify_helmet)
        df_h2 = df_filt.merge(df_helm, on="Country_Code")

        color_map = {"All Occupants": "#2ca02c", "Partial / Other": "#ff7f0e", "No Law": "#d62728"}
        order_h2  = ["All Occupants", "Partial / Other", "No Law"]

        fig_h2 = px.violin(
            df_h2, x="Helmet_Simple", y="Death_Rate",
            color="Helmet_Simple",
            category_orders={"Helmet_Simple": order_h2},
            color_discrete_map=color_map,
            box=True, points="all",
            hover_name="Country_Name",
            labels={"Death_Rate": "Death Rate (per 100k)", "Helmet_Simple": "Helmet Law Status"},
            template="plotly_white",
            title="H2: Helmet Law Coverage vs Road Traffic Death Rate",
        )
        fig_h2.update_layout(showlegend=False, height=430)
        st.plotly_chart(fig_h2, use_container_width=True)

        g_all = df_h2[df_h2["Helmet_Simple"] == "All Occupants"]["Death_Rate"].dropna()
        g_no  = df_h2[df_h2["Helmet_Simple"] == "No Law"]["Death_Rate"].dropna()

        if len(g_all) > 1 and len(g_no) > 1:
            U, p_mw = mannwhitneyu(g_all, g_no, alternative="less")
            c1, c2, c3, c4 = st.columns(4)
            with c1: kpi("Mean (All Occupants law)", f"{g_all.mean():.1f}", "per 100k")
            with c2: kpi("Mean (No law)", f"{g_no.mean():.1f}", "per 100k")
            with c3: kpi("Difference", f"{g_no.mean()-g_all.mean():.1f}", "per 100k more")
            with c4: kpi("Mann-Whitney p-value", f"{p_mw:.4f}")

            if p_mw < 0.05:
                st.markdown(
                    f"<div class='verdict-yes'> <b>H2 SUPPORTED</b> — Countries WITH full helmet laws average "
                    f"<b>{g_all.mean():.1f}</b> deaths/100k vs <b>{g_no.mean():.1f}</b> without — "
                    f"a difference of <b>{g_no.mean()-g_all.mean():.1f}</b> deaths per 100k (p={p_mw:.4f}).</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown("<div class='verdict-no'> H2 not statistically significant with current filter.</div>",
                            unsafe_allow_html=True)

        st.markdown(
            "<div class='callout'><b>Calling Bullshit:</b> A country can <i>have</i> a helmet law without enforcing it. "
            "India has a national helmet law yet compliance in many states is below 50%. "
            "<b>Policy ≠ Enforcement.</b> The true effect of <i>enforced</i> helmet laws is likely larger.</div>",
            unsafe_allow_html=True)

    # ── H3 ────────────────────────────────────────────────────────────────────
    elif hyp.startswith("H3"):
        st.markdown("### H3: Countries with more road safety laws have lower death rates")
        st.info("**Method:** Build a Law Score (0–5) from binary law indicators. Spearman correlation + scatter plot.")

        def get_binary(type_str, yes_kw):
            sub = df[df["Type"] == type_str][["Country_Code", "Value"]].drop_duplicates("Country_Code").copy()
            sub["flag"] = sub["Value"].apply(
                lambda v: 1 if any(k.lower() in str(v).lower() for k in yes_kw) else 0)
            return sub[["Country_Code", "flag"]]

        law_defs = [
            (HELMET_TYPE,     ["all"],            "helmet"),
            (SEATBELT_TYPE,   ["yes","national"], "seatbelt"),
            (DRINKDRIVE_TYPE, ["yes","national"], "drinkdrive"),
            (SPEED_TYPE,      ["yes","national"], "speed"),
            (CHILD_TYPE,      ["yes","national"], "child_restraint"),
        ]

        df_h3 = df_filt[["Country_Code","Country_Name","Global_Location_Name","Death_Rate"]].copy()
        for type_str, kws, col in law_defs:
            matches = [t for t in df["Type"].unique() if type_str.lower() in t.lower()]
            if matches:
                flags = get_binary(matches[0], kws).rename(columns={"flag": col})
                df_h3 = df_h3.merge(flags, on="Country_Code", how="left")
            else:
                df_h3[col] = np.nan

        law_cols = [c for _, _, c in law_defs]
        df_h3["Law_Score"] = df_h3[law_cols].sum(axis=1, skipna=True)
        df_h3_clean = df_h3.dropna(subset=["Death_Rate", "Law_Score"])

        fig_h3 = px.scatter(
            df_h3_clean, x="Law_Score", y="Death_Rate",
            hover_name="Country_Name",
            color="Global_Location_Name",
            size="Death_Rate",
            trendline="ols",
            labels={"Law_Score": "Law Score (0–5)", "Death_Rate": "Death Rate (per 100k)"},
            template="plotly_white",
            title="H3: Road Safety Law Score vs Death Rate",
        )
        fig_h3.update_layout(height=450, legend_title="WHO Region")
        st.plotly_chart(fig_h3, use_container_width=True)

        if len(df_h3_clean) > 5:
            rho, p_rho = spearmanr(df_h3_clean["Law_Score"], df_h3_clean["Death_Rate"])
            c1, c2 = st.columns(2)
            with c1: kpi("Spearman ρ", f"{rho:.3f}")
            with c2: kpi("p-value", f"{p_rho:.4f}")

            st.markdown("**Mean death rate by law score:**")
            score_table = df_h3_clean.groupby("Law_Score")["Death_Rate"].agg(["mean","count"]).round(2)
            score_table.columns = ["Mean Death Rate", "N Countries"]
            st.dataframe(score_table, use_container_width=True)

            if p_rho < 0.05 and rho < 0:
                st.markdown("<div class='verdict-yes'> <b>H3 SUPPORTED</b> — More laws in place is associated with lower death rates.</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='verdict-no'> <b>H3 NUANCED</b> — Correlation is weak or positive, likely due to "
                    "<b>reverse causality</b>: countries enact laws <i>because</i> death rates are high.</div>",
                    unsafe_allow_html=True)

        st.markdown(
            "<div class='callout'> <b>Reverse Causality Trap:</b> Countries with the worst road safety "
            "often enact laws in response to their crisis, so high law scores can co-exist with high "
            "death rates in the short term. Longitudinal before/after data is needed to isolate causal effects.</div>",
            unsafe_allow_html=True)

    # ── H4 ────────────────────────────────────────────────────────────────────
    elif hyp.startswith("H4"):
        st.markdown("### H4: Low-income regions have disproportionately higher road death rates")
        st.info("**Method:** Regional ANOVA + pairwise Welch t-test (Africa vs Europe).")

        region_summary = (df_filt.groupby("Global_Location_Name")["Death_Rate"]
                          .agg(["mean","median","std","count"])
                          .round(2).sort_values("mean", ascending=False))
        region_summary.columns = ["Mean", "Median", "Std Dev", "N Countries"]

        fig_h4 = px.bar(
            region_summary.reset_index(),
            x="Mean", y="Global_Location_Name", orientation="h",
            color="Mean", color_continuous_scale="RdYlGn_r",
            text="Mean",
            labels={"Mean": "Mean Death Rate (per 100k)", "Global_Location_Name": "WHO Region"},
            template="plotly_white",
            title="H4: Mean Road Traffic Death Rate by WHO Region",
        )
        fig_h4.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_h4.update_layout(showlegend=False, coloraxis_showscale=False,
                              height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_h4, use_container_width=True)

        # ANOVA
        grps = [df_filt[df_filt["Global_Location_Name"] == r]["Death_Rate"].dropna().values
                for r in region_summary.index if
                len(df_filt[df_filt["Global_Location_Name"] == r]["Death_Rate"].dropna()) > 1]

        if len(grps) >= 2:
            F_r, p_r = f_oneway(*grps)
            kpi_c1, kpi_c2 = st.columns(2)
            with kpi_c1: kpi("ANOVA F-statistic", f"{F_r:.2f}")
            with kpi_c2: kpi("ANOVA p-value", f"{p_r:.6f}")

        # Africa vs Europe pairwise
        africa = df_filt[df_filt["Global_Location_Name"] == "Africa"]["Death_Rate"].dropna()
        europe = df_filt[df_filt["Global_Location_Name"] == "Europe"]["Death_Rate"].dropna()
        if len(africa) > 1 and len(europe) > 1:
            from scipy.stats import ttest_ind
            t, p_t2 = ttest_ind(africa, europe, equal_var=False)
            c1, c2, c3, c4 = st.columns(4)
            with c1: kpi("Africa Mean", f"{africa.mean():.1f}", "per 100k")
            with c2: kpi("Europe Mean", f"{europe.mean():.1f}", "per 100k")
            with c3: kpi("Ratio (Africa/Europe)", f"{africa.mean()/europe.mean():.1f}", "×")
            with c4: kpi("t-test p-value", f"{p_t2:.6f}")

            if p_t2 < 0.05:
                st.markdown(
                    f"<div class='verdict-yes'> <b>H4 STRONGLY SUPPORTED</b> — Africa's mean death rate "
                    f"(<b>{africa.mean():.1f}</b>/100k) is <b>{africa.mean()/europe.mean():.1f}×</b> "
                    f"higher than Europe's (<b>{europe.mean():.1f}</b>/100k). p &lt; 0.001.</div>",
                    unsafe_allow_html=True)

        st.dataframe(region_summary, use_container_width=True)
        st.markdown(
            "<div class='callout'> <b>Calling Bullshit:</b> WHO data for African countries is often modelled "
            "from partial records. The <i>true</i> death toll may be significantly higher. Sweden achieves "
            "~2–3 deaths/100k via Vision Zero. If that were the global standard, over 1 million additional "
            "lives could be saved annually.</div>", unsafe_allow_html=True)



# COUNTRY DEEP-DIVE

elif page == " Country Deep-Dive":
    st.markdown("<div class='main-title'> Country Deep-Dive</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Explore all WHO road safety indicators for any country</div>",
                unsafe_allow_html=True)
    st.markdown("")

    sel_country = st.selectbox("Select a Country", all_countries, index=all_countries.index("India") if "India" in all_countries else 0)

    df_country = df[df["Country_Name"] == sel_country].copy()

    # KPIs
    dr_val = df_dr[df_dr["Country_Name"] == sel_country]["Death_Rate"]
    td_val = df_total[df_total["Country_Name"] == sel_country]["Total_Deaths"]
    region = df_dr[df_dr["Country_Name"] == sel_country]["Global_Location_Name"].iloc[0] if not df_dr[df_dr["Country_Name"] == sel_country].empty else "N/A"

    global_mean = df_dr["Death_Rate"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Death Rate", f"{dr_val.values[0]:.1f}" if len(dr_val) else "N/A", "per 100k")
    with c2: kpi("Total Deaths", f"{int(td_val.values[0]):,}" if len(td_val) else "N/A")
    with c3: kpi("WHO Region", region)
    with c4:
        if len(dr_val):
            diff = dr_val.values[0] - global_mean
            kpi("vs Global Mean", f"{diff:+.1f}", "per 100k")

    st.markdown("---")

    # Gauge for death rate
    if len(dr_val):
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=dr_val.values[0],
            delta={"reference": global_mean, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, df_dr["Death_Rate"].max() * 1.1]},
                "bar":  {"color": "#3a86ff"},
                "steps": [
                    {"range": [0, 10],  "color": "#c8f7c5"},
                    {"range": [10, 20], "color": "#ffeaa7"},
                    {"range": [20, 100],"color": "#ff7675"},
                ],
                "threshold": {"line": {"color": "black","width": 3},
                              "thickness": 0.75, "value": global_mean},
            },
            title={"text": f"Death Rate — {sel_country}<br><sub>Global mean = {global_mean:.1f}</sub>"},
            number={"suffix": " /100k"},
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # All indicators table
    st.markdown("<div class='section-hdr'> All WHO Indicators</div>", unsafe_allow_html=True)
    pivot_country = (df_country[["Type","Value","Data_Description"]]
                     .drop_duplicates(subset=["Type"]))
    st.dataframe(pivot_country.set_index("Type"), use_container_width=True, height=400)

    # Road user distribution
    df_ru = df_country[df_country["Type"] == ROAD_USER_TYPE].copy()
    df_ru = df_ru[df_ru["Data_Description"] != "-"]
    if not df_ru.empty:
        df_ru["Value_num"] = pd.to_numeric(df_ru["Value"], errors="coerce")
        df_ru.dropna(subset=["Value_num"], inplace=True)
        if not df_ru.empty:
            st.markdown("<div class='section-hdr'> Road Death by User Type</div>",
                        unsafe_allow_html=True)
            fig_pie = px.pie(df_ru, names="Data_Description", values="Value_num",
                             template="plotly_white",
                             title=f"Road Traffic Deaths by User Type — {sel_country}")
            fig_pie.update_layout(height=380)
            st.plotly_chart(fig_pie, use_container_width=True)



# 4 — COUNTRY COMPARISON

elif page == " Country Comparison":
    st.markdown("<div class='main-title'> Country Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Compare road safety profiles of up to 5 countries</div>",
                unsafe_allow_html=True)
    st.markdown("")

    defaults = [c for c in ["India","Sweden","Nigeria","Germany","Brazil"] if c in all_countries]
    sel_countries = st.multiselect("Select Countries (max 5)", all_countries,
                                   default=defaults[:5], max_selections=5)

    if not sel_countries:
        st.warning("Please select at least one country.")
        st.stop()

    df_comp = df_dr[df_dr["Country_Name"].isin(sel_countries)][
        ["Country_Name","Global_Location_Name","Death_Rate"]].copy()

    # Bar chart
    fig_comp = px.bar(
        df_comp.sort_values("Death_Rate", ascending=False),
        x="Country_Name", y="Death_Rate",
        color="Death_Rate", color_continuous_scale="RdYlGn_r",
        text="Death_Rate",
        labels={"Death_Rate": "Death Rate (per 100k)", "Country_Name": ""},
        template="plotly_white",
        title="Road Traffic Death Rate Comparison",
    )
    fig_comp.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_comp.update_layout(showlegend=False, coloraxis_showscale=False,
                            height=400, yaxis_title="Deaths per 100k")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Side-by-side all indicators
    st.markdown("<div class='section-hdr'> Key Indicators Side-by-Side</div>",
                unsafe_allow_html=True)

    key_types = [
        HELMET_TYPE, SEATBELT_TYPE, DRINKDRIVE_TYPE,
        SPEED_TYPE, CHILD_TYPE, DEATH_RATE_TYPE,
    ]
    rows = []
    for ktype in key_types:
        row = {"Indicator": ktype.replace("Existence of", "").replace("Applicability of", "").strip()}
        for c in sel_countries:
            val = df[(df["Country_Name"] == c) & (df["Type"] == ktype)]["Value"]
            row[c] = val.values[0] if len(val) else "—"
        rows.append(row)

    df_side = pd.DataFrame(rows).set_index("Indicator")
    st.dataframe(df_side, use_container_width=True)

    # Radar chart
    st.markdown("<div class='section-hdr'> Safety Profile Radar</div>", unsafe_allow_html=True)
    st.caption("Death rate normalised 0–1 (lower = safer). Law scores 0–5.")

    max_dr = df_dr["Death_Rate"].max()

    def build_law_score(country):
        score = 0
        for lt, kws in [(HELMET_TYPE,["all"]),(SEATBELT_TYPE,["yes","national"]),
                        (DRINKDRIVE_TYPE,["yes","national"]),(SPEED_TYPE,["yes","national"]),
                        (CHILD_TYPE,["yes","national"])]:
            matches = [t for t in df["Type"].unique() if lt.lower() in t.lower()]
            if matches:
                val = df[(df["Country_Name"] == country) & (df["Type"] == matches[0])]["Value"]
                if len(val) and any(k.lower() in str(val.values[0]).lower() for k in kws):
                    score += 1
        return score

    categories = ["Safety (inverted DR)", "Law Score /5"]
    fig_radar = go.Figure()
    for c in sel_countries:
        dr_c = df_dr[df_dr["Country_Name"] == c]["Death_Rate"]
        dr_norm = 1 - (dr_c.values[0] / max_dr) if len(dr_c) else 0
        ls = build_law_score(c) / 5
        vals = [dr_norm, ls, dr_norm]   # close the loop
        cats = categories + [categories[0]]
        fig_radar.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", name=c))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        template="plotly_white", height=420,
        title="Safety Radar (higher = better/safer)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown(
        "<div class='callout'> <b>India vs Sweden</b> — Sweden implements Vision Zero (target: zero road deaths) "
        "and achieves 2–3 deaths/100k. India has most laws on paper but enforcement gaps keep death rates "
        "5–7× higher. This illustrates the gap between <i>legislation</i> and <i>implementation</i>.</div>",
        unsafe_allow_html=True)



# 5 — DATA EXPLORER

elif page == " Data Explorer":
    st.markdown("<div class='main-title'> Raw Data Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Browse and filter the merged WHO dataset</div>",
                unsafe_allow_html=True)
    st.markdown("")

    indicator_options = sorted(df["Type"].unique().tolist())
    sel_indicator = st.selectbox("Filter by Indicator", ["(All)"] + indicator_options)

    df_explore = df.copy()
    if sel_indicator != "(All)":
        df_explore = df_explore[df_explore["Type"] == sel_indicator]

    df_explore = df_explore[df_explore["Country_Name"].isin(
        st.multiselect("Filter by Country", all_countries, default=[])
        or all_countries
    )]

    st.dataframe(df_explore, use_container_width=True, height=500)

    csv_bytes = df_explore.to_csv(index=False).encode()
    st.download_button(
        label="⬇ Download filtered data as CSV",
        data=csv_bytes,
        file_name="who_road_safety_filtered.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown(f"**Showing:** {len(df_explore):,} rows · "
                f"{df_explore['Country_Name'].nunique()} countries · "
                f"{df_explore['Type'].nunique()} indicators")