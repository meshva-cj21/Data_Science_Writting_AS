# Road Safety Around the World — A Data-Driven Investigation

**Data Source:** [WHO Global Health Observatory — Road Safety](https://www.who.int/data/gho/data/themes/road-safety)

---

## File Structure

```
DATA_SCIENCE_WRITTING_AS/
│
├── data/                                                        # Raw WHO CSV datasets 
│   ├── applicability_of_national_motorcycle_helmet_law_to_all_occupants.csv
│   ├── applicability_of_seat_belt_to_all_occupants.csv
│   ├── attribution_of_road_traffic_deaths_to_alcohol.csv
│   ├── availability_of_funding_for_national_road_safety_strategy.csv
│   ├── blood_alcohol_concentration.csv
│   ├── definition_of_drink_driving_by_BAC.csv
│   ├── distribution_of_road_traffic_deaths_by_type_of_road_user.csv
│   ├── estimated_number_of_road_traffic_deaths.csv
│   ├── estimated_road_traffic_death_rate.csv
│   ├── existence_of_a_national_child_restraint_law.csv
│   ├── existence_of_a_national_road_safety_strategy.csv
│   ├── existence_of_a_road_safety_lead_agency.csv
│   ├── existence_of_a_universal_access_telephone_number.csv
│   ├── existence_of_national_drink_driving_law.csv
│   ├── existence_of_national_seat_belt_law.csv
│   ├── existence_of_national_speed_limits.csv
│   ├── law_requires_helmet_to_be_fastened.csv
│   ├── maximum_speed_limits.csv
│   ├── seat_belt_wearing_rate.csv
│   └── vehile_standards.csv
│
├── Road_Safety_Analysis.ipynb    # Main analysis notebook — hypotheses, stats, all visuals
├── app.py                        # Streamlit dashboard — interactive version of the analysis
├── merged_dataset.csv            # All 20 CSVs merged into one unified file (auto-generated)
└── README.md                     # This file
```

### Role of each file / folder

| File / Folder | Role |
|---|---|
| `data/` | The 20 raw WHO CSV files. Each file covers one road safety indicator (e.g. helmet laws, death rates, alcohol limits). These are the source of all analysis. |
| `Road_Safety_Analysis.ipynb` | The main analysis notebook. Loads and merges all 20 CSVs, tests 4 hypotheses with statistical methods, and produces all charts. |
| `app.py` | Streamlit dashboard. A fully interactive version of the notebook — same data, same hypotheses, but with filters, dropdowns, and live charts. |
| `merged_dataset.csv` | Created automatically the first time either the notebook or `app.py` runs. All 20 CSVs are merged into this single file. Once it exists, it is loaded directly instead of re-fetching everything. |

---

## How the Data Files Connect

All 20 CSV files share the same structure and are merged on a common `Country_Code` column into one long-format DataFrame. Each file contributes a different indicator (`Type`) to the merged dataset.

The key connections used in the analysis:

| Analysis | Data Files Used |
|---|---|
| Death rate (main outcome) | `estimated_road_traffic_death_rate.csv` |
| H1 — Income vs Death Rate | `estimated_road_traffic_death_rate.csv` + WHO region as income proxy |
| H2 — Helmet Laws | `applicability_of_national_motorcycle_helmet_law_to_all_occupants.csv` + death rate |
| H3 — Law Score | `existence_of_national_seat_belt_law.csv`, `existence_of_national_drink_driving_law.csv`, `existence_of_national_speed_limits.csv`, `existence_of_a_national_child_restraint_law.csv`, `applicability_of_national_motorcycle_helmet_law_to_all_occupants.csv` |
| H4 — Regional Inequality | `estimated_road_traffic_death_rate.csv` grouped by WHO region |
| Additional Insights | `seat_belt_wearing_rate.csv`, `attribution_of_road_traffic_deaths_to_alcohol.csv`, `distribution_of_road_traffic_deaths_by_type_of_road_user.csv` |

---

## Notebook — `Road_Safety_Analysis.ipynb`

### Where to run

We can run it on Google colab or VS code.

The notebook fetches all 20 WHO CSV files automatically from GitHub on first run. No manual data download is required.

---

## Dashboard — `app.py`

### Required packages

```bash
pip install streamlit plotly pandas numpy scipy requests pillow
```

### How to run

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

On first run, `app.py` fetches all 20 CSVs from GitHub and saves `merged_dataset.csv` locally. Every subsequent run loads from that local file instantly.
