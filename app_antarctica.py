"""
Antarctica FCL Freight Rate Forecasting Dashboard
==================================================
Standalone Streamlit app for the Antarctica → World lanes model.
Tabs: Overview | Seasonality | Model with COVID Forecast | Stable Model Forecast |
      Validation (Jan25–Apr26) | Model Comparison | Data Sources
"""

import os, json, warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
OUT  = os.path.join(BASE, 'outputs', 'antarctica')

# ── Persistence helpers ───────────────────────────────────────────────────────
PERSIST_FILE = os.path.join(DATA, 'xfactor_overrides.json')

XFACTOR_DEFAULTS = {
    # Macro variable overrides (live values Apr 2026)
    'brent_crude':        74.0,   # USD/bbl
    'usdcny':              7.28,  # USD/CNY
    'bdry_etf':           10.4,   # BDRY ETF price
    # Conflict zone toggles (0 = inactive, 1 = active)
    'dummy_ukraine':       1,
    'dummy_red_sea':       1,
    'dummy_hormuz':        0,
    'dummy_panama':        0,
    'dummy_covid':         0,
    'dummy_supply_crunch': 0,
    # Developer coefficient multipliers (1.0 = model default)
    'coef_brent':    1.0,
    'coef_usdcny':   1.0,
    'coef_bdry':     1.0,
    'coef_ukraine':  1.0,
    'coef_red_sea':  1.0,
}

def load_overrides():
    if os.path.exists(PERSIST_FILE):
        try:
            with open(PERSIST_FILE) as f:
                saved = json.load(f)
            merged = dict(XFACTOR_DEFAULTS)
            merged.update(saved)
            return merged
        except Exception:
            pass
    return dict(XFACTOR_DEFAULTS)

def save_overrides(ov):
    os.makedirs(os.path.dirname(PERSIST_FILE), exist_ok=True)
    with open(PERSIST_FILE, 'w') as f:
        json.dump(ov, f, indent=2)

# ── Lane-differentiated conflict impact matrix ─────────────────────────────────
# Scale: 1.0 = full impact as learned from training data
#        0.0 = no impact (route does not use this chokepoint)
# Based on: ITF-OECD 2024, EIA 2023, UNCTAD 2024

# Region classification for each lane
LANE_REGION = {
    'CNSHA': 'east_asia',    'JPTYO': 'east_asia',    'KRPUS': 'east_asia',
    'THBKK': 'se_asia',      'INNSA': 'south_asia',
    'DEHAM': 'europe',       'NLRTM': 'europe',        'GBFXT': 'europe',
    'USNYC': 'americas',     'USLAX': 'americas',      'ARBUE': 'americas',
    'AEJEA': 'middle_east',  'QAHMD': 'middle_east',
    'NGLOS': 'west_africa',
    'AUSYD': 'oceania',
}

# Red Sea / Suez impact by region
# Logic: Europe/Med = 1.0 (primary Suez users), Americas = 0.15 (trans-Pacific/Atlantic)
RED_SEA_SCALE = {
    'europe':      1.0,   # Primary Suez users; full rerouting cost
    'middle_east': 0.9,   # Adjacent to disruption; direct surcharge
    'east_africa': 0.9,   # Red Sea is primary northbound corridor
    'south_asia':  0.7,   # Suez preferred; Cape adds ~6 days
    'se_asia':     0.5,   # Mixed routes; some trans-Pacific, some Suez
    'east_asia':   0.4,   # Trans-Pacific routes unaffected; only Europe-bound via Suez
    'west_africa': 0.3,   # Cape route already common; partial impact
    'oceania':     0.2,   # Indian Ocean/trans-Pacific; Suez rarely used
    'americas':    0.15,  # Trans-Pacific/Atlantic; Suez not primary
}

# Ukraine / Black Sea impact by region
UKRAINE_SCALE = {
    'europe':      0.8,   # Direct energy cost impact; port disruption risk
    'middle_east': 0.4,   # Indirect via oil/gas market
    'south_asia':  0.3,
    'se_asia':     0.2,
    'east_asia':   0.2,
    'west_africa': 0.15,
    'oceania':     0.1,
    'americas':    0.1,
}

# Panama Canal impact by region
PANAMA_SCALE = {
    'americas':    0.8,   # Direct trans-Pacific/Atlantic impact
    'east_asia':   0.4,   # Trans-Pacific routes affected
    'se_asia':     0.3,
    'oceania':     0.3,
    'south_asia':  0.1,
    'europe':      0.1,
    'middle_east': 0.05,
    'west_africa': 0.05,
    'east_africa': 0.05,
}

def get_lane_impact_scales(lane):
    """Return conflict impact scales for a given lane based on its region."""
    region = LANE_REGION.get(lane, 'east_asia')
    return {
        'red_sea': RED_SEA_SCALE.get(region, 0.3),
        'ukraine': UKRAINE_SCALE.get(region, 0.2),
        'panama':  PANAMA_SCALE.get(region, 0.1),
    }

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    'bg':        '#F7F9FC',
    'card':      '#FFFFFF',
    'border':    '#D0D7E3',
    'navy':      '#0A1628',
    'blue':      '#1B3A6B',
    'teal':      '#0E6B8C',
    'red':       '#8B1A1A',
    'green':     '#1A5C2A',
    'amber':     '#B8860B',
    'purple':    '#4B0082',
    'grey':      '#5A6478',
    'lightgrey': '#E8ECF2',
    'high':      '#1A5C2A',
    'medium':    '#B8860B',
    'low':       '#8B1A1A',
}

LANE_NAMES = {
    'CNSHA': 'China – Shanghai',    'JPTYO': 'Japan – Tokyo',
    'KRPUS': 'South Korea – Busan', 'INNSA': 'India – Nhava Sheva',
    'THBKK': 'Thailand – Bangkok',  'DEHAM': 'Germany – Hamburg',
    'NLRTM': 'Netherlands – Rotterdam', 'GBFXT': 'UK – Felixstowe',
    'USNYC': 'USA – New York',      'USLAX': 'USA – Los Angeles',
    'ARBUE': 'Argentina – Buenos Aires', 'AEJEA': 'UAE – Jebel Ali',
    'NGLOS': 'Nigeria – Lagos',     'AUSYD': 'Australia – Sydney',
    'QAHMD': 'Qatar – Hamad',
}

DEMO_LANES = list(LANE_NAMES.keys())

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Antarctica FCL Forecast",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
      background-color: #F7F9FC !important; color: #0A1628 !important;
  }
  [data-testid="stSidebar"] { background-color: #FFFFFF !important; }
  [data-testid="stSidebar"] * { color: #0A1628 !important; }
  h1,h2,h3,h4,h5,h6 { color: #0A1628 !important; font-weight: 700 !important; }
  p, li, span, div, label { color: #0A1628 !important; }
  .stSelectbox label, .stSlider label, .stRadio label, .stToggle label {
      color: #0A1628 !important; font-weight: 600 !important;
  }
  [data-testid="stMetricLabel"] {
      color: #1B3A6B !important; font-weight: 700 !important;
      font-size: 0.78rem !important; text-transform: uppercase;
  }
  [data-testid="stMetricValue"] { color: #0A1628 !important; font-weight: 800 !important; font-size: 1.6rem !important; }
  [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #D0D7E3; border-radius: 8px; padding: 12px 16px; }
  .stTabs [data-baseweb="tab"] { color: #1B3A6B !important; font-weight: 600; }
  .stTabs [aria-selected="true"] { border-bottom: 3px solid #1B3A6B !important; color: #0A1628 !important; }
  .stDataFrame { border: 1px solid #D0D7E3 !important; border-radius: 6px; }
  .stButton > button { background-color: #1B3A6B !important; color: #FFFFFF !important; font-weight: 700; border-radius: 6px; border: none; }
  .stButton > button:hover { background-color: #0A1628 !important; }
  .btn-neutral > button { background-color: #5A6478 !important; }
  .btn-reset  > button { background-color: #8B1A1A !important; }
  .reliability-high   { background:#E8F5E9; color:#1A5C2A; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.82rem; }
  .reliability-medium { background:#FFF8E1; color:#B8860B; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.82rem; }
  .reliability-low    { background:#FFEBEE; color:#8B1A1A; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.82rem; }
  .info-box { background:#EEF4FF; border-left:4px solid #1B3A6B; padding:12px 16px; border-radius:0 6px 6px 0; margin:8px 0; }
  .info-box p { color:#0A1628 !important; margin:0; font-size:0.9rem; }
  .warn-box { background:#FFF8E1; border-left:4px solid #B8860B; padding:12px 16px; border-radius:0 6px 6px 0; margin:8px 0; }
  .warn-box p { color:#0A1628 !important; margin:0; font-size:0.9rem; }
  .formula-box { background:#F0F4FF; border:1px solid #B0C4DE; border-radius:8px; padding:14px 18px; margin:10px 0; font-family:monospace; color:#0A1628 !important; font-size:0.95rem; font-weight:600; }
  .impact-tag-high   { background:#FFEBEE; color:#8B1A1A; padding:2px 8px; border-radius:10px; font-size:0.78rem; font-weight:700; }
  .impact-tag-medium { background:#FFF8E1; color:#B8860B; padding:2px 8px; border-radius:10px; font-size:0.78rem; font-weight:700; }
  .impact-tag-low    { background:#E8F5E9; color:#1A5C2A; padding:2px 8px; border-radius:10px; font-size:0.78rem; font-weight:700; }
  .winner-badge { background:#1A5C2A; color:#FFFFFF; padding:2px 8px; border-radius:10px; font-size:0.78rem; font-weight:700; }
  .section-header { font-size:0.72rem; font-weight:800; color:#1B3A6B !important; text-transform:uppercase; letter-spacing:0.08em; margin:12px 0 4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_panel():
    return pd.read_parquet(os.path.join(DATA, 'antarctica_monthly_panel.parquet'))

@st.cache_data(ttl=3600)
def load_forecasts():
    fp = os.path.join(OUT, 'all_forecasts_antarctica.json')
    if os.path.exists(fp):
        with open(fp) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=3600)
def load_validation(path_tag):
    fp = os.path.join(OUT, f'{path_tag}_val_table.csv')
    if os.path.exists(fp):
        return pd.read_csv(fp)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_backtest(path_tag, lane):
    # Use v2.0 backtest for path1, cap10 normalised validation for path2
    if path_tag == 'path1':
        fp = os.path.join(OUT, f'path1v2_backtest_{lane}.csv')
    elif path_tag == 'path2':
        fp = os.path.join(OUT, f'path2_cap10_validation_{lane}.csv')
    else:
        fp = os.path.join(OUT, f'{path_tag}_backtest_{lane}.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        # Normalise column names: stk_pred -> stacked_pred, sar_pred -> sarimax_pred
        df = df.rename(columns={'stk_pred': 'stacked_pred', 'sar_pred': 'sarimax_pred'})
        return df
    return pd.DataFrame()
@st.cache_data(ttl=3600)
def load_val_csv(path_tag, lane):
    # Use v2.0 validation for path1, cap10 normalised for path2
    if path_tag == 'path1':
        fp = os.path.join(OUT, f'path1v2_validation_{lane}.csv')
    elif path_tag == 'path2':
        fp = os.path.join(OUT, f'path2_cap10_validation_{lane}.csv')
    else:
        fp = os.path.join(OUT, f'{path_tag}_validation_{lane}.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        # Normalise column names: stk_pred -> stacked_pred, sar_pred -> sarimax_pred
        df = df.rename(columns={'stk_pred': 'stacked_pred', 'sar_pred': 'sarimax_pred'})
        return df
    return pd.DataFrame()
@st.cache_data(ttl=3600)
def load_comparison():
    fp = os.path.join(OUT, 'path_comparison.csv')
    if os.path.exists(fp):
        return pd.read_csv(fp)
    return pd.DataFrame()

def rel_badge(rel):
    cls = {'High':'reliability-high','Medium':'reliability-medium','Low':'reliability-low'}.get(rel,'reliability-low')
    return f'<span class="{cls}">{rel}</span>'

def impact_tag(scale):
    if scale >= 0.7:
        return f'<span class="impact-tag-high">High ({scale:.0%})</span>'
    elif scale >= 0.35:
        return f'<span class="impact-tag-medium">Medium ({scale:.0%})</span>'
    else:
        return f'<span class="impact-tag-low">Low ({scale:.0%})</span>'

def compute_live_forecast(path_tag, lane, ov, horizon=6):
    """
    Re-run a live forecast using the user's X-factor overrides.
    Loads from the correct bundle file (path1_final_models.pkl / path2_final_models.pkl)
    and builds the correct feature matrices for both model components.
    Returns dict with 'dates', 'sarimax_pred', 'xgb_corr', 'stacked_pred' or None.
    """
    import traceback
    model_dir  = os.path.join(BASE, 'models', 'antarctica')
    # Use v2.0 bundle for path1 (lane-specific FX + Maersk proxy)
    bundle_name = 'path1v2_final_models.pkl' if path_tag == 'path1' else f'{path_tag}_final_models.pkl'
    bundle_file = os.path.join(model_dir, bundle_name)
    exog_file  = os.path.join(DATA, 'exog_features_antarctica.parquet')
    panel_file = os.path.join(DATA, 'antarctica_monthly_panel.parquet')

    if not os.path.exists(bundle_file) or not os.path.exists(exog_file):
        return None

    try:
        # ── Load model bundle ──────────────────────────────────────────────────
        with open(bundle_file, 'rb') as f:
            all_models = pickle.load(f)

        if lane not in all_models:
            return None

        m          = all_models[lane]
        sar_res    = m['sarimax_result']
        xgb_model  = m.get('xgb_model')
        last_y     = m['last_y']          # Series of log1p(rate) training values
        last_exog  = m['last_exog']       # DataFrame of training exog (full history)
        xgb_feat_cols = m.get('xgb_feat_cols', [])

        # ── Determine SARIMAX exog columns ────────────────────────────────────
        try:
            sar_exog_cols = sar_res.model.exog_names
        except Exception:
            sar_exog_cols = list(last_exog.columns)

        # ── Load exog and panel data ───────────────────────────────────────────
        exog_df  = pd.read_parquet(exog_file)
        panel_df = pd.read_parquet(panel_file)

        last_date = exog_df.index[-1]
        fc_dates  = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')

        # ── Get lane-specific conflict impact scales ───────────────────────────
        scales = get_lane_impact_scales(lane)

        # Hormuz: if active, add 40% to Brent crude (oil price shock mechanism)
        effective_brent = float(ov['brent_crude'])
        if ov.get('dummy_hormuz', 0):
            effective_brent = effective_brent * 1.40

        # ── Build base exog values for forecast period ────────────────────────
        # Lane-scaled conflict dummies
        red_sea_val = float(ov.get('dummy_red_sea', 0)) * scales['red_sea']
        ukraine_val = float(ov.get('dummy_ukraine', 0)) * scales['ukraine']
        hormuz_val  = (float(ov.get('dummy_hormuz', 0)) * 0.5
                       if LANE_REGION.get(lane, '') == 'middle_east' else 0.0)
        panama_val  = float(ov.get('dummy_panama', 0)) * scales['panama']
        covid_val   = float(ov.get('dummy_covid', 0))
        supply_val  = float(ov.get('dummy_supply_crunch', 0))

        # Apply developer coefficient multipliers to continuous vars
        brent_eff  = effective_brent * float(ov.get('coef_brent', 1.0))
        usdcny_eff = float(ov['usdcny']) * float(ov.get('coef_usdcny', 1.0))
        bdry_eff   = float(ov['bdry_etf']) * float(ov.get('coef_bdry', 1.0))
        ukraine_eff = ukraine_val * float(ov.get('coef_ukraine', 1.0))
        red_sea_eff = red_sea_val * float(ov.get('coef_red_sea', 1.0))

        # Last known values for variables not overridden by user
        last_indpro  = float(exog_df['us_indpro'].iloc[-1])
        last_cfnai   = float(exog_df['us_cfnai'].iloc[-1])

        # ── Build SARIMAX exog matrix ─────────────────────────────────────────
        # Map from column name to override value
        # Lane-specific FX: use the correct currency pair for this lane
        lane_fx_col = {
            'CNSHA': 'usd_cny', 'JPTYO': 'usd_cny', 'KRPUS': 'usd_cny',
            'THBKK': 'usd_cny',
            'DEHAM': 'usd_eur', 'NLRTM': 'usd_eur', 'GBFXT': 'usd_eur',
            'INNSA': 'usd_inr',
            'AUSYD': 'usd_aud',
            'ARBUE': 'usd_ars',
            'NGLOS': 'usd_ngn',
            'AEJEA': None, 'QAHMD': None,  # AED/QAR pegged to USD
            'USNYC': None, 'USLAX': None,  # USD domestic
        }.get(lane, 'usd_cny')
        lane_fx_val = float(exog_df[lane_fx_col].iloc[-1]) if lane_fx_col and lane_fx_col in exog_df.columns else usdcny_eff
        last_maersk = float(exog_df['maersk_proxy'].iloc[-1]) if 'maersk_proxy' in exog_df.columns else 15000.0
        exog_value_map = {
            'brent_crude':         brent_eff,
            'usdcny':              usdcny_eff,
            'usd_cny':             usdcny_eff,
            'usd_eur':             lane_fx_val if lane_fx_col == 'usd_eur' else float(exog_df['usd_eur'].iloc[-1]) if 'usd_eur' in exog_df.columns else 1.08,
            'usd_inr':             lane_fx_val if lane_fx_col == 'usd_inr' else float(exog_df['usd_inr'].iloc[-1]) if 'usd_inr' in exog_df.columns else 83.0,
            'usd_aud':             lane_fx_val if lane_fx_col == 'usd_aud' else float(exog_df['usd_aud'].iloc[-1]) if 'usd_aud' in exog_df.columns else 1.52,
            'usd_ars':             lane_fx_val if lane_fx_col == 'usd_ars' else float(exog_df['usd_ars'].iloc[-1]) if 'usd_ars' in exog_df.columns else 900.0,
            'usd_ngn':             lane_fx_val if lane_fx_col == 'usd_ngn' else float(exog_df['usd_ngn'].iloc[-1]) if 'usd_ngn' in exog_df.columns else 1500.0,
            'maersk_proxy':        last_maersk,
            'us_indpro':           last_indpro,
            'us_cfnai':            last_cfnai,
            'bdry_etf':            bdry_eff,
            'dummy_covid':         covid_val,
            'dummy_supply_crunch': supply_val,
            'dummy_ukraine':       ukraine_eff,
            'dummy_red_sea':       red_sea_eff,
            'dummy_hormuz':        hormuz_val,
            # Path 1 extra shock dummies — always 0 in forecast (historical shocks)
            'shock_covid_spike':   0.0,
            'shock_post_crash':    0.0,
            # china_exports if present
            'china_exports':       float(exog_df['china_exports'].iloc[-1])
                                   if 'china_exports' in exog_df.columns else 0.0,
        }

        fc_sar_rows = []
        for _ in fc_dates:
            row = {col: exog_value_map.get(col, 0.0) for col in sar_exog_cols}
            fc_sar_rows.append(row)
        fc_sar_exog = pd.DataFrame(fc_sar_rows, index=fc_dates)[sar_exog_cols]

        # ── Run SARIMAX forecast ───────────────────────────────────────────────
        fc_res    = sar_res.forecast(steps=horizon, exog=fc_sar_exog.values)
        raw_vals  = np.array(fc_res)
        # Models trained on log1p(rate); if values look like log-space, exponentiate
        sar_preds = np.expm1(raw_vals) if raw_vals.max() < 15 else raw_vals
        sar_preds = np.clip(sar_preds, 0.01, None)

        # ── Build XGBoost feature matrix ──────────────────────────────────────
        xgb_corr = np.zeros(horizon)
        if xgb_model is not None and len(xgb_feat_cols) > 0:
            try:
                # Get the rate series for this lane
                if lane in panel_df.columns:
                    rate_series = panel_df[lane].dropna()
                else:
                    rate_series = pd.Series(dtype=float)

                # Build rolling history: last known rate values (in original space)
                rate_hist = list(rate_series.values[-12:]) if len(rate_series) >= 12 else list(rate_series.values)

                # Also get last known exog values for lagged features
                last_exog_vals = {col: float(last_exog[col].iloc[-1])
                                  for col in last_exog.columns if col in last_exog.columns}

                xgb_rows = []
                for h in range(horizon):
                    month_num = fc_dates[h].month
                    month_sin = np.sin(2 * np.pi * month_num / 12)
                    month_cos = np.cos(2 * np.pi * month_num / 12)

                    # Rate lags (use last known actuals, then rolling forward with SARIMAX preds)
                    hist_extended = rate_hist + list(sar_preds[:h])
                    def get_lag(n):
                        idx = len(hist_extended) - n
                        return float(hist_extended[idx]) if idx >= 0 else float(rate_hist[-1] if rate_hist else 0)

                    row = {}
                    for col in xgb_feat_cols:
                        if col == 'rate_lag1':  row[col] = get_lag(1)
                        elif col == 'rate_lag2':  row[col] = get_lag(2)
                        elif col == 'rate_lag3':  row[col] = get_lag(3)
                        elif col == 'rate_lag6':  row[col] = get_lag(6)
                        elif col == 'rate_lag12': row[col] = get_lag(12)
                        elif col == 'rate_roll3': row[col] = np.mean([get_lag(i) for i in range(1,4)])
                        elif col == 'rate_roll6': row[col] = np.mean([get_lag(i) for i in range(1,7)])
                        elif col == 'month_sin':  row[col] = month_sin
                        elif col == 'month_cos':  row[col] = month_cos
                        elif col == 'sar_pred':   row[col] = float(sar_preds[h])
                        elif col == 'brent_crude':         row[col] = brent_eff
                        elif col == 'brent_crude_l1':      row[col] = last_exog_vals.get('brent_crude', brent_eff)
                        elif col == 'usdcny':              row[col] = usdcny_eff
                        elif col == 'usd_cny':             row[col] = usdcny_eff
                        elif col == 'usdcny_l1':           row[col] = last_exog_vals.get('usdcny', usdcny_eff)
                        elif col == 'usd_cny_l1':          row[col] = last_exog_vals.get('usd_cny', usdcny_eff)
                        elif col == 'usd_eur':             row[col] = exog_value_map.get('usd_eur', 1.08)
                        elif col == 'usd_eur_l1':          row[col] = last_exog_vals.get('usd_eur', exog_value_map.get('usd_eur', 1.08))
                        elif col == 'usd_inr':             row[col] = exog_value_map.get('usd_inr', 83.0)
                        elif col == 'usd_inr_l1':          row[col] = last_exog_vals.get('usd_inr', exog_value_map.get('usd_inr', 83.0))
                        elif col == 'usd_aud':             row[col] = exog_value_map.get('usd_aud', 1.52)
                        elif col == 'usd_aud_l1':          row[col] = last_exog_vals.get('usd_aud', exog_value_map.get('usd_aud', 1.52))
                        elif col == 'usd_ars':             row[col] = exog_value_map.get('usd_ars', 900.0)
                        elif col == 'usd_ars_l1':          row[col] = last_exog_vals.get('usd_ars', exog_value_map.get('usd_ars', 900.0))
                        elif col == 'usd_ngn':             row[col] = exog_value_map.get('usd_ngn', 1500.0)
                        elif col == 'usd_ngn_l1':          row[col] = last_exog_vals.get('usd_ngn', exog_value_map.get('usd_ngn', 1500.0))
                        elif col == 'maersk_proxy':        row[col] = last_maersk
                        elif col == 'maersk_proxy_l1':     row[col] = last_exog_vals.get('maersk_proxy', last_maersk)
                        elif col == 'us_indpro':           row[col] = last_indpro
                        elif col == 'us_indpro_l1':        row[col] = last_exog_vals.get('us_indpro', last_indpro)
                        elif col == 'us_cfnai':            row[col] = last_cfnai
                        elif col == 'us_cfnai_l1':         row[col] = last_exog_vals.get('us_cfnai', last_cfnai)
                        elif col == 'bdry_etf':            row[col] = bdry_eff
                        elif col == 'bdry_etf_l1':         row[col] = last_exog_vals.get('bdry_etf', bdry_eff)
                        elif col == 'dummy_covid':         row[col] = covid_val
                        elif col == 'dummy_covid_l1':      row[col] = last_exog_vals.get('dummy_covid', 0.0)
                        elif col == 'dummy_supply_crunch': row[col] = supply_val
                        elif col == 'dummy_supply_crunch_l1': row[col] = last_exog_vals.get('dummy_supply_crunch', 0.0)
                        elif col == 'dummy_ukraine':       row[col] = ukraine_eff
                        elif col == 'dummy_ukraine_l1':    row[col] = last_exog_vals.get('dummy_ukraine', ukraine_eff)
                        elif col == 'dummy_red_sea':       row[col] = red_sea_eff
                        elif col == 'dummy_red_sea_l1':    row[col] = last_exog_vals.get('dummy_red_sea', red_sea_eff)
                        elif col == 'dummy_hormuz':        row[col] = hormuz_val
                        elif col == 'dummy_hormuz_l1':     row[col] = last_exog_vals.get('dummy_hormuz', hormuz_val)
                        elif col == 'shock_covid_spike':   row[col] = 0.0
                        elif col == 'shock_post_crash':    row[col] = 0.0
                        else:                              row[col] = 0.0
                    xgb_rows.append(row)

                xgb_feat_df = pd.DataFrame(xgb_rows, columns=xgb_feat_cols)
                raw_corr    = xgb_model.predict(xgb_feat_df.values)
                cap         = 0.30 * sar_preds
                xgb_corr    = np.clip(raw_corr, -cap, cap)
            except Exception:
                xgb_corr = np.zeros(horizon)

        stacked = np.clip(sar_preds + xgb_corr, 0.01, None)

        return {
            'dates':        [d.strftime('%Y-%m-%d') for d in fc_dates],
            'sarimax_pred': sar_preds.tolist(),
            'xgb_corr':     xgb_corr.tolist(),
            'stacked_pred': stacked.tolist(),
        }
    except Exception:
        return None

def make_chart_layout(title, xtitle='', ytitle=''):
    return dict(
        title=dict(text=title, font=dict(color='#0A1628', size=15, family='Arial Black')),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        font=dict(color='#0A1628', size=12),
        xaxis=dict(title=xtitle, color='#0A1628', gridcolor='#E8ECF2',
                   linecolor='#D0D7E3', tickfont=dict(color='#0A1628')),
        yaxis=dict(title=ytitle, color='#0A1628', gridcolor='#E8ECF2',
                   linecolor='#D0D7E3', tickfont=dict(color='#0A1628')),
        legend=dict(bgcolor='#F7F9FC', bordercolor='#D0D7E3', borderwidth=1,
                    font=dict(color='#0A1628')),
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode='x unified'
    )

# ── Load data ──────────────────────────────────────────────────────────────────
panel     = load_panel()
forecasts = load_forecasts()
comp_df   = load_comparison()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚢 Antarctica FCL")
    st.markdown("**Origin:** Antarctica (ANTXYZ)")
    st.markdown("---")

    lane_options = {f"{LANE_NAMES.get(l,l)} ({l})": l for l in DEMO_LANES}
    sel_label    = st.selectbox("Select Destination Lane", list(lane_options.keys()), key='lane_sel')
    sel_lane     = lane_options[sel_label]

    # Show lane impact scales for selected lane
    scales = get_lane_impact_scales(sel_lane)
    region = LANE_REGION.get(sel_lane, 'unknown').replace('_', ' ').title()
    st.caption(f"Region: **{region}**")

    st.markdown("---")

    # ── Load persisted overrides ──────────────────────────────────────────────
    if 'xfactor' not in st.session_state:
        st.session_state['xfactor'] = load_overrides()
    ov = st.session_state['xfactor']

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: CONFLICT ZONE TOGGLES
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-header">⚔️ Active Conflict Zones</p>', unsafe_allow_html=True)
    st.caption("Toggle to activate the corresponding shock in the forecast. Impact scales automatically by lane.")

    # Red Sea
    rs_scale = scales['red_sea']
    rs_label = f"🔴 Red Sea / Suez  ·  {impact_tag(rs_scale)}"
    ov['dummy_red_sea'] = int(st.toggle(
        "Red Sea / Suez Disruption",
        value=bool(ov['dummy_red_sea']), key='tog_red_sea',
        help=f"Impact on this lane: {rs_scale:.0%} of full effect (learned from 2023–2024 data). "
             f"Europe/Med lanes = 100%, Americas = 15%."))
    st.markdown(f"<small style='color:#5A6478'>Lane impact: {impact_tag(rs_scale)}</small>", unsafe_allow_html=True)

    # Hormuz
    st.markdown("")
    ov['dummy_hormuz'] = int(st.toggle(
        "🔴 Strait of Hormuz Risk",
        value=bool(ov['dummy_hormuz']), key='tog_hormuz',
        help="Hormuz controls 25% of global seaborne oil (not container trade). "
             "Activating this applies a +40% uplift to Brent crude, which cascades through "
             "the fuel surcharge mechanism. Middle East lanes also receive a direct 50% surcharge."))
    if ov['dummy_hormuz']:
        effective_brent = ov['brent_crude'] * 1.40
        st.markdown(
            f"<small style='color:#8B1A1A'>⚠️ Hormuz active: Brent → ATD {effective_brent:.1f}/bbl (+40% oil shock)</small>",
            unsafe_allow_html=True)

    # Ukraine
    uk_scale = scales['ukraine']
    st.markdown("")
    ov['dummy_ukraine'] = int(st.toggle(
        "🔴 Ukraine / Black Sea War",
        value=bool(ov['dummy_ukraine']), key='tog_ukraine',
        help=f"Impact on this lane: {uk_scale:.0%} of full effect (learned from 2022 data). "
             f"Europe lanes = 80%, Americas/Oceania = 10%."))
    st.markdown(f"<small style='color:#5A6478'>Lane impact: {impact_tag(uk_scale)}</small>", unsafe_allow_html=True)

    # Panama
    pa_scale = scales['panama']
    st.markdown("")
    ov['dummy_panama'] = int(st.toggle(
        "🟡 Panama Canal Restriction",
        value=bool(ov.get('dummy_panama', 0)), key='tog_panama',
        help=f"Impact on this lane: {pa_scale:.0%} of full effect. "
             f"Americas/trans-Pacific lanes = 80%, others minimal."))
    st.markdown(f"<small style='color:#5A6478'>Lane impact: {impact_tag(pa_scale)}</small>", unsafe_allow_html=True)

    # Historical dummies (collapsed by default)
    with st.expander("🕐 Historical Shock Dummies (for scenario testing)", expanded=False):
        ov['dummy_covid'] = int(st.toggle(
            "COVID Demand Shock (2020–2021)",
            value=bool(ov.get('dummy_covid', 0)), key='tog_covid',
            help="Activates the COVID demand surge dummy. For historical scenario testing only."))
        ov['dummy_supply_crunch'] = int(st.toggle(
            "Container Supply Crunch (2021–2022)",
            value=bool(ov.get('dummy_supply_crunch', 0)), key='tog_supply',
            help="Activates the 2021–2022 container shortage dummy."))

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: MACRO CONDITIONS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-header">📊 Macro Conditions</p>', unsafe_allow_html=True)
    st.caption("Set forecast-period values for key drivers. Defaults = live values (Apr 2026).")

    ov['brent_crude'] = st.slider(
        "Brent Crude Oil (USD/bbl)", 40.0, 160.0,
        float(ov['brent_crude']), 1.0, key='sl_brent',
        help="Forecast-period Brent crude price. Live: ~$74/bbl (Apr 2026). "
             "Note: Hormuz toggle adds +40% on top of this value.")

    ov['usdcny'] = st.slider(
        "USD / CNY Exchange Rate", 6.0, 9.0,
        float(ov['usdcny']), 0.05, key='sl_usdcny',
        help="Forecast-period USD/CNY rate. Live: ~7.28 (Apr 2026). "
             "Higher = weaker CNY = slightly lower USD-denominated rates from China.")

    ov['bdry_etf'] = st.slider(
        "BDRY ETF — Dry Bulk Proxy", 3.0, 50.0,
        float(ov['bdry_etf']), 0.5, key='sl_bdry',
        help="BDRY ETF price as proxy for Baltic Dry Index. Live: ~10.4 (Apr 2026). "
             "Rising BDRY = tightening shipping capacity = upward rate pressure.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: DEVELOPER CONTROLS (password-protected)
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🔧 Developer Controls", expanded=False):
        dev_pw = st.text_input("Developer password", type="password", key='dev_pw')
        if dev_pw == "antxyz2024":
            st.success("Developer mode unlocked")
            st.markdown("**Coefficient Multipliers**")
            st.caption("Scale each variable's learned Baseline Model effect. 1.0 = model default. "
                       ">1 amplifies, <1 dampens. Use when you believe the model underweights a factor.")
            ov['coef_brent']   = st.slider("Brent coefficient ×", 0.0, 3.0, float(ov.get('coef_brent',1.0)),   0.05, key='cm_brent',
                                           help="Multiplier on the Baseline Model Brent crude coefficient.")
            ov['coef_usdcny']  = st.slider("USD/CNY coefficient ×", 0.0, 3.0, float(ov.get('coef_usdcny',1.0)),  0.05, key='cm_usdcny')
            ov['coef_bdry']    = st.slider("BDRY coefficient ×", 0.0, 3.0, float(ov.get('coef_bdry',1.0)),    0.05, key='cm_bdry')
            ov['coef_ukraine'] = st.slider("Ukraine coefficient ×", 0.0, 3.0, float(ov.get('coef_ukraine',1.0)), 0.05, key='cm_ukraine')
            ov['coef_red_sea'] = st.slider("Red Sea coefficient ×", 0.0, 3.0, float(ov.get('coef_red_sea',1.0)), 0.05, key='cm_redsea')

            st.markdown("---")
            st.markdown("**Current Effective Exog Values (this lane)**")
            eff_brent = ov['brent_crude'] * (1.40 if ov.get('dummy_hormuz') else 1.0) * ov.get('coef_brent',1.0)
            st.markdown(f"""
            | Variable | Raw | Lane Scale | Effective |
            |----------|-----|-----------|-----------|
            | Brent Crude | {ov['brent_crude']:.1f} | {'×1.40 (Hormuz)' if ov.get('dummy_hormuz') else '×1.00'} | **{eff_brent:.1f}** |
            | USD/CNY | {ov['usdcny']:.2f} | ×{ov.get('coef_usdcny',1.0):.2f} | **{ov['usdcny']*ov.get('coef_usdcny',1.0):.2f}** |
            | BDRY ETF | {ov['bdry_etf']:.1f} | ×{ov.get('coef_bdry',1.0):.2f} | **{ov['bdry_etf']*ov.get('coef_bdry',1.0):.1f}** |
            | Red Sea dummy | {ov['dummy_red_sea']} | ×{scales['red_sea']:.2f} | **{ov['dummy_red_sea']*scales['red_sea']*ov.get('coef_red_sea',1.0):.2f}** |
            | Ukraine dummy | {ov['dummy_ukraine']} | ×{scales['ukraine']:.2f} | **{ov['dummy_ukraine']*scales['ukraine']*ov.get('coef_ukraine',1.0):.2f}** |
            """)
        elif dev_pw:
            st.error("Incorrect password")

    # ── Save / Reset buttons ──────────────────────────────────────────────────
    st.markdown("")
    col_s, col_n, col_r = st.columns(3)
    with col_s:
        if st.button("💾 Save", key='btn_save', help="Save current settings to disk"):
            save_overrides(ov)
            st.success("Saved!")
    with col_n:
        if st.button("⬜ Neutral", key='btn_neutral',
                     help="Set all conflict toggles OFF, macro to live values. Does not reset dev multipliers."):
            neutral = dict(ov)
            neutral.update({
                'dummy_ukraine': 0, 'dummy_red_sea': 0, 'dummy_hormuz': 0,
                'dummy_panama': 0, 'dummy_covid': 0, 'dummy_supply_crunch': 0,
                'brent_crude': 74.0, 'usdcny': 7.28, 'bdry_etf': 10.4,
            })
            st.session_state['xfactor'] = neutral
            save_overrides(neutral)
            st.rerun()
    with col_r:
        if st.button("↺ Reset All", key='btn_reset',
                     help="Reset everything including developer multipliers to factory defaults"):
            st.session_state['xfactor'] = dict(XFACTOR_DEFAULTS)
            save_overrides(XFACTOR_DEFAULTS)
            st.rerun()

    # Persist to session state
    st.session_state['xfactor'] = ov

    st.markdown("---")
    st.caption("Data: FRED · OECD · yfinance · Actual rate data")
    st.caption("Model: Baseline Model + Adjustment Engine Stacking")
    st.caption("Impact scales: ITF-OECD 2024 · EIA 2023 · UNCTAD 2024")

# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("# 🚢 Antarctica FCL Freight Rate Forecasting")
st.markdown(
    f"**Origin:** Antarctica (ANTXYZ) &nbsp;|&nbsp; "
    f"**Destination:** {LANE_NAMES.get(sel_lane, sel_lane)} ({sel_lane}) &nbsp;|&nbsp; "
    f"**Currency:** ATD (Antarctica Dollars) &nbsp;|&nbsp; **Unit:** WM/RT &nbsp;|&nbsp; "
    f"**Region:** {LANE_REGION.get(sel_lane,'').replace('_',' ').title()}"
)

# Active conditions banner
active_conflicts = []
if ov.get('dummy_red_sea'):  active_conflicts.append(f"Red Sea ({scales['red_sea']:.0%} impact)")
if ov.get('dummy_ukraine'):  active_conflicts.append(f"Ukraine ({scales['ukraine']:.0%} impact)")
if ov.get('dummy_hormuz'):   active_conflicts.append("Hormuz → Brent +40%")
if ov.get('dummy_panama'):   active_conflicts.append(f"Panama ({scales['panama']:.0%} impact)")
if ov.get('dummy_covid'):    active_conflicts.append("COVID dummy")
if ov.get('dummy_supply_crunch'): active_conflicts.append("Supply Crunch dummy")

if active_conflicts:
    st.markdown(
        f'<div class="warn-box"><p>⚠️ <strong>Active X-Factors:</strong> {" · ".join(active_conflicts)} '
        f'| Brent: ATD {ov["brent_crude"]*(1.40 if ov.get("dummy_hormuz") else 1.0):.1f}/bbl '
        f'| BDRY ETF: {ov["bdry_etf"]:.1f} | USD/CNY: {ov["usdcny"]:.2f}</p></div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="info-box"><p>ℹ️ <strong>Neutral conditions:</strong> No active conflict dummies. '
        f'Brent: ATD {ov["brent_crude"]:.1f}/bbl | BDRY ETF: {ov["bdry_etf"]:.1f} | USD/CNY: {ov["usdcny"]:.2f}</p></div>',
        unsafe_allow_html=True)

tabs = st.tabs(["📊 Overview", "📅 Seasonality", "🔵 Model with COVID",
                "🟢 Stable Model", "🔴 Test Mode", "🎯 Validation 2025–2026",
                "⚖️ Model Comparison", "🗄️ Data Sources"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## Overview — All Lanes")

    val1 = load_validation('path1')
    val2 = load_validation('path2')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Lanes Modelled", "15")
    with c2:
        med1 = val1['val_sar_mape'].median() if not val1.empty else 0
        st.metric("Model with COVID — Median MAPE", f"{med1:.1f}%", help="Jan 2025–Apr 2026 validation")
    with c3:
        med2 = val2['val_sar_mape'].median() if not val2.empty else 0
        st.metric("Stable Model — Median MAPE", f"{med2:.1f}%", help="Jan 2025–Apr 2026 validation")
    with c4:
        winner_count = (comp_df['winner'] == 'Path1').sum() if not comp_df.empty else 0
        st.metric("Model with COVID Wins", f"{winner_count}/15 lanes")

    st.markdown("---")
    st.markdown("### Historical Rate Chart — Selected Lane")

    y = panel[sel_lane].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y.values, mode='lines',
                             name=f'{sel_lane} Actual',
                             line=dict(color=C['navy'], width=2)))
    fig.update_layout(**make_chart_layout(
        f"Historical FCL Rate — {LANE_NAMES.get(sel_lane,sel_lane)} ({sel_lane})",
        ytitle="Rate (ATD / WM/RT)"))
    st.plotly_chart(fig, use_container_width=True, key='overview_hist')

    st.markdown("### All Lanes — Rate Overview")
    rows = []
    for l in DEMO_LANES:
        y_l = panel[l].dropna()
        if len(y_l) == 0:
            continue
        rel = 'Unknown'
        if not val1.empty and l in val1['lane'].values:
            rel = val1.loc[val1['lane']==l, 'reliability'].values[0]
        sc = get_lane_impact_scales(l)
        rows.append({
            'Lane': l,
            'Destination': LANE_NAMES.get(l, l),
            'Region': LANE_REGION.get(l,'').replace('_',' ').title(),
            'Data Points': len(y_l),
            'Mean Rate': f"ATD {y_l.mean():.1f}",
            'Min': f"ATD {y_l.min():.1f}",
            'Max': f"ATD {y_l.max():.1f}",
            'Reliability': rel,
            'Red Sea Impact': f"{sc['red_sea']:.0%}",
            'Ukraine Impact': f"{sc['ukraine']:.0%}",
        })
    ov_df = pd.DataFrame(rows)
    st.dataframe(ov_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-box">
    <p><strong>Reliability Score:</strong> Based on coefficient of variation (CV) and backtest MAPE.
    <strong>High</strong> = CV &le; 0.40 and MAPE &le; 25% &nbsp;|&nbsp;
    <strong>Medium</strong> = CV &le; 0.60 and MAPE &le; 50% &nbsp;|&nbsp;
    <strong>Low</strong> = high volatility lanes (typically long-haul routes affected by 2021–2022 supply shock).</p>
    <p><strong>Impact columns</strong> show the fraction of the full conflict dummy effect applied to each lane,
    based on route dependency on the affected chokepoint (ITF-OECD 2024, EIA 2023).</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SEASONALITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Seasonality Analysis")

    st.markdown("### Monthly Seasonality — Selected Lane")
    y = panel[sel_lane].dropna()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    if len(y) >= 12:
        seasonal = y.groupby(y.index.month).mean()
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=[month_names[m-1] for m in seasonal.index],
            y=seasonal.values,
            marker_color=C['blue'],
            text=[f"ATD {v:.1f}" for v in seasonal.values],
            textposition='outside',
            textfont=dict(color=C['navy'], size=11)
        ))
        overall_mean = y.mean()
        fig_s.add_hline(y=overall_mean, line_dash='dash',
                        line_color=C['red'], line_width=1.5,
                        annotation_text=f"Mean ATD {overall_mean:.1f}",
                        annotation_font_color=C['red'])
        fig_s.update_layout(**make_chart_layout(
            f"Average Monthly Rate — {LANE_NAMES.get(sel_lane,sel_lane)} ({sel_lane})",
            ytitle="Avg Rate (ATD / WM/RT)"))
        st.plotly_chart(fig_s, use_container_width=True, key='seasonal_bar')

        st.markdown("#### Seasonal Index (% deviation from annual mean)")
        idx_vals = (seasonal / overall_mean - 1) * 100
        fig_idx = go.Figure()
        colors = [C['green'] if v >= 0 else C['red'] for v in idx_vals.values]
        fig_idx.add_trace(go.Bar(
            x=[month_names[m-1] for m in idx_vals.index],
            y=idx_vals.values,
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in idx_vals.values],
            textposition='outside',
            textfont=dict(color=C['navy'], size=11)
        ))
        fig_idx.add_hline(y=0, line_color=C['navy'], line_width=1)
        fig_idx.update_layout(**make_chart_layout(
            f"Seasonal Index — {LANE_NAMES.get(sel_lane,sel_lane)}",
            ytitle="% deviation from annual mean"))
        st.plotly_chart(fig_idx, use_container_width=True, key='seasonal_idx')

        # Year-over-year overlay
        st.markdown("#### Year-over-Year Rate Overlay")
        st.caption("Each line = one calendar year. Shows whether the seasonal pattern repeats consistently.")
        fig_yoy = go.Figure()
        yoy_colors = [C['navy'], C['blue'], C['teal'], C['green'], C['amber'], C['purple'], C['grey']]
        years = sorted(y.index.year.unique())
        for i, yr in enumerate(years):
            yr_data = y[y.index.year == yr]
            fig_yoy.add_trace(go.Scatter(
                x=[month_names[m-1] for m in yr_data.index.month],
                y=yr_data.values,
                mode='lines+markers',
                name=str(yr),
                line=dict(color=yoy_colors[i % len(yoy_colors)], width=2),
                marker=dict(size=5)
            ))
        fig_yoy.update_layout(**make_chart_layout(
            f"Year-over-Year Overlay — {LANE_NAMES.get(sel_lane,sel_lane)}",
            ytitle="Rate (ATD / WM/RT)"))
        st.plotly_chart(fig_yoy, use_container_width=True, key='yoy_overlay')

    st.markdown("---")
    st.markdown("### Seasonality Heatmap — All Lanes")
    st.caption("Average rate by month across all 15 lanes. Darker = higher rate.")

    heat_data = []
    for l in DEMO_LANES:
        y_l = panel[l].dropna()
        if len(y_l) >= 12:
            s = y_l.groupby(y_l.index.month).mean()
            heat_data.append([s.get(m, np.nan) for m in range(1, 13)])
        else:
            heat_data.append([np.nan]*12)

    heat_arr = np.array(heat_data, dtype=float)
    fig_heat = go.Figure(go.Heatmap(
        z=heat_arr,
        x=month_names,
        y=[LANE_NAMES.get(l,l) for l in DEMO_LANES],
        colorscale='Blues',
        colorbar=dict(title=dict(text="Avg Rate (ATD)", side='right'),
                      tickfont=dict(color='#0A1628')),
        hovertemplate='Lane: %{y}<br>Month: %{x}<br>Avg Rate: ATD %{z:.1f}<extra></extra>'
    ))
    fig_heat.update_layout(
        title=dict(text="Average Monthly Rate Heatmap — All Lanes", font=dict(color='#0A1628', size=14)),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        font=dict(color='#0A1628'),
        xaxis=dict(tickfont=dict(color='#0A1628')),
        yaxis=dict(tickfont=dict(color='#0A1628')),
        margin=dict(l=200, r=60, t=50, b=50),
        height=500
    )
    st.plotly_chart(fig_heat, use_container_width=True, key='heat_all')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL WITH COVID
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## Model with COVID — Full Data + Shock Dummies")

    st.markdown("""
    <div class="formula-box">
    Final Forecast = Baseline Model Prediction + Adjustment Engine Correction (±30% cap)<br>
    Training window: Jul 2019 – Dec 2024 · Exogenous: Brent, USD/CNY, BDRY, IndPro, CFNAI, China Exports + Conflict Dummies
    </div>
    """, unsafe_allow_html=True)

    # Try live re-forecast with current X-factor settings
    live_fc1 = compute_live_forecast('path1', sel_lane, ov, horizon=6)
    fc1_static = forecasts.get('path1', {}).get(sel_lane, {})
    val1_df  = load_validation('path1')
    bt1_df   = load_backtest('path1', sel_lane)
    vl1_df   = load_val_csv('path1', sel_lane)

    # Use live forecast if available, else fall back to static
    fc1_data = live_fc1 if live_fc1 else fc1_static.get('raw', {})
    reliability = fc1_static.get('reliability', 'Unknown')

    st.markdown(f"**Lane Reliability:** {rel_badge(reliability)}", unsafe_allow_html=True)

    sar_m1 = val1_df.loc[val1_df['lane']==sel_lane, 'val_sar_mape'].values[0] \
             if not val1_df.empty and sel_lane in val1_df['lane'].values else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Validation MAPE (SAR)", f"{sar_m1:.1f}%" if sar_m1 else "N/A",
                  help="Jan 2025–Apr 2026 out-of-sample Baseline Model MAPE")
    with c2:
        last_val = panel[sel_lane].dropna().iloc[-1]
        st.metric("Last Known Rate", f"ATD {last_val:.2f}")
    with c3:
        if fc1_data.get('stacked_pred'):
            st.metric("Next Month Forecast", f"ATD {fc1_data['stacked_pred'][0]:.2f}",
                      help="Baseline Model + Adjustment Engine correction with current X-factor settings")
    with c4:
        if fc1_data.get('stacked_pred'):
            st.metric("6-Month Forecast", f"ATD {fc1_data['stacked_pred'][-1]:.2f}")

    if fc1_data.get('stacked_pred'):
        y_hist = panel[sel_lane].dropna()
        fig_fc = go.Figure()

        # Historical
        fig_fc.add_trace(go.Scatter(
            x=y_hist.index, y=y_hist.values, mode='lines',
            name='Historical Actual', line=dict(color=C['navy'], width=2)))

        # Backtest
        if not bt1_df.empty:
            fig_fc.add_trace(go.Scatter(
                x=bt1_df.index, y=bt1_df['stacked_pred'], mode='lines',
                name='Backtest (Stacked)', line=dict(color=C['teal'], width=1.5, dash='dot')))

        # Validation actuals
        if not vl1_df.empty:
            fig_fc.add_trace(go.Scatter(
                x=vl1_df.index, y=vl1_df['actual'], mode='lines+markers',
                name='Validation Actual', line=dict(color=C['navy'], width=2),
                marker=dict(symbol='circle', size=6, color=C['navy'])))
            fig_fc.add_trace(go.Scatter(
                x=vl1_df.index, y=vl1_df['sarimax_pred'], mode='lines',
                name='Validation Baseline', line=dict(color=C['amber'], width=1.5, dash='dash')))
            fig_fc.add_trace(go.Scatter(
                x=vl1_df.index, y=vl1_df['stacked_pred'], mode='lines',
                name='Validation Stacked', line=dict(color=C['teal'], width=1.5, dash='dash')))

        # Live X-factor forecast
        dates_fc = pd.to_datetime(fc1_data['dates'])
        fig_fc.add_trace(go.Scatter(
            x=dates_fc, y=fc1_data['sarimax_pred'], mode='lines+markers',
            name='Baseline Forecast', line=dict(color=C['amber'], width=2, dash='dash'),
            marker=dict(size=6)))
        fig_fc.add_trace(go.Scatter(
            x=dates_fc, y=fc1_data['stacked_pred'], mode='lines+markers',
            name='Stacked Forecast (X-factors applied)', line=dict(color=C['blue'], width=3),
            marker=dict(size=8, symbol='diamond')))

        # Confidence band (±15% of stacked)
        stacked_arr = np.array(fc1_data['stacked_pred'])
        fig_fc.add_trace(go.Scatter(
            x=list(dates_fc) + list(dates_fc[::-1]),
            y=list(stacked_arr * 1.15) + list((stacked_arr * 0.85)[::-1]),
            fill='toself', fillcolor='rgba(27,58,107,0.08)',
            line=dict(color='rgba(0,0,0,0)'), name='±15% Confidence Band',
            showlegend=True))

        fig_fc.add_vline(x=pd.Timestamp('2025-01-01').timestamp()*1000,
                         line_dash='dot', line_color=C['grey'], line_width=1,
                         annotation_text="Validation Start", annotation_font_color=C['grey'])

        fig_fc.update_layout(**make_chart_layout(
            f"Model with COVID — {LANE_NAMES.get(sel_lane,sel_lane)} ({sel_lane})",
            ytitle="Rate (ATD / WM/RT)"))
        st.plotly_chart(fig_fc, use_container_width=True, key='p1_forecast')

        # Forecast table
        st.markdown("#### 6-Month Forecast Table")
        rows = []
        for h, dt in enumerate(fc1_data['dates']):
            sar_v = fc1_data['sarimax_pred'][h]
            stk_v = fc1_data['stacked_pred'][h]
            xgb_v = fc1_data.get('xgb_corr', [0]*6)[h]
            rows.append({
                'Month': dt[:7],
                'Baseline Model': f"ATD {sar_v:.2f}",
                'Adjustment Engine': f"ATD {xgb_v:+.2f}",
                'Final Forecast': f"ATD {stk_v:.2f}",
                'Lower (−15%)': f"ATD {stk_v*0.85:.2f}",
                'Upper (+15%)': f"ATD {stk_v*1.15:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if live_fc1:
            st.markdown("""
            <div class="info-box">
            <p>✅ <strong>Live forecast</strong> — computed using your current X-factor settings.
            Change the sidebar controls and the forecast updates automatically.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No Model with COVID forecast available for this lane.")

    # Backtest chart
    if not bt1_df.empty:
        st.markdown("#### Walk-Forward Backtest — Actual vs Predicted")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt1_df.index, y=bt1_df['actual'],
                                    mode='lines', name='Actual',
                                    line=dict(color=C['navy'], width=2)))
        fig_bt.add_trace(go.Scatter(x=bt1_df.index, y=bt1_df['sarimax_pred'],
                                    mode='lines', name='Baseline Model',
                                    line=dict(color=C['amber'], width=1.5, dash='dash')))
        fig_bt.add_trace(go.Scatter(x=bt1_df.index, y=bt1_df['stacked_pred'],
                                    mode='lines', name='Final Forecast',
                                    line=dict(color=C['teal'], width=1.5, dash='dot')))
        fig_bt.update_layout(**make_chart_layout("Model with COVID — Walk-Forward Backtest",
                                                  ytitle="Rate (ATD / WM/RT)"))
        st.plotly_chart(fig_bt, use_container_width=True, key='p1_backtest')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: STABLE MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## Stable Model — Post-Jul 2022 Split")

    st.markdown("""
    <div class="formula-box">
    Final Forecast = Baseline Model Prediction + Adjustment Engine Correction (±30% cap)<br>
    Training window: Jul 2022 – Dec 2024 (normalised post-shock market only)
    </div>
    """, unsafe_allow_html=True)

    live_fc2    = compute_live_forecast('path2', sel_lane, ov, horizon=6)
    fc2_static  = forecasts.get('path2', {}).get(sel_lane, {})
    val2_df     = load_validation('path2')
    bt2_df      = load_backtest('path2', sel_lane)
    vl2_df      = load_val_csv('path2', sel_lane)

    fc2_data    = live_fc2 if live_fc2 else fc2_static.get('raw', {})
    reliability2 = fc2_static.get('reliability', 'Unknown')

    st.markdown(f"**Lane Reliability:** {rel_badge(reliability2)}", unsafe_allow_html=True)

    sar_m2 = val2_df.loc[val2_df['lane']==sel_lane, 'val_sar_mape'].values[0] \
             if not val2_df.empty and sel_lane in val2_df['lane'].values else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Validation MAPE (SAR)", f"{sar_m2:.1f}%" if sar_m2 else "N/A",
                  help="Jan 2025–Apr 2026 out-of-sample")
    with c2:
        last_val = panel[sel_lane].dropna().iloc[-1]
        st.metric("Last Known Rate", f"ATD {last_val:.2f}")
    with c3:
        if fc2_data.get('stacked_pred'):
            st.metric("Next Month Forecast", f"ATD {fc2_data['stacked_pred'][0]:.2f}")
    with c4:
        if fc2_data.get('stacked_pred'):
            st.metric("6-Month Forecast", f"ATD {fc2_data['stacked_pred'][-1]:.2f}")

    if fc2_data.get('stacked_pred'):
        y_hist = panel[sel_lane].dropna()
        fig_fc2 = go.Figure()
        fig_fc2.add_trace(go.Scatter(x=y_hist.index, y=y_hist.values, mode='lines',
                                     name='Historical Actual', line=dict(color=C['navy'], width=2)))
        if not bt2_df.empty:
            fig_fc2.add_trace(go.Scatter(x=bt2_df.index, y=bt2_df['stacked_pred'], mode='lines',
                                          name='Backtest (Stacked)',
                                          line=dict(color=C['teal'], width=1.5, dash='dot')))
        if not vl2_df.empty:
            fig_fc2.add_trace(go.Scatter(x=vl2_df.index, y=vl2_df['actual'],
                                          mode='lines+markers', name='Validation Actual',
                                          line=dict(color=C['navy'], width=2),
                                          marker=dict(size=6, color=C['navy'])))
            fig_fc2.add_trace(go.Scatter(x=vl2_df.index, y=vl2_df['sarimax_pred'],
                                          mode='lines', name='Validation Baseline',
                                          line=dict(color=C['amber'], width=1.5, dash='dash')))
            fig_fc2.add_trace(go.Scatter(x=vl2_df.index, y=vl2_df['stacked_pred'],
                                          mode='lines', name='Validation Stacked',
                                          line=dict(color=C['teal'], width=1.5, dash='dash')))

        dates_fc2 = pd.to_datetime(fc2_data['dates'])
        fig_fc2.add_trace(go.Scatter(
            x=dates_fc2, y=fc2_data['sarimax_pred'], mode='lines+markers',
            name='Baseline Forecast', line=dict(color=C['amber'], width=2, dash='dash'),
            marker=dict(size=6)))
        fig_fc2.add_trace(go.Scatter(
            x=dates_fc2, y=fc2_data['stacked_pred'], mode='lines+markers',
            name='Stacked Forecast (X-factors applied)', line=dict(color=C['green'], width=3),
            marker=dict(size=8, symbol='diamond')))

        stacked_arr2 = np.array(fc2_data['stacked_pred'])
        fig_fc2.add_trace(go.Scatter(
            x=list(dates_fc2) + list(dates_fc2[::-1]),
            y=list(stacked_arr2 * 1.15) + list((stacked_arr2 * 0.85)[::-1]),
            fill='toself', fillcolor='rgba(26,92,42,0.08)',
            line=dict(color='rgba(0,0,0,0)'), name='±15% Confidence Band'))

        fig_fc2.add_vline(x=pd.Timestamp('2022-07-01').timestamp()*1000,
                          line_dash='dot', line_color=C['grey'], line_width=1,
                          annotation_text="Training Start", annotation_font_color=C['grey'])
        fig_fc2.add_vline(x=pd.Timestamp('2025-01-01').timestamp()*1000,
                          line_dash='dot', line_color=C['amber'], line_width=1,
                          annotation_text="Validation Start", annotation_font_color=C['amber'])

        fig_fc2.update_layout(**make_chart_layout(
            f"Stable Model — {LANE_NAMES.get(sel_lane,sel_lane)} ({sel_lane})",
            ytitle="Rate (ATD / WM/RT)"))
        st.plotly_chart(fig_fc2, use_container_width=True, key='p2_forecast')

        st.markdown("#### 6-Month Forecast Table")
        rows = []
        for h, dt in enumerate(fc2_data['dates']):
            sar_v = fc2_data['sarimax_pred'][h]
            stk_v = fc2_data['stacked_pred'][h]
            xgb_v = fc2_data.get('xgb_corr', [0]*6)[h]
            rows.append({
                'Month': dt[:7],
                'Baseline Model': f"ATD {sar_v:.2f}",
                'Adjustment Engine': f"ATD {xgb_v:+.2f}",
                'Final Forecast': f"ATD {stk_v:.2f}",
                'Lower (−15%)': f"ATD {stk_v*0.85:.2f}",
                'Upper (+15%)': f"ATD {stk_v*1.15:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No Stable Model forecast available for this lane.")

    if not bt2_df.empty:
        st.markdown("#### Walk-Forward Backtest — Actual vs Predicted")
        fig_bt2 = go.Figure()
        fig_bt2.add_trace(go.Scatter(x=bt2_df.index, y=bt2_df['actual'],
                                     mode='lines', name='Actual',
                                     line=dict(color=C['navy'], width=2)))
        fig_bt2.add_trace(go.Scatter(x=bt2_df.index, y=bt2_df['sarimax_pred'],
                                     mode='lines', name='Baseline Model',
                                     line=dict(color=C['amber'], width=1.5, dash='dash')))
        fig_bt2.add_trace(go.Scatter(x=bt2_df.index, y=bt2_df['stacked_pred'],
                                     mode='lines', name='Final Forecast',
                                     line=dict(color=C['teal'], width=1.5, dash='dot')))
        fig_bt2.update_layout(**make_chart_layout("Stable Model — Walk-Forward Backtest",
                                                   ytitle="Rate (ATD / WM/RT)"))
        st.plotly_chart(fig_bt2, use_container_width=True, key='p2_backtest')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: TEST MODE — SARIMAX → LSTM → XGBoost Triple-Stack
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🔴 Test Mode — Triple-Stack Architecture")
    st.markdown("""
    <div class="warn-box">
    <p>⚠️ <strong>Experimental.</strong> This tab tests a three-layer architecture:
    <strong>Baseline Model → LSTM → Adjustment Engine</strong>.
    The LSTM learns temporal patterns in the Baseline Model residuals before passing
    them to the Adjustment Engine. Trained on two lanes only:
    <strong>THBKK</strong> (lowest MAPE) and <strong>USLAX</strong> (highest MAPE).
    Results are compared directly against the current Baseline Model + Adjustment Engine stack.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load test mode summary
    tm_summary_fp = os.path.join(OUT, 'testmode_summary.json')
    if not os.path.exists(tm_summary_fp):
        st.error("Test mode outputs not found. Run train_lstm_stack.py first.")
    else:
        with open(tm_summary_fp) as f:
            tm_summary = json.load(f)

        # ── Architecture diagram ──────────────────────────────────────────────
        st.markdown("### Architecture")
        st.markdown("""
        <div class="formula-box">
        Layer 1 — Baseline Model: captures linear trend + seasonality + macro exog<br>
        Layer 2 — LSTM (NEW): learns temporal patterns in Baseline Model residuals (6-month lookback)<br>
        Layer 3 — Adjustment Engine: final non-linear correction using all features + LSTM output<br>
        <br>
        Final = Baseline Model Prediction + Adjustment Engine Correction (±30% cap)
        </div>
        """, unsafe_allow_html=True)

        # ── MAPE Comparison Summary ───────────────────────────────────────────
        st.markdown("### Accuracy Comparison — Validation Jan 2025 – Mar 2026")
        st.caption("Lower MAPE = better. Positive improvement = triple-stack is worse (LSTM adds noise at this data scale).")

        comp_rows = []
        for lane, res in tm_summary.items():
            lane_name = {'THBKK': 'Thailand – Bangkok', 'USLAX': 'USA – Los Angeles'}.get(lane, lane)
            imp = res['improvement_pp']
            comp_rows.append({
                'Lane': lane,
                'Destination': lane_name,
                'Baseline Model MAPE': f"{res['baseline_sar_mape']:.1f}%",
                'Baseline Final MAPE': f"{res['baseline_stk_mape']:.1f}%",
                'Triple-Stack MAPE': f"{res['triple_final_mape']:.1f}%",
                'Change vs Baseline': f"{imp:+.1f}pp",
                'Verdict': '✅ Better' if imp > 0 else '❌ Worse',
                'Train Obs': res['n_train'],
                'LSTM Epochs': res['lstm_epochs'],
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # ── Per-lane detailed charts ──────────────────────────────────────────
        for lane in ['THBKK', 'USLAX']:
            lane_name = {'THBKK': 'Thailand – Bangkok', 'USLAX': 'USA – Los Angeles'}.get(lane, lane)
            res = tm_summary.get(lane, {})

            st.markdown(f"---")
            st.markdown(f"### {lane} — {lane_name}")

            # Metrics row
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Baseline Model MAPE", f"{res.get('baseline_sar_mape', 0):.1f}%",
                          help="SARIMAX-only validation MAPE (Jan 2025–Mar 2026)")
            with mc2:
                st.metric("Baseline Final MAPE", f"{res.get('baseline_stk_mape', 0):.1f}%",
                          help="Baseline Model + Adjustment Engine validation MAPE")
            with mc3:
                st.metric("Triple-Stack MAPE", f"{res.get('triple_final_mape', 0):.1f}%",
                          help="Baseline Model + LSTM + Adjustment Engine validation MAPE")
            with mc4:
                imp = res.get('improvement_pp', 0)
                st.metric("MAPE Change", f"{imp:+.1f}pp",
                          delta=f"{imp:+.1f}pp",
                          delta_color="normal" if imp > 0 else "inverse",
                          help="Positive = triple-stack improved; Negative = triple-stack is worse")

            # Load validation data
            val_fp  = os.path.join(OUT, f'testmode_{lane}_validation.csv')
            bt_fp   = os.path.join(OUT, f'testmode_{lane}_backtest.csv')
            bl_val_fp = os.path.join(OUT, f'path1_validation_{lane}.csv')

            if os.path.exists(val_fp):
                val_tm  = pd.read_csv(val_fp, index_col=0, parse_dates=True)
                bl_val  = pd.read_csv(bl_val_fp, index_col=0, parse_dates=True) if os.path.exists(bl_val_fp) else pd.DataFrame()

                # Validation chart
                fig_tm = go.Figure()

                # Actual
                fig_tm.add_trace(go.Scatter(
                    x=val_tm.index, y=val_tm['actual'],
                    mode='lines+markers', name='Actual Rate',
                    line=dict(color=C['navy'], width=3),
                    marker=dict(size=8)))

                # Baseline Model prediction (from path1 validation)
                if not bl_val.empty:
                    fig_tm.add_trace(go.Scatter(
                        x=bl_val.index, y=bl_val['sarimax_pred'],
                        mode='lines+markers', name='Baseline Model',
                        line=dict(color=C['amber'], width=2, dash='dash'),
                        marker=dict(size=6)))
                    fig_tm.add_trace(go.Scatter(
                        x=bl_val.index, y=bl_val['stacked_pred'],
                        mode='lines+markers', name='Baseline Final (2-layer)',
                        line=dict(color=C['teal'], width=2, dash='dot'),
                        marker=dict(size=6)))

                # Triple-stack prediction
                fig_tm.add_trace(go.Scatter(
                    x=val_tm.index, y=val_tm['triple_stack'],
                    mode='lines+markers', name='Triple-Stack (3-layer)',
                    line=dict(color=C['red'], width=2.5),
                    marker=dict(size=7, symbol='diamond')))

                fig_tm.update_layout(**make_chart_layout(
                    f"Validation Comparison — {lane_name} ({lane})",
                    ytitle="Rate (ATD / WM/RT)"))
                st.plotly_chart(fig_tm, use_container_width=True, key=f'tm_val_{lane}')

                # Validation table
                st.markdown("#### Validation Table")
                vt_rows = []
                for i, row in val_tm.iterrows():
                    bl_pred = bl_val.loc[i, 'stacked_pred'] if (not bl_val.empty and i in bl_val.index) else None
                    bl_err  = abs(row['actual'] - bl_pred) / row['actual'] * 100 if bl_pred else None
                    ts_err  = abs(row['actual'] - row['triple_stack']) / row['actual'] * 100
                    vt_rows.append({
                        'Month': str(i)[:7],
                        'Actual': f"ATD {row['actual']:.2f}",
                        'Baseline Final': f"ATD {bl_pred:.2f}" if bl_pred else 'N/A',
                        'Baseline Error': f"{bl_err:.1f}%" if bl_err else 'N/A',
                        'Triple-Stack': f"ATD {row['triple_stack']:.2f}",
                        'Triple Error': f"{ts_err:.1f}%",
                        'Better': '✅ Triple' if (bl_err and ts_err < bl_err) else ('✅ Baseline' if bl_err else '—'),
                    })
                st.dataframe(pd.DataFrame(vt_rows), use_container_width=True, hide_index=True)

            # Backtest chart
            if os.path.exists(bt_fp):
                bt_tm = pd.read_csv(bt_fp, index_col=0, parse_dates=True)
                bl_bt_fp = os.path.join(OUT, f'path1_backtest_{lane}.csv')
                bl_bt = pd.read_csv(bl_bt_fp, index_col=0, parse_dates=True) if os.path.exists(bl_bt_fp) else pd.DataFrame()

                st.markdown("#### Walk-Forward Backtest Comparison")
                fig_bt_tm = go.Figure()

                if not bl_bt.empty:
                    fig_bt_tm.add_trace(go.Scatter(
                        x=bl_bt.index, y=bl_bt['actual'],
                        mode='lines', name='Actual',
                        line=dict(color=C['navy'], width=2)))
                    fig_bt_tm.add_trace(go.Scatter(
                        x=bl_bt.index, y=bl_bt['stacked_pred'],
                        mode='lines', name='Baseline Final (2-layer)',
                        line=dict(color=C['teal'], width=1.5, dash='dot')))

                fig_bt_tm.add_trace(go.Scatter(
                    x=bt_tm.index, y=bt_tm['triple_stack'],
                    mode='lines', name='Triple-Stack (3-layer)',
                    line=dict(color=C['red'], width=2)))

                fig_bt_tm.update_layout(**make_chart_layout(
                    f"Backtest Comparison — {lane_name} ({lane})",
                    ytitle="Rate (ATD / WM/RT)"))
                st.plotly_chart(fig_bt_tm, use_container_width=True, key=f'tm_bt_{lane}')

        # ── Interpretation ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Interpretation")
        st.markdown("""
        <div class="info-box">
        <p><strong>Why the triple-stack may underperform:</strong><br>
        With only ~54 monthly observations per lane, the LSTM has insufficient data to learn
        generalised temporal patterns. It tends to overfit the training residuals, producing
        larger errors on the unseen validation period. This confirms the theoretical expectation:
        LSTM requires substantially more data (typically 500+ sequences) to outperform simpler models.</p>
        <p><strong>When to revisit this architecture:</strong><br>
        If weekly or daily freight rate data becomes available (expanding observations by 4–30×),
        the LSTM layer would have enough signal to learn meaningful residual dynamics and is
        likely to improve accuracy — particularly for high-volatility lanes like USLAX.</p>
        <p><strong>Current recommendation:</strong> The two-layer Baseline Model + Adjustment Engine
        stack remains the production model. This tab is retained for ongoing research and
        re-evaluation as data volume grows.</p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: VALIDATION 2025–2026
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## True Out-of-Sample Validation — Jan 2025 to Apr 2026")
    st.markdown("""
    <div class="info-box">
    <p>This tab shows how well each model predicted rates for Jan 2025–Apr 2026 using only data
    available up to Dec 2024. This is the most honest test of model accuracy — no look-ahead bias.</p>
    </div>
    """, unsafe_allow_html=True)

    vl1 = load_val_csv('path1', sel_lane)
    vl2 = load_val_csv('path2', sel_lane)

    if not vl1.empty or not vl2.empty:
        fig_val = go.Figure()
        if not vl1.empty:
            fig_val.add_trace(go.Scatter(x=vl1.index, y=vl1['actual'],
                                         mode='lines+markers', name='Actual Rate',
                                         line=dict(color=C['navy'], width=3),
                                         marker=dict(size=8)))
            fig_val.add_trace(go.Scatter(x=vl1.index, y=vl1['sarimax_pred'],
                                         mode='lines+markers', name='Model with COVID Baseline',
                                         line=dict(color=C['blue'], width=2, dash='dash'),
                                         marker=dict(size=6)))
            fig_val.add_trace(go.Scatter(x=vl1.index, y=vl1['stacked_pred'],
                                         mode='lines+markers', name='Model with COVID Final',
                                         line=dict(color=C['teal'], width=2, dash='dot'),
                                         marker=dict(size=6)))
        if not vl2.empty:
            fig_val.add_trace(go.Scatter(x=vl2.index, y=vl2['sarimax_pred'],
                                         mode='lines+markers', name='Stable Model Baseline',
                                         line=dict(color=C['green'], width=2, dash='dash'),
                                         marker=dict(size=6)))
            fig_val.add_trace(go.Scatter(x=vl2.index, y=vl2['stacked_pred'],
                                         mode='lines+markers', name='Stable Model Final',
                                         line=dict(color=C['amber'], width=2, dash='dot'),
                                         marker=dict(size=6)))

        fig_val.update_layout(**make_chart_layout(
            f"Validation — {LANE_NAMES.get(sel_lane,sel_lane)} ({sel_lane})",
            ytitle="Rate (ATD / WM/RT)"))
        st.plotly_chart(fig_val, use_container_width=True, key='val_chart')

        # Validation metrics table
        val1_row = val1_df.loc[val1_df['lane']==sel_lane] if not val1_df.empty else pd.DataFrame()
        val2_row = val2_df.loc[val2_df['lane']==sel_lane] if not val2_df.empty else pd.DataFrame()

        if not val1_row.empty or not val2_row.empty:
            st.markdown("#### Validation Metrics")
            metric_rows = []
            for tag, row_df in [('Model with COVID', val1_row), ('Stable Model', val2_row)]:
                if not row_df.empty:
                    r = row_df.iloc[0]
                    metric_rows.append({
                        'Model': tag,
                        'SAR MAPE': f"{r.get('val_sar_mape', 'N/A'):.1f}%" if pd.notna(r.get('val_sar_mape')) else 'N/A',
                        'STK MAPE': f"{r.get('val_stk_mape', 'N/A'):.1f}%" if pd.notna(r.get('val_stk_mape')) else 'N/A',
                        'Reliability': r.get('reliability', 'Unknown'),
                        'Winner': '✅ Winner' if r.get('winner') else '',
                    })
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Validation data not yet available. Share Jan 2025–Apr 2026 actual rates to populate this tab.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("## Model Comparison — Model with COVID vs Stable Model")

    if not comp_df.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            p1_wins = (comp_df['winner'] == 'Path1').sum()
            st.metric("Model with COVID Wins", f"{p1_wins}/15 lanes")
        with c2:
            p2_wins = (comp_df['winner'] == 'Path2').sum()
            st.metric("Stable Model Wins", f"{p2_wins}/15 lanes")
        with c3:
            med_diff = (comp_df['P1_SAR_MAPE'] - comp_df['P2_SAR_MAPE']).median()
            st.metric("Median MAPE Difference", f"{med_diff:+.1f}%",
                      help="Positive = Stable Model is better on average")

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=comp_df['lane'], y=comp_df['P1_SAR_MAPE'],
            name='Model with COVID Baseline MAPE', marker_color=C['blue']))
        fig_comp.add_trace(go.Bar(
            x=comp_df['lane'], y=comp_df['P2_SAR_MAPE'],
            name='Stable Model Baseline MAPE', marker_color=C['green']))
        fig_comp.update_layout(**make_chart_layout(
            "Baseline Model MAPE by Lane — Model with COVID vs Stable Model",
            ytitle="MAPE (%)"), barmode='group')
        st.plotly_chart(fig_comp, use_container_width=True, key='comp_bar')

        # Winner table
        disp = comp_df.copy()
        disp['Winner'] = disp['winner'].map({'Path1': '🔵 Model with COVID', 'Path2': '🟢 Stable Model'})
        disp['P1 MAPE'] = disp['P1_SAR_MAPE'].apply(lambda x: f"{x:.1f}%")
        disp['P2 MAPE'] = disp['P2_SAR_MAPE'].apply(lambda x: f"{x:.1f}%")
        disp['Destination'] = disp['lane'].map(LANE_NAMES)
        st.dataframe(disp[['lane','Destination','P1 MAPE','P2 MAPE','Winner']].rename(columns={'lane':'Lane'}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Comparison data not yet available.")

    st.markdown("""
    <div class="info-box">
    <p><strong>Model with COVID:</strong> Full history Jul 2019–Dec 2024 with explicit shock dummies for COVID and supply crunch.
    Better for lanes with stable seasonality and sufficient pre-shock data.</p>
    <p><strong>Stable Model:</strong> Post-Jul 2022 only — excludes the extreme 2021–2022 spike period.
    Better for volatile lanes where the spike period distorts the model's baseline.</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8: DATA SOURCES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("## Data Sources")

    st.markdown("""
    | Source | Series / Data | Access | Used For |
    |--------|--------------|--------|---------|
    | **FRED API** (St. Louis Fed) | Brent Crude (`DCOILBRENTEU`), USD/CNY (`DEXCHUS`), US IndPro (`INDPRO`), CFNAI (`CFNAI`), China Exports (`XTEXVA01CNM667S`) | Free (API key required) | Macro exogenous variables |
    | **OECD SDMX REST** | China CLI, US CLI | Free (no key) | Leading indicators |
    | **Yahoo Finance / yfinance** | BDRY ETF (`BDRY`) | Free (no key) | Dry bulk shipping proxy (BDI) |
    | **Actual Rate Data** | FCL rates Antarctica → World | Proprietary | Target variable (WM/RT in ATD) |
    | [**ITF-OECD 2024**](https://www.itf-oecd.org/sites/default/files/docs/maritime-freight-costs-red-sea-crisis.pdf) | Red Sea crisis lane impact data | Free (public report) | Conflict impact scaling matrix |
    | [**EIA 2023**](https://www.eia.gov/international/analysis/regions-of-interest/Strait_of_Hormuz) | Strait of Hormuz oil flow data | Free (public) | Hormuz oil-shock mechanism |
    | [**UNCTAD 2024**](https://unctad.org/publication/review-maritime-transport-2024) | Global trade chokepoint data | Free (public) | Impact matrix calibration |
    """)

    st.markdown("---")
    st.markdown("### Conflict Impact Matrix — Linear Scaling")
    st.markdown("Impact scales applied to conflict dummies per lane, based on route dependency on each chokepoint.")

    impact_rows = []
    for l in DEMO_LANES:
        sc = get_lane_impact_scales(l)
        impact_rows.append({
            'Lane': l,
            'Destination': LANE_NAMES.get(l, l),
            'Region': LANE_REGION.get(l,'').replace('_',' ').title(),
            'Red Sea Scale': f"{sc['red_sea']:.0%}",
            'Ukraine Scale': f"{sc['ukraine']:.0%}",
            'Panama Scale':  f"{sc['panama']:.0%}",
            'Hormuz': 'Oil price shock (+40% Brent) for all lanes; +50% direct surcharge for Middle East',
        })
    st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-box">
    <p><strong>Hormuz mechanism:</strong> The Strait of Hormuz controls ~25% of global seaborne oil but
    carries negligible container trade. Its impact on FCL rates is therefore modelled as an
    <em>indirect oil price shock</em> (+40% uplift to Brent crude when toggle is active), which then
    propagates through the Baseline Model Brent coefficient. Middle East lanes (QAHMD, AEJEA) additionally
    receive a direct 50% surcharge dummy.</p>
    <p><strong>Linear scaling:</strong> Impact = toggle_value × lane_scale × coefficient_multiplier.
    A scale of 100% means the full effect learned from training data is applied.
    A scale of 15% means only 15% of the full effect is applied to that lane.</p>
    <p><strong>Sources:</strong>
    <a href="https://www.itf-oecd.org/sites/default/files/docs/maritime-freight-costs-red-sea-crisis.pdf" target="_blank">ITF-OECD 2024 — Maritime Freight Costs and the Red Sea Crisis</a> &nbsp;|
    <a href="https://www.eia.gov/international/analysis/regions-of-interest/Strait_of_Hormuz" target="_blank">EIA 2023 — Strait of Hormuz</a> &nbsp;|
    <a href="https://unctad.org/publication/review-maritime-transport-2024" target="_blank">UNCTAD 2024 — Review of Maritime Transport</a></p>
    </div>
    """, unsafe_allow_html=True)
