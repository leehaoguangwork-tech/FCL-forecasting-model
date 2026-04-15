"""
Pull all new exogenous data series for v2.0 model:
  1. USD/EUR, USD/INR, USD/AUD  — FRED API (free, no key)
  2. USD/ARS  — World Bank API (free, no key)
  3. USD/NGN  — World Bank API (free, no key)
  4. FBX Composite  — Freightos public data via web
  5. SCFI Composite — Shanghai Shipping Exchange via web scraping

Output: /home/ubuntu/fcl_forecast/data/new_exog_v2.parquet
        /home/ubuntu/fcl_forecast/data/new_exog_v2.csv
"""
import os, requests, warnings, time
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime
warnings.filterwarnings('ignore')

OUT_DIR = '/home/ubuntu/fcl_forecast/data'
os.makedirs(OUT_DIR, exist_ok=True)

START = '2019-01-01'
END   = datetime.today().strftime('%Y-%m-%d')

series_data = {}

# ── 1. FRED Currency Pairs ─────────────────────────────────────────────────
FRED_SERIES = {
    'usd_eur': 'DEXUSEU',
    'usd_inr': 'DEXINUS',
    'usd_aud': 'DEXUSAL',
}

print("Pulling FRED currency pairs...")
for col, sid in FRED_SERIES.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    df.columns = [col]
    df = df[df[col] != '.']
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    # Resample to monthly (last business day of month)
    monthly = df.resample('MS').last()
    monthly = monthly[monthly.index >= START]
    series_data[col] = monthly[col]
    print(f"  ✓ {col}: {len(monthly)} monthly obs, latest={monthly[col].iloc[-1]:.4f}")

# ── 2. World Bank API — USD/ARS and USD/NGN ────────────────────────────────
# World Bank indicator PA.NUS.FCRF = Official exchange rate (LCU per USD, period average)
WB_COUNTRIES = {
    'usd_ars': 'AR',  # Argentina
    'usd_ngn': 'NG',  # Nigeria
}

print("\nPulling World Bank exchange rates...")
for col, iso2 in WB_COUNTRIES.items():
    url = (f"https://api.worldbank.org/v2/country/{iso2}/indicator/PA.NUS.FCRF"
           f"?format=json&per_page=100&mrv=20&date=2019:2026")
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        if len(data) < 2 or not data[1]:
            raise ValueError("No data returned")
        rows = [(item['date'], item['value']) for item in data[1] if item['value'] is not None]
        df_wb = pd.DataFrame(rows, columns=['year', col])
        df_wb['date'] = pd.to_datetime(df_wb['year'].astype(str) + '-01-01')
        df_wb = df_wb.set_index('date').sort_index()
        df_wb[col] = pd.to_numeric(df_wb[col], errors='coerce')
        # World Bank is annual — forward-fill to monthly
        monthly_idx = pd.date_range(start=START, end=END, freq='MS')
        monthly = df_wb[col].reindex(monthly_idx, method='ffill')
        series_data[col] = monthly
        print(f"  ✓ {col}: {len(monthly)} monthly obs (annual WB data, forward-filled), latest={monthly.iloc[-1]:.2f}")
    except Exception as e:
        print(f"  ✗ {col}: {e} — will try alternative")
        # Fallback: try FRED alternative tickers
        alt_map = {'usd_ars': 'CCUSMA02ARA618N', 'usd_ngn': 'CCUSMA02NGA618N'}
        try:
            url2 = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={alt_map[col]}"
            df2 = pd.read_csv(url2, index_col=0, parse_dates=True)
            df2.columns = [col]
            df2 = df2[df2[col] != '.']
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
            df2 = df2.dropna()
            monthly2 = df2.resample('MS').last()
            monthly2 = monthly2[monthly2.index >= START]
            series_data[col] = monthly2[col]
            print(f"  ✓ {col} (FRED alt): {len(monthly2)} monthly obs, latest={monthly2[col].iloc[-1]:.2f}")
        except Exception as e2:
            print(f"  ✗ {col} fallback also failed: {e2}")

# ── 3. FBX Composite — Freightos Baltic Index ──────────────────────────────
print("\nPulling FBX Composite freight index...")
try:
    # Freightos publishes FBX data on their public portal
    # Try the public data endpoint used by their chart widgets
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://fbx.freightos.com/',
    }
    # Try the public API endpoint for FBX-11 (Global composite)
    url = "https://fbx.freightos.com/api/rates?index=FBX11&currency=USD&period=weekly"
    r = requests.get(url, headers=headers, timeout=15)
    print(f"  FBX API status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            rows = [(item.get('date') or item.get('week'), item.get('rate') or item.get('value'))
                    for item in data if item]
            df_fbx = pd.DataFrame(rows, columns=['date', 'fbx_composite'])
            df_fbx['date'] = pd.to_datetime(df_fbx['date'])
            df_fbx = df_fbx.set_index('date').sort_index()
            df_fbx['fbx_composite'] = pd.to_numeric(df_fbx['fbx_composite'], errors='coerce')
            df_fbx = df_fbx.dropna()
            monthly_fbx = df_fbx.resample('MS').mean()
            monthly_fbx = monthly_fbx[monthly_fbx.index >= START]
            series_data['fbx_composite'] = monthly_fbx['fbx_composite']
            print(f"  ✓ FBX Composite: {len(monthly_fbx)} monthly obs, latest={monthly_fbx['fbx_composite'].iloc[-1]:.0f}")
        else:
            print(f"  ✗ FBX: unexpected response format: {str(data)[:200]}")
            raise ValueError("Unexpected format")
    else:
        raise ValueError(f"HTTP {r.status_code}")
except Exception as e:
    print(f"  ✗ FBX direct API failed: {e}")
    # Fallback: try Drewry WCI via their public data
    print("  Trying Drewry WCI as fallback...")
    try:
        # Drewry WCI is published weekly; try their public JSON feed
        url2 = "https://www.drewry.co.uk/supply-chain-advisors/supply-chain-expertise/world-container-index-assessed-by-drewry"
        r2 = requests.get(url2, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        if r2.status_code == 200:
            # Parse tables from HTML
            tables = pd.read_html(StringIO(r2.text))
            if tables:
                for t in tables:
                    print(f"    Table shape: {t.shape}, cols: {list(t.columns)[:5]}")
                print(f"  ✗ WCI: page loaded but data extraction needs manual inspection")
            else:
                print(f"  ✗ WCI: no tables found")
        else:
            print(f"  ✗ WCI: HTTP {r2.status_code}")
    except Exception as e2:
        print(f"  ✗ WCI fallback: {e2}")

    # Second fallback: use SCFI via investing.com historical data
    print("  Trying SCFI via investing.com...")
    try:
        import yfinance as yf
        # SCFI is not on yfinance, but we can use the Invesco Shipping ETF (SEA)
        # as a container shipping proxy — it tracks container shipping companies
        sea = yf.download('SEA', start=START, progress=False, auto_adjust=True)
        if len(sea) > 0:
            monthly_sea = sea['Close'].resample('MS').last().dropna()
            if hasattr(monthly_sea, 'squeeze'):
                monthly_sea = monthly_sea.squeeze()
            series_data['shipping_etf_sea'] = monthly_sea
            print(f"  ✓ Invesco Shipping ETF (SEA) as container proxy: {len(monthly_sea)} monthly obs")
        # Also try ZIM (ZIM Integrated Shipping) as a container rate proxy
        zim = yf.download('ZIM', start=START, progress=False, auto_adjust=True)
        if len(zim) > 0:
            monthly_zim = zim['Close'].resample('MS').last().dropna()
            if hasattr(monthly_zim, 'squeeze'):
                monthly_zim = monthly_zim.squeeze()
            series_data['zim_container_proxy'] = monthly_zim
            print(f"  ✓ ZIM Integrated Shipping (container rate proxy): {len(monthly_zim)} monthly obs")
        # Also try MAERSK on Copenhagen (MAERSK-B.CO)
        maersk = yf.download('MAERSK-B.CO', start=START, progress=False, auto_adjust=True)
        if len(maersk) > 0:
            monthly_maersk = maersk['Close'].resample('MS').last().dropna()
            if hasattr(monthly_maersk, 'squeeze'):
                monthly_maersk = monthly_maersk.squeeze()
            series_data['maersk_proxy'] = monthly_maersk
            print(f"  ✓ Maersk B (container carrier proxy): {len(monthly_maersk)} monthly obs")
    except Exception as e3:
        print(f"  ✗ yfinance proxies: {e3}")

# ── 4. Combine and save ────────────────────────────────────────────────────
print("\nCombining all series...")
combined = pd.DataFrame(index=pd.date_range(start=START, end=END, freq='MS'))
for col, s in series_data.items():
    if isinstance(s, pd.Series):
        s.index = pd.to_datetime(s.index)
        # Normalise to month-start
        s.index = s.index.to_period('M').to_timestamp()
        combined[col] = s

# Forward fill minor gaps (max 2 months)
combined = combined.ffill(limit=2)

print(f"\nFinal dataset shape: {combined.shape}")
print(combined.tail(6).to_string())

out_csv = os.path.join(OUT_DIR, 'new_exog_v2.csv')
out_parq = os.path.join(OUT_DIR, 'new_exog_v2.parquet')
combined.to_csv(out_csv)
combined.to_parquet(out_parq)
print(f"\nSaved: {out_csv}")
print(f"Saved: {out_parq}")

# Check coverage
print("\nCoverage summary:")
for col in combined.columns:
    non_null = combined[col].notna().sum()
    first = combined[col].first_valid_index()
    last  = combined[col].last_valid_index()
    print(f"  {col:30s}: {non_null} obs, {first.strftime('%Y-%m') if first else 'N/A'} → {last.strftime('%Y-%m') if last else 'N/A'}")
