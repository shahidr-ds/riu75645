import io
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Multi-Company Stock Analyzer",
    page_icon="📈",
    layout="wide",
)

COMPANIES = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Alphabet (Google)": "GOOGL",
    "NVIDIA": "NVDA",
    "Meta Platforms": "META",
    "Tesla": "TSLA",
    "Walmart": "WMT",
    "JPMorgan Chase": "JPM",
    "Berkshire Hathaway (Class B)": "BRK-B",
    "Coca-Cola": "KO",
    "Netflix": "NFLX",
    "Intel": "INTC",
    "Cisco": "CSCO",
    "Exxon Mobil": "XOM",
}


@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers: tuple, start_date: date, end_date_exclusive: date):
    """Download daily historical prices. Yahoo's end date is exclusive."""
    data = yf.download(
        tickers=list(tickers),
        start=start_date.isoformat(),
        end=end_date_exclusive.isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    # yfinance returns a MultiIndex for multiple tickers and may do so for one ticker too.
    if isinstance(data.columns, pd.MultiIndex):
        level0 = list(data.columns.get_level_values(0))
        level1 = list(data.columns.get_level_values(1))
        if "Adj Close" in level0:
            prices = data["Adj Close"].copy()
        elif "Close" in level0:
            prices = data["Close"].copy()
        elif "Adj Close" in level1:
            prices = data.xs("Adj Close", axis=1, level=1).copy()
        elif "Close" in level1:
            prices = data.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        column = "Adj Close" if "Adj Close" in data.columns else "Close"
        if column not in data.columns:
            return pd.DataFrame()
        prices = data[[column]].rename(columns={column: tickers[0]})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    # Keep only selected tickers and sort the date index.
    available = [ticker for ticker in tickers if ticker in prices.columns]
    prices = prices.loc[:, available].sort_index()
    prices = prices.dropna(how="all")
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def download_company_profiles(tickers: tuple) -> pd.DataFrame:
    """Fetch company profile, leadership, shares and financial details from Yahoo Finance."""
    rows = []
    for ticker_symbol in tickers:
        row = {
            "Ticker": ticker_symbol,
            "CEO": None,
            "CEO reported annual compensation (USD)": None,
            "Share price (USD)": None,
            "Market capitalization (USD)": None,
            "Enterprise value (USD)": None,
            "Shares outstanding": None,
            "Float shares": None,
            "Shares short": None,
            "Accounting net worth / shareholders' equity (USD)": None,
            "Total assets (USD)": None,
            "Total liabilities (USD)": None,
            "Total cash (USD)": None,
            "Total debt (USD)": None,
            "Revenue (USD)": None,
            "Net income (USD)": None,
            "Trailing EPS (USD)": None,
            "Trailing P/E": None,
            "Forward P/E": None,
            "Dividend yield (%)": None,
            "52-week low (USD)": None,
            "52-week high (USD)": None,
            "Employees": None,
            "Sector": None,
            "Industry": None,
            "Website": None,
            "Data source": "Yahoo Finance",
        }
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info or {}

            officers = info.get("companyOfficers") or []
            ceo = next(
                (o for o in officers
                 if "chief executive officer" in str(o.get("title", "")).lower()
                 or str(o.get("title", "")).strip().lower() == "ceo"
                 or " ceo" in str(o.get("title", "")).lower()),
                None,
            )
            if ceo:
                row["CEO"] = ceo.get("name")
                # totalPay is total compensation, not necessarily base salary.
                row["CEO reported annual compensation (USD)"] = (
                    ceo.get("totalPay") if ceo.get("totalPay") is not None else ceo.get("salary")
                )

            field_map = {
                "Share price (USD)": "currentPrice",
                "Market capitalization (USD)": "marketCap",
                "Enterprise value (USD)": "enterpriseValue",
                "Shares outstanding": "sharesOutstanding",
                "Float shares": "floatShares",
                "Shares short": "sharesShort",
                "Accounting net worth / shareholders' equity (USD)": "totalStockholderEquity",
                "Total assets (USD)": "totalAssets",
                "Total liabilities (USD)": "totalLiabilities",
                "Total cash (USD)": "totalCash",
                "Total debt (USD)": "totalDebt",
                "Revenue (USD)": "totalRevenue",
                "Net income (USD)": "netIncomeToCommon",
                "Trailing EPS (USD)": "trailingEps",
                "Trailing P/E": "trailingPE",
                "Forward P/E": "forwardPE",
                "Dividend yield (%)": "dividendYield",
                "52-week low (USD)": "fiftyTwoWeekLow",
                "52-week high (USD)": "fiftyTwoWeekHigh",
                "Employees": "fullTimeEmployees",
                "Sector": "sector",
                "Industry": "industry",
                "Website": "website",
            }
            for output_name, info_key in field_map.items():
                value = info.get(info_key)
                if output_name == "Dividend yield (%)" and value is not None:
                    value = value * 100
                row[output_name] = value

            # Yahoo's profile dictionary may omit balance-sheet totals; use the latest
            # balance sheet as a fallback where available.
            if any(row[k] is None for k in (
                "Accounting net worth / shareholders' equity (USD)",
                "Total assets (USD)", "Total liabilities (USD)"
            )):
                try:
                    balance_sheet = stock.balance_sheet
                    if balance_sheet is not None and not balance_sheet.empty:
                        latest_col = balance_sheet.columns[0]
                        aliases = {
                            "Accounting net worth / shareholders' equity (USD)": [
                                "Stockholders Equity", "Total Stockholder Equity",
                                "Total Equity Gross Minority Interest",
                            ],
                            "Total assets (USD)": ["Total Assets"],
                            "Total liabilities (USD)": [
                                "Total Liabilities Net Minority Interest", "Total Liabilities"
                            ],
                        }
                        for output_name, candidates in aliases.items():
                            if row[output_name] is None:
                                for candidate in candidates:
                                    if candidate in balance_sheet.index:
                                        value = balance_sheet.loc[candidate, latest_col]
                                        if pd.notna(value):
                                            row[output_name] = float(value)
                                            break
                except Exception:
                    pass
        except Exception:
            # Keep the company in the table even if its profile endpoint fails.
            pass
        rows.append(row)
    return pd.DataFrame(rows)


def make_excel(prices: pd.DataFrame, ticker_to_name: dict, company_profiles: pd.DataFrame = None) -> bytes:
    output = io.BytesIO()
    export = prices.copy()
    export.columns = [f"{ticker_to_name.get(c, c)} ({c})" for c in export.columns]
    export.index = pd.to_datetime(export.index).date
    export.index.name = "Date"

    with pd.ExcelWriter(output, engine="openpyxl", datetime_format="yyyy-mm-dd") as writer:
        export.to_excel(writer, sheet_name="Closing Prices")
        returns = export.pct_change() * 100
        returns.to_excel(writer, sheet_name="Daily Change (%)")
        summary = pd.DataFrame({
            "Company / Ticker": export.columns,
            "First available price": [export[c].dropna().iloc[0] if export[c].notna().any() else None for c in export.columns],
            "Last available price": [export[c].dropna().iloc[-1] if export[c].notna().any() else None for c in export.columns],
            "Change (%)": [
                ((export[c].dropna().iloc[-1] / export[c].dropna().iloc[0]) - 1) * 100
                if export[c].notna().sum() >= 2 and export[c].dropna().iloc[0] != 0 else None
                for c in export.columns
            ],
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)
        if company_profiles is not None and not company_profiles.empty:
            company_profiles.to_excel(writer, sheet_name="Company Details", index=False)
    return output.getvalue()


st.title("📈 Multi-Company Stock Analyzer")
st.write(
    "Select companies and a date range to compare historical stock prices. "
    "Data is retrieved from Yahoo Finance using `yfinance`."
)

with st.sidebar:
    st.header("Analysis settings")
    selected_names = st.multiselect(
        "Select one or more companies",
        options=list(COMPANIES.keys()),
        default=["Apple", "Microsoft", "NVIDIA"],
        help="You can select multiple companies.",
    )

    today = date.today()
    default_start = today - timedelta(days=365)
    start_date = st.date_input("Start date", value=default_start, max_value=today)
    end_date = st.date_input("End date", value=today, min_value=start_date, max_value=today)

    chart_mode = st.radio(
        "Chart type",
        ["Closing price (USD)", "Normalized performance (start = 100)"],
        help="Normalized performance compares percentage movement from the first available date.",
    )
    download_button_label = "Download Excel file"

    run_analysis = st.button("Download and analyze", type="primary", use_container_width=True)

if not selected_names:
    st.info("Select at least one company from the sidebar.")
    st.stop()

if start_date > end_date:
    st.error("The start date must be on or before the end date.")
    st.stop()

selected_tickers = tuple(COMPANIES[name] for name in selected_names)
ticker_to_name = {COMPANIES[name]: name for name in selected_names}

# Load on first visit and whenever the settings change; button offers a deliberate refresh.
settings_key = (selected_tickers, start_date, end_date)
if "loaded_settings" not in st.session_state:
    st.session_state.loaded_settings = None
if "prices" not in st.session_state:
    st.session_state.prices = pd.DataFrame()

if run_analysis or st.session_state.loaded_settings != settings_key:
    end_exclusive = end_date + timedelta(days=1)
    with st.spinner("Fetching historical prices from Yahoo Finance..."):
        try:
            prices = download_prices(selected_tickers, start_date, end_exclusive)
            st.session_state.prices = prices
            st.session_state.loaded_settings = settings_key
        except Exception as exc:
            st.error(f"Could not download data. Please try again. Details: {exc}")
            st.stop()

prices = st.session_state.prices

if prices.empty:
    st.warning(
        "No price data was returned for this selection and date range. "
        "Check the dates/tickers or try again later."
    )
    st.stop()

missing = [t for t in selected_tickers if t not in prices.columns or prices[t].dropna().empty]
if missing:
    st.warning("No data was available for: " + ", ".join(f"{ticker_to_name.get(t, t)} ({t})" for t in missing))

display_prices = prices.rename(columns={t: ticker_to_name.get(t, t) for t in prices.columns})
if chart_mode == "Normalized performance (start = 100)":
    chart_data = display_prices.copy()
    for col in chart_data.columns:
        first_valid = chart_data[col].dropna()
        if not first_valid.empty and first_valid.iloc[0] != 0:
            chart_data[col] = chart_data[col] / first_valid.iloc[0] * 100
        else:
            chart_data[col] = pd.NA
    y_title = "Indexed price (first available value = 100)"
else:
    chart_data = display_prices
    y_title = "Adjusted closing price (USD)" if "Closing price" in chart_mode else "Price"

chart_long = (
    chart_data.reset_index()
    .melt(id_vars="Date", var_name="Company", value_name="Price")
    .dropna(subset=["Price"])
)

fig = px.line(
    chart_long,
    x="Date",
    y="Price",
    color="Company",
    title="Historical Stock Price Comparison" if chart_mode.startswith("Closing") else "Relative Performance Comparison",
    labels={"Price": y_title, "Date": "Date"},
    template="plotly_white",
)
fig.update_layout(legend_title_text="Company", hovermode="x unified")
fig.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)

# Key metrics for each selected company.
metric_cols = st.columns(min(len(display_prices.columns), 4))
for i, company in enumerate(display_prices.columns):
    series = display_prices[company].dropna()
    if series.empty:
        continue
    change_pct = ((series.iloc[-1] / series.iloc[0]) - 1) * 100 if len(series) >= 2 and series.iloc[0] else None
    with metric_cols[i % len(metric_cols)]:
        st.metric(
            label=f"{company} · latest available",
            value=f"${series.iloc[-1]:,.2f}",
            delta=f"{change_pct:+.2f}% over selected period" if change_pct is not None else None,
        )


st.subheader("Company details, CEO, shares and financials")
st.caption(
    "Net worth is shown as shareholders' equity when available; market capitalization is the "
    "market value of the company's shares and is not the same as accounting net worth. CEO pay "
    "may be total compensation rather than base salary. Figures depend on Yahoo Finance coverage."
)
with st.spinner("Fetching CEO, share-count and financial details..."):
    company_profiles = download_company_profiles(selected_tickers)

company_profiles.insert(
    0, "Company",
    [ticker_to_name.get(t, t) for t in company_profiles["Ticker"]]
)

money_columns = [
    "CEO reported annual compensation (USD)", "Share price (USD)",
    "Market capitalization (USD)", "Enterprise value (USD)",
    "Accounting net worth / shareholders' equity (USD)", "Total assets (USD)",
    "Total liabilities (USD)", "Total cash (USD)", "Total debt (USD)",
    "Revenue (USD)", "Net income (USD)", "Trailing EPS (USD)",
    "52-week low (USD)", "52-week high (USD)",
]
profile_display = company_profiles.copy()
for column in money_columns:
    if column in profile_display.columns:
        profile_display[column] = profile_display[column].apply(
            lambda value: f"${value:,.2f}" if pd.notna(value) else "Not available"
        )
for column in ["Shares outstanding", "Float shares", "Shares short", "Employees"]:
    if column in profile_display.columns:
        profile_display[column] = profile_display[column].apply(
            lambda value: f"{value:,.0f}" if pd.notna(value) else "Not available"
        )
for column in ["Trailing P/E", "Forward P/E", "Dividend yield (%)"]:
    if column in profile_display.columns:
        profile_display[column] = profile_display[column].apply(
            lambda value: f"{value:,.2f}" + ("%" if column == "Dividend yield (%)" else "")
            if pd.notna(value) else "Not available"
        )
profile_display = profile_display.fillna("Not available")
st.dataframe(profile_display, use_container_width=True, hide_index=True)

st.subheader("Historical data")
table = display_prices.copy()
table.index = pd.to_datetime(table.index).date
table.index.name = "Date"
st.dataframe(table, use_container_width=True)

excel_bytes = make_excel(prices, ticker_to_name, company_profiles)
st.download_button(
    label=download_button_label,
    data=excel_bytes,
    file_name=f"stock_data_{start_date.isoformat()}_to_{end_date.isoformat()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

st.caption(
    "Data source: Yahoo Finance via yfinance. Historical data may be delayed, incomplete, "
    "or revised. This application is for educational and informational use only and is not investment advice."
)
