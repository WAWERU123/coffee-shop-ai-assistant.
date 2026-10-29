import os
import pandas as pd
from dotenv import load_dotenv
from google import genai
import streamlit as st

# Page configuration
st.set_page_config(page_title="Coffee Shop AI Analytics Assistant", layout="wide")

# 1. Load environment variables & initialize Gemini client securely
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# 2. Load dataset (cached for performance)
@st.cache_data
def load_data():
    df = pd.read_csv('coffee_shop_sales_cleaned.csv')
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['month'] = df['transaction_date'].dt.month_name()
    return df

df = load_data()

# Pre-calculate high-level metrics for dashboard cards
total_revenue = df['total_sales'].sum() if 'total_sales' in df.columns else 0
total_transactions = len(df)
top_category = df.groupby('product_category')['total_sales'].sum().idxmax() if 'product_category' in df.columns else "N/A"

# ==========================================
# PRIORITY 4: DYNAMIC ANALYTICAL FUNCTIONS
# ==========================================

def revenue_by_store():
    return (
        df.groupby("store_location")
        ["total_sales"]
        .sum()
        .sort_values(ascending=False)
    )

def monthly_sales_by_store():
    return df.groupby(["store_location", "month"])["total_sales"].sum().reset_index()

def top_categories_performance():
    return df.groupby("product_category").agg(
        total_revenue=('total_sales', 'sum'),
        total_qty=('transaction_qty', 'sum')
    ).sort_values(by='total_revenue', ascending=False)

def peak_transaction_hours():
    return df.groupby("hour")["transaction_id"].count().sort_values(ascending=False)

# ==========================================
# DYNAMIC INTENT ROUTER
# ==========================================

def get_dynamic_context(question):
    q = question.lower()
    
    # Direct function matching based on user intent keywords
    if "highest revenue" in q or "revenue by store" in q or "generates more revenue" in q:
        result = revenue_by_store()
        return f"Live Dataset Output (Revenue by Store):\n{result.to_string()}"
        
    elif "month" in q or "lowest sale" in q or "lowest sales" in q or "trend" in q:
        result = monthly_sales_by_store()
        # Filter granularly if a specific store is named
        if "lower manhattan" in q:
            filtered = result[result["store_location"].str.lower().str.contains("lower manhattan")]
            return f"Live Dataset Output (Monthly Sales - Lower Manhattan):\n{filtered.to_string(index=False)}"
        elif "hell's kitchen" in q or "hells kitchen" in q:
            filtered = result[result["store_location"].str.lower().str.contains("hell")]
            return f"Live Dataset Output (Monthly Sales - Hell's Kitchen):\n{filtered.to_string(index=False)}"
        elif "astoria" in q:
            filtered = result[result["store_location"].str.lower().str.contains("astoria")]
            return f"Live Dataset Output (Monthly Sales - Astoria):\n{filtered.to_string(index=False)}"
        return f"Live Dataset Output (Monthly Sales by Store):\n{result.to_string(index=False)}"
        
    elif "category" in q or "product" in q or "top" in q:
        result = top_categories_performance()
        return f"Live Dataset Output (Category Performance):\n{result.to_string()}"
        
    elif "hour" in q or "time" in q or "busy" in q or "peak" in q:
        result = peak_transaction_hours()
        return f"Live Dataset Output (Peak Hours):\n{result.head(5).to_string()}"
        
    # Fallback high-level snapshot if no explicit rule matches
    return f"""
    Live Dataset Snapshot:
    - Total Transactions: {len(df):,}
    - Total Revenue: ${df['total_sales'].sum():,.2f}
    - Revenue by Store: {revenue_by_store().to_dict()}
    """

# ==========================================
# STREAMLIT UI & GEMINI INTEGRATION
# ==========================================

st.title("☕ Coffee Shop Sales AI Analytics Engine")
st.markdown("Powered by **Live Pandas Functions** and **Google Gemini**.")

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"${total_revenue:,.2f}")
col2.metric("Total Transactions", f"{total_transactions:,}")
col3.metric("Top Category", top_category)

st.divider()

user_question = st.text_input(
    "Ask an analytical question about your coffee sales data:", 
    placeholder="e.g., which month had the lowest sale in Lower Manhattan"
)

if user_question:
    with st.spinner("Executing live pandas analytical functions..."):
        # Dynamically retrieve fresh data straight from dataset functions
        live_data_context = get_dynamic_context(user_question)
        
        prompt = f"""
        You are an expert AI data analyst assistant for a coffee shop chain.
        Answer the user's question accurately using *only* the dynamic data output below, which was queried live via pandas analytical functions. 
        Never guess or generalize; cite the exact figures and months from the live result.

        Live Pandas Function Output:
        {live_data_context}

        User Question: {user_question}
        """

        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt
            )
            st.success("Analysis Complete (Dynamic Pandas Execution)")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"An error occurred: {e}")
