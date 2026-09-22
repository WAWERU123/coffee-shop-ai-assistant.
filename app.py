import os
import pandas as pd
from dotenv import load_dotenv
from google import genai
import streamlit as st

# Page configuration
st.set_page_config(page_title="Coffee Shop AI Assistant", layout="wide")

# 1. Load environment variables & initialize Gemini client securely
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# 2. Load dataset (cached for performance)
@st.cache_data
def load_data():
    return pd.read_csv('coffee_shop_sales_cleaned.csv')

df = load_data()

# 3. Pre-calculate comprehensive summary statistics for rich context
total_revenue = df['total_sales'].sum() if 'total_sales' in df.columns else 0
store_sales = df.groupby('store_location')['total_sales'].sum().to_dict() if 'store_location' in df.columns else {}
top_category = df.groupby('product_category')['total_sales'].sum().idxmax() if 'product_category' in df.columns else "N/A"

# Detailed store-by-product breakdown table for granular questions
store_product_breakdown = df.groupby(['store_location', 'product_category'])['transaction_id'].count().reset_index()

data_context = f"""
Summary data from the coffee shop dataset (149k rows):
- Total transactions: {len(df):,}
- Total overall revenue: ${total_revenue:,.2f}
- Revenue by store location: {store_sales}
- Top-performing product category overall: {top_category}
- Store-by-Product Order Counts (Sample/Full breakdown):
{store_product_breakdown.to_string()}
"""

# 4. Streamlit UI Dashboard Layout
st.title("☕ Coffee Shop Sales AI Assistant")
st.markdown("Explore your 149k-row sales dataset and ask questions powered by Gemini.")

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"${total_revenue:,.2f}")
col2.metric("Total Transactions", f"{len(df):,}")
col3.metric("Top Category", top_category)

st.divider()

# 5. Interactive Chat / Question Input
user_question = st.text_input(
    "Ask a question about your coffee sales data:", 
    placeholder="e.g., Which product has the lowest orders at Astoria?"
)

if user_question:
    with st.spinner("Gemini is analyzing your data..."):
        prompt = f"""
        You are an expert AI data analyst assistant for a coffee shop chain.
        Answer the user's question accurately based *only* on the provided data context. Keep the answer professional, clear, and concise.

        Data Context:
        {data_context}

        User Question: {user_question}
        """

        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt
            )
            st.success("Analysis Complete")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"An error occurred: {e}")
