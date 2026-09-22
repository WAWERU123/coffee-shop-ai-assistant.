# Coffee Shop Sales AI Assistant ☕🤖

An interactive, data-driven web application and AI assistant built to analyze retail transaction trends and deliver real-time business insights from coffee shop sales data.

---

## 🚀 Key Features
- **Interactive Sales Dashboard:** Built with **Streamlit** to visualize revenue trends, peak operating hours, product category performance, and customer purchasing patterns.
- **Natural Language AI Assistant:** Integrated with Google's **Gemini API** to allow stakeholders to query sales data conversationally and receive instant, context-aware analytics.
- **Data Pipeline & Cleaning:** Features structured data processing scripts to handle raw transactional logs, clean missing values, and prepare high-integrity datasets for modeling.

---

## 🛠️ Tech Stack & Tools
- **Language:** Python
- **Frontend / UI:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **AI / LLM Integration:** Google GenAI SDK (Gemini)
- **Version Control:** Git & GitHub



## 📁 Project Architecture & File Structure
```text
coffee-shop-ai-assistant/
│
├── app.py                      # Main Streamlit web application interface
├── coffe_sales.ipynb           # Jupyter notebook for exploratory data analysis (EDA)
├── Coffee Shop Sales.csv       # Raw transaction dataset
├── coffee_shop_sales_cleaned.csv # Processed and cleaned dataset ready for querying
├── .gitignore                  # Excludes sensitive environment files and checkpoints
└── README.md                   # Project documentation
