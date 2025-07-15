import torch
import mysql.connector
import pandas as pd
import plotly.express as px
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# XAMPP MySQL Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),  
    "database": os.getenv("DB_DATABASE"),
    "password": os.getenv("DB_PASSWORD")
}

MODEL_NAME = "defog/sqlcoder-7b"
SUMMARY_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# Define Schema Structure here
# Important Note: Be descriptive in the naming of columns and tables
SCHEMA = """
Tables:
- studies(id, title, author, year, citations, domain)
- experiments(id, study_id, method, p_value, effect_size)
Relationships:
- studies.id → experiments.study_id
"""

def load_model():
    print("⚙️ Loading 7B model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"✅ QUERY Model loaded | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tokenizer

# def load_summary_model():
#     print("⚙️ Loading summarization model...")
#     sum_model = AutoModelForCausalLM.from_pretrained(SUMMARY_MODEL, device_map="auto", torch_dtype=torch.float16)
#     sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL)
#     print(f"✅ SUMMARY Model loaded | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
#     return sum_model, sum_tokenizer

class MySQLDatabase:
    def __init__(self):
        self.conn = mysql.connector.connect(**DB_CONFIG)
    
    def execute_query(self, sql):
        cursor = self.conn.cursor(dictionary=True)  # Get results as dicts
        cursor.execute(sql)
        
        if cursor.with_rows:
            return pd.DataFrame(cursor.fetchall())
        return None

# def generate_sql(question, model, tokenizer):
#     prompt = f"""### MySQL Task:
# Convert to MySQL-compatible SQL:
# Question: {question}

# ### Schema:
# {SCHEMA}

# ### SQL:
# SELECT"""
    
#     # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
#     outputs = model.generate(**inputs, max_new_tokens=200)
#     # return "SELECT" + tokenizer.decode(outputs[0]).split("SELECT")[-1]
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     sql_start = decoded.find("SELECT")
#     if sql_start == -1:
#         raise ValueError("No SELECT statement found.")
#     cleaned_sql = decoded[sql_start:].strip().replace("NULLS LAST", "")
#     return cleaned_sql

def generate_sql(question, model, tokenizer):
    prompt = f"""### MySQL Task:
Convert to MySQL-compatible SQL:
Question: {question}

### Schema:
{SCHEMA}

### SQL:
SELECT"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    sql_start = decoded.find("SELECT")
    if sql_start == -1:
        raise ValueError("No SELECT statement found.")
    
    raw_sql = decoded[sql_start:].strip()

    # 🛠️ Apply basic PostgreSQL-to-MySQL compatibility patches
    fixed_sql = raw_sql

    # Replace date_trunc('year', to_timestamp(year, 'YYYY')) → year
    fixed_sql = re.sub(r"date_trunc\('year',\s*to_timestamp\((\w+),\s*'YYYY'\)\)", r"\1", fixed_sql, flags=re.IGNORECASE)
    
    # Remove or replace any lingering to_timestamp(...)
    fixed_sql = re.sub(r"to_timestamp\(([^,]+),\s*'[^']*'\)", r"\1", fixed_sql, flags=re.IGNORECASE)

    fixed_sql = decoded[sql_start:].strip().replace("NULLS LAST", "")

    return fixed_sql



# def visualize(df, question):
#     if "trend" in question.lower():
#         fig = px.line(df, x=df.columns[0], y=df.columns[1])
#     elif "compar" in question.lower():
#         fig = px.bar(df, x=df.columns[0], y=df.columns[1], barmode='group')
#     else:
#         fig = px.scatter(df.head(20))
#     fig.show()
def visualize(df, question):
    numeric_cols = df.select_dtypes(include='number').columns
    non_numeric_cols = df.select_dtypes(exclude='number').columns

    if df.empty or len(df.columns) < 2:
        print("📉 Not enough data to plot.")
        return

    # Use the first non-numeric as x (e.g., category), and first numeric as y
    if "trend" in question.lower() or "over time" in question.lower():
        if len(numeric_cols) >= 1 and len(non_numeric_cols) >= 1:
            fig = px.line(df, x=non_numeric_cols[0], y=numeric_cols[0])
        else:
            print("📉 Can't plot line chart due to missing numeric or categorical data.")
            return
    elif "compar" in question.lower():
        if len(numeric_cols) >= 1 and len(non_numeric_cols) >= 1:
            fig = px.bar(df, x=non_numeric_cols[0], y=numeric_cols[0], barmode='group')
        else:
            print("📊 Can't plot bar chart due to missing numeric or categorical data.")
            return
    else:
        # Fallback to plotting first 2 numeric columns, if available
        if len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
        else:
            print("📉 Not enough numeric columns to plot.")
            return

    fig.show()


# def generate_natural_response(question, df, sum_model, sum_tokenizer):
#     table_str = df.head(5).to_string(index=False)
#     prompt = f"""
# You are a helpful assistant. Here's a user question and sample data. Summarize the key findings in 2-4 friendly sentences.

# Question: {question}

# Sample Data:
# {table_str}

# Summary:
# """

#     inputs = sum_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
#     outputs = sum_model.generate(
#         **inputs,
#         max_new_tokens=150,
#         temperature=0.7,
#         do_sample=True,
#         pad_token_id=sum_tokenizer.eos_token_id,
#         eos_token_id=sum_tokenizer.eos_token_id
#     )
#     summary = sum_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return summary.split("Summary:")[-1].strip()

def generate_natural_response(question, df, model, tokenizer):
    table_str = df.head(5).to_string(index=False)

    prompt = f"""
### User Question:
{question}

### Sample Data:
{table_str}

### Task:
Write a short, friendly, natural-language summary of the key findings in 2-3 sentences.

Summary:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.split("Summary:")[-1].strip()



def main():
    model, tokenizer = load_model()
    # sum_model, sum_tokenizer = load_summary_model()
    db = MySQLDatabase()
    
    # print("\nTry these sample questions:")
    # print("1. Show CRISPR studies with p-value < 0.05")
    # print("2. Compare citation counts by domain")
    # print("3. Exit")
    print("\nThis is a SAMPLE SCHEMA ONLY!")
    print("Tables:")
    print("\tstudies(id, title, author, year, citations, domain)")
    print("\texperiments(id, study_id, method, p_value, effect_size)")
    
    while True:
        question = input("\n🔍 Your question: ").strip()
        if question.lower() in ('3', 'exit', 'quit'):
            break
            
        try:
            sql = generate_sql(question, model, tokenizer)
            print(f"\nGenerated MySQL:\n{sql}")
            
            results = db.execute_query(sql)
            if results is None:
                print("✅ Executed (no results)")
                continue
                
            print(f"\n📊 Results ({len(results)} rows):")
            print(results.head())
            
            if not results.empty:
                print("\n🗣️ Summary:")
                print(generate_natural_response(question, results, model, tokenizer))
                visualize(results, question)
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()