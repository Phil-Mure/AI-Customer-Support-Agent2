#create a virtual environment with python3 -m venv .venv and activate with source .venv/bin/activate
# pip install -qU "langchain[google-genai]" langchain-openai langchain-core langgraph langchain-community beautifulsoup4 faiss-cpu pymysql sqlalchemy selenium pandas dotenv pymysql sqlalchemy mysql-connector-python playwright && playwright install

from ai import login, fill_submission_form, fill_application_form, fill_working_info_form, skip_guarantor_page, fill_reference_contact_form, fill_product_info_form, fill_dealer_and_tenure_fields
import faiss
import bs4
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.utilities import SQLDatabase
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from langchain_community.agent_toolkits import create_sql_agent
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from playwright.sync_api import sync_playwright
import getpass
import os
from langchain.chat_models import init_chat_model
import time
import requests

from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")

#  ---  Initializing the database with LangChain  ---
db = SQLDatabase.from_uri("mysql+mysqlconnector://root:FQfC$peBp_^2Y2L@178.128.49.31:3306/crm_002_db")
print(db.dialect)
print(db.get_usable_table_names())
#Create SQL Chain
db_chain = create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=True)

# ---  (OPTION) Initializing datase with pandas ---
db_url = "mysql+pymysql://root:FQfC$peBp_^2Y2L@178.128.49.31:3306/crm_002_db"
# Create the SQLAlchemy engine
engine = create_engine(db_url)

# Setting up currently logged in user and capturing the User ID
USER_ID = 9 # 9 is for testing purposes only. The actual user ID will be gotted when user clicks the "Ok" button in the desktop app
SELECTED_OPTION = "Chailease"
SELECTED_LOAN = "Chailease"
CHAILEASE_URL = "https://e-submission.chailease.com.my/" # Dashboard URL
LOGING_PAGE = "https://e-submission.chailease.com.my/login"
SUBMISSION_PAGE = "https://e-submission.chailease.com.my/submission"
FORM_URL = url = f"https://e-submission.chailease.com.my/apply?actionType=new&submissionNo=0&companyId={USER_ID}2&disableConvertUrl=true&lending=H&productCode=H-007-0000&productName=HP%20for%20Super%20Bike%20-%20New&productType=007"

# Querrying the DB to get login details to use at https://e-submission.chailease.com.my/login
login_details = db.run(f"""
     SELECT ID, AccountName, Password FROM LoanAccounts WHERE ID = {USER_ID};
""")
login_credentials = db_chain.invoke(login_details)
print(login_credentials)
credentials_result = login_credentials

# Getting the actual USser Account and Password for Login
results = credentials_result["input"]
# Convert string of database output to actual list
id = ast.literal_eval(results)[0][0]
username = ast.literal_eval(results)[0][1]
password = ast.literal_eval(results)[0][2]


# ----  Page 1 of 7: Filling Personal Data --- 
# Fetch data from Personal Info table for User ID {USER_ID}
query = f"""
  SELECT "Address", "Bumi", "Email", "Gender", "ID", "Loan Status", "Marital Status",
        "Name", "No of year in residence", "NRIC", "Ownership Status", "Phone Number",
        "Race", "Stay in registered address", "Timestamp", "Title",
        "Where user stay(If not stay in registered address)"
  FROM "Personal Info"
WHERE ID = {USER_ID}
"""
db_result = db_chain.run(query)
personal_info = parse_to_dict(db_result)

# ---  Employment Data Page 2/7 ---

#Getting Employment Info from Database
# Fetch Working Info where ID = USER_ID
query = f"""
SELECT * FROM "Working Info" WHERE ID = {USER_ID}
"""
working_info_raw = db_chain.run(query)
working_info = parse_to_dict(working_info_raw)
print(working_info)


#  ---- Skip the Guarantor: page 3/7 ----


# -- Reference Contact Query: Page 4/7 ---
ref_df = pd.read_sql(f'SELECT * FROM `Reference Contact` WHERE ID = {USER_ID}', engine)
ref_contact = ref_df.iloc[0]


#  ---  Page 5/7: Collateral ---
# Get product info for given USER ID
product_df = pd.read_sql(f"""SELECT "Brand", "Down Payment", "ID", "Model", "NRIC", "Number Plate", "Price", "Product Type", 
"Tenure" FROM `Product Info` WHERE ID = {USER_ID}""", engine)
product_info = product_df.iloc[:, :]

model_df = pd.read_sql(f"""SELECT "Chailease" FROM `Model Map` WHERE "Webform" = '{product_info.get("Model")}' AND ID = {USER_ID}""", engine)
model_result = model_df.iloc[:, :]
model_result


# --- Page 6/7 Terms and Conditions ---
product_query = f"""
SELECT "Brand", "Down Payment", "ID", "Model", "NRIC", "Number Plate", "Price", "Product Type", "Tenure"
FROM "Product Info"
WHERE NRIC = {USER_ID}
"""
product_info_raw = db_chain.run(product_query)
product_info = parse_to_dict(product_info_raw)
print(product_info)



def main():
    """"Automating the entire scrapping logic"""
    with sync_playwright() as p:
        # Launch browser (headless or not)
        browser = p.chromium.launch(headless=True)

        # Create one shared browser context
        context = browser.new_context()

        # Create one page for the session (can be reused or recreated)
        page = context.new_page()

        # Perform login (pass page or context as needed)
        login(page, username, password)

        # Initial form submission after login
        fill_submission_form(page)

        # Use the same context (same cookies/session) for all steps
        fill_application_form(context, personal_info) # Personal Info Page 1/7
        fill_working_info_form(context, working_info) # Employment Info Page 2/7
        skip_guarantor_page(page) # Guarantor: Page 3/7
        fill_reference_contact_form(page, context, ref_contact) # Reference Page 4/7
        fill_product_info_form(context, product_info, model_name) # Collateral Page 5/7
        fill_dealer_and_tenure_fields(context, product_info) # Sales Terms Page 6/7

        # Optional: submit application or close browser
        # context.pages[-1].get_by_role("button", name="Submit").click()

        browser.close()
    print("Form filled successfully!")

    if __name__ == "__main__":
      main()

    # run "chailease_bot.py" in terminal to run this script
