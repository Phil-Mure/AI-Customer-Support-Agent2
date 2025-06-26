# Installing necessary modules
# pip install -qU "langchain[google-genai]" langchain-openai langchain-core langgraph langchain-community beautifulsoup4 faiss-cpu pymysql sqlalchemy selenium pandas pymysql sqlalchemy mysql-connector-python playwright && playwright install

# Importing necessary modules
# Importing necessary modules
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

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

os.environ["LANGSMITH_TRACING"] = "true"
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass()



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

#reCAPTCHA with Playwright. Playwright is light and Faster than Selenium. Changed to Playwright + Requests + BeautifulSoup 
from playwright.sync_api import sync_playwright
import time
import requests

# ---  LOGIN Page with reCAPTCHA ---
def login(page, username, password):
    page.goto("https://e-submission.chailease.com.my/login")

    page.fill("#account", username)
    page.fill("#userPassword", password)

    # Wait for iframe with reCAPTCHA
    iframe_element = page.frame_locator("iframe[src*='recaptcha']")
    iframe = iframe_element.frame()

    # Click on the checkbox inside the iframe
    iframe.locator("#ngrecaptcha-0").click()

    # Optional: wait and observe
    page.wait_for_timeout(3000)

    cookies = context.cookies()

    print("Login Successfull")  

# Python code to convert database output to Python Dictionary
def parse_to_dict(text: str) -> dict:
    data = {}
    lines = text.strip().splitlines()
    for line in lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            value = value.strip()
            # Convert values
            if value.lower() == "none":
                value = None
            elif value.isdigit():
                value = int(value)
            data[key.strip()] = value
    return data
  

# ----   Page 1/7 ---
#After Login, the following is the submission page ---
def fill_submission_form(page):
    page.goto("https://e-submission.chailease.com.my/submission")
    time.sleep(3)

    # === Step 1: Select radio with value="new"
    page.locator('input[name="action"][value="new"]').check()

    # === Step 2: Select dropdown with "HP for Super - New"
    dropdown = page.locator('select.ng-tns-c77-16')  
    dropdown.select_option(label="HP for Super - New")

    # === Step 3: Fill ID No
    page.get_by_label("ID No").fill(f"{USER_ID}")

    # === Step 4: Click Next
    page.get_by_role("button", name="Next").click()
    page.wait_for_timeout(3000)

    print("First Submission!.")


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

def fill_application_form(context, personal_info):
    page = context.new_page()
    page.goto(FORM_URL)
    page.wait_for_timeout(3000)
    # Map DB fields to formcontrolnames
    field_map = {
        "idNo": personal_info.get("NRIC"),
        "firstname": personal_info.get("Name"),
        "gender": personal_info.get("Gender"),
        "email": personal_info.get("Email"),
        "mobileNo": personal_info.get("Phone Number"),
        "residentialAddress": personal_info.get("Address"),
        "race": personal_info.get("Race"),
        "maritalStatus": personal_info.get("Marital Status"),
        "ownershipStatus": personal_info.get("Ownership Status"),
        "yearOfStay": personal_info.get("No of year in residence"),
        "stayInRegisteredAddress": personal_info.get("Stay in registered address"),
        "altStayAddress": personal_info.get("Where user stay(If not stay in registered address)"),
        "isBumiputera": personal_info.get("Bumi"),
        "title": personal_info.get("Title")
    }

    # Fill all matching fields
    for control_name, value in field_map.items():
        try:
            if value is not None:
                page.locator(f'[formcontrolname="{control_name}"]').fill(str(value))
        except Exception as e:
            print(f"[WARN] Could not fill {control_name}: {e}")

    print("Personal Info filled successfully.")


# ---  Employment Data Page 2/7 ---

#Getting Employment Info from Database
# Fetch Working Info where ID = USER_ID
query = f"""
SELECT * FROM "Working Info" WHERE ID = {USER_ID}
"""
working_info_raw = db_chain.run(query)
working_info = parse_to_dict(working_info_raw)
print(working_info)

def fill_working_info_form(context, working_info):
    page = context.pages
    page.goto(FORM_URL)

    # 1. Occupation (Position)
    if working_info.get("Position"):
        page.locator('#occupation').click()
        page.locator(f'text="{working_info["Position"]}"').click()

    # 2. Employer Name
    if working_info.get("Company Name"):
        page.locator('input[name="employerName"]').fill(working_info["Company Name"])

    # 3. Monthly Income
    if working_info.get("Net Salary"):
        page.get_by_label("Monthly Income").fill(str(working_info["Net Salary"]))

    # 4. Work Address
    if working_info.get("Company Address"):
        page.get_by_label("Work Address").fill(working_info["Company Address"])

    # 5. Work Phone No.
    if working_info.get("Company Phone Number"):
        page.get_by_label("Work Phone No.").fill(working_info["Company Phone Number"])

    # 6. Working in Singapore radio button
    if working_info.get("Working in Singapore"):
        answer = str(working_info["Working in Singapore"]).strip().lower()
        if "yes" in answer or "true" in answer or answer == "1":
            page.locator('input[name="workInSingapore"][value="true"]').check()
        elif "no" in answer or answer == "0":
            page.locator('input[name="workInSingapore"][value="false"]').check()

    print("[INFO] Working Info section filled.")

#  ---- Skip the Guarantor: page 3/7 ----
def skip_guarantor_page(page):
    page.goto(FORM_URL)
    try:
        # Try clicking the "Next" or "Skip" button
        page.get_by_role("button", name="Next").click()
        page.wait_for_timeout(2000)
        print("[INFO] Skipped Guarantor page using 'Next' button.")
    except:
        try:
            page.get_by_role("button", name="Skip").click()
            page.wait_for_timeout(2000)
            print("[INFO] Skipped Guarantor page using 'Skip' button.")
        except Exception as e:
            print(f"[WARN] Could not skip Guarantor page: {e}")


# -- Reference Contact Query: Page 4/7 ---
ref_df = pd.read_sql(f'SELECT * FROM `Reference Contact` WHERE ID = {USER_ID}', engine)
ref_contact = ref_df.iloc[0]
#Filling Reference form
def fill_reference_contact_form(page, context, ref_contact):
    page.goto(FORM_URL)
    try:
        if ref_contact.get("Name"):
            page.locator('[formcontrolname="firstName"]').fill(ref_contact["Name"])

        if ref_contact.get("Phone Number"):
            page.locator('[formcontrolname="mobilePhone"]').fill(ref_contact["Phone Number"])

        if ref_contact.get("Relation to user"):
            # Click and select value from dropdown (PrimeNG)
            page.locator('[formcontrolname="relationship"]').click()
            page.locator(f'text="{ref_contact["Relation to user"]}"').click()

        print("[INFO] Reference Contact form filled.")
    except Exception as e:
        print(f"[ERROR] Could not fill Reference Contact section: {e}")


#  ---  Page 5/7: Collateral ---
# Get product info for given USER ID
product_df = pd.read_sql(f"""SELECT "Brand", "Down Payment", "ID", "Model", "NRIC", "Number Plate", "Price", "Product Type", 
"Tenure" FROM `Product Info` WHERE ID = {USER_ID}""", engine)
product_info = product_df.iloc[:, :]

model_df = pd.read_sql(f"""SELECT "Chailease" FROM `Model Map` WHERE "Webform" = '{product_info.get("Model")}' AND ID = {USER_ID}""", engine)
model_result = model_df.iloc[:, :]
model_result


model_name = model_result.get("Chailease")
model_name
def fill_product_info_form(context, product_info, model_name):
    page = context.new_page()
    page.goto(FORM_URL)
    # If no model is found, raise and stop
    if model_name.empty:
        print("AUTOMATION STOPS NOW - INCORRECT/NON-EXISTING MODEL")
        raise Exception(f"No model found for Webform Model: {product_info.get('Model')}")
        page = context.pages[-1]  # continue on current page

    # 1. Select brand from dropdown
    if product_info.get("Brand"):
        page.locator('[formcontrolname="brand"]').click()
        page.locator(f'text="{product_info["Brand"]}"').click()

    # 2. Select model from mapped Chailease model
    if model_name:
        page.locator('[formcontrolname="model"]').click()
        page.locator(f'text="{model_name}"').click()

    # 3. Date of Manufacture (hardcoded or from DB)
    page.get_by_label("Date of Manufacture").fill("012025")

    # 4. Plate No
    if product_info.get("Number Plate"):
        page.get_by_label("Plate No.").fill(product_info["Number Plate"])

    # 5. Purchase Price
    if product_info.get("Price"):
        page.get_by_label("Purchase Price").fill(str(product_info["Price"]))

    # 6. Down Payment
    if product_info.get("Down Payment"):
        page.get_by_label("Down Payment").fill(str(product_info["Down Payment"]))

    print("[INFO] Product Info section filled successfully.")


# --- Page 6/7 Terms and Conditions ---
product_query = f"""
SELECT "Brand", "Down Payment", "ID", "Model", "NRIC", "Number Plate", "Price", "Product Type", "Tenure"
FROM "Product Info"
WHERE NRIC = {USER_ID}
"""
product_info_raw = db_chain.run(product_query)
product_info = parse_to_dict(product_info_raw)
print(product_info)

def fill_dealer_and_tenure_fields(context, product_info):
    page = context.pages[-1]

    # 1. Dealer Sales (p-dropdown[formcontrolname="dealerSalesId"])
    page.locator('[formcontrolname="dealerSalesId"]').click()
    page.locator('text="MOTOSING SDN BHD"').click()

    # 2. Marketing Officer - Select First Option
    page.locator('[formcontrolname="marketingOfficer"]').click()
    page.locator('.p-dropdown-item').nth(0).click()

    # 3. Tenure Month from Product Info
    if product_info.get("Tenure"):
        page.locator('[formcontrolname="tenureMonth"]').click()
        page.locator(f'text="{str(product_info["Tenure"])}"').click()

    print("[INFO] Dealer, Marketing Officer and Tenure fields filled.")
