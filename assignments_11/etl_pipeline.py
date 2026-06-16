# Video demo:  https://youtu.be/EOL1lzhkIvw

# ----- Project -- Full ETL Pipeline -----

import requests
from prefect import flow, task
from openai import OpenAI
import os
import json 
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from datetime import date
from dotenv import load_dotenv


load_dotenv()

# Limit number of records processed in transform step
MAX_RECORDS = 24

# ----- EXTRACT-----
# Fetch weather data from Open-Meteo API with retry logic 

@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude:float) -> dict:

    # API endpoint with query parameters
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,precipitation"
        f"&forecast_days=7"
    )
    # Send request to API
    response = requests.get(url)
    response.raise_for_status()

    print(f"Extracted weather data for {latitude}, {longitude}")

    # Return raw JSON response
    return response.json()


# ----- TRANSFORM -----
# Reshape hourly data and classify weather using OpenAI

@task
def transform(data: dict, max_records: int) -> list:
    
    # Valid classification labels
    VALID_LABELS = {"good", "marginal", "bad"}
    
    # System prompt for OpenAI classification task  
    SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)
    # Initialize OpenAI client
    client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

    hourly = data["hourly"]

    # Convert parallel lists into structured records
    records = []

    for i in range(min(max_records, len(hourly["time"]))):
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i]
        })

    enriched = []
    
    # Classify first 24 records using OpenAI
    for i, record in enumerate(records[:24]):

        # Build prompt for model
        user_msg = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )
    
        # Call OpenAI API
        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ]
        )

        # Normalize model output
        raw_label = response.choices[0].message.content.strip().lower()
        label = raw_label if raw_label in VALID_LABELS else "unknown"

        # Add classification to record
        enriched.append({**record, "conditions": label})

        # Progress logging every 6 records
        if (i+1) % 6 == 0:
            print(f"Classified {i+1}/{len(records[:24])} records")
    
    print(f"Transform complete: {len(enriched)} records enriched")

    return enriched


# ----- LOAD -----
# Upload processed data to Azure Blob Storage as JSON
@task
def load(records: list, blob_path: str) -> None:

    # Azure Storage account details
    ACCOUNT_URL = "https://aiperictd2026sa.blob.core.windows.net"
    CONTAINER = "pipeline-data"

    # Authenticate using Azure Default Credentials
    credential = DefaultAzureCredential()

    # Create blob container client 
    container = ContainerClient(
        ACCOUNT_URL,
        CONTAINER,
        credential = credential
    )

    # Convert Python objects to JSON bytes
    payload = json.dumps(records).encode("utf-8")

    # Upload to Azure Blob Storage (overwrite enabled)
    container.upload_blob(
        blob_path,
        payload,
        overwrite=True
    )

    print(f"Loaded {len(payload)} bytes to {blob_path}")


# ----- FLOW -----
# Orchestrates Extract -> Transform -> Load pipeline
@flow(log_prints=True)
def etl_pipeline(
    latitude: float = 35.2271,
    longitude: float = -80.8431
):
    # Create date-based folder for partitioning
    today = date.today().isoformat()

    # Final blob storage path 
    blob_path = f"final/{today}/weather_etl.json"

    # Step 1: Extract data 
    data = extract(latitude, longitude)

    # Step 2: Transform + classify
    enriched = transform(
        data,
        max_records=MAX_RECORDS
    )

    # Step 3: Load into Azure Blob Storage
    load(enriched, blob_path)
    print(f"Pipeline complete. Results at {blob_path}")


if __name__ == "__main__":
    etl_pipeline()