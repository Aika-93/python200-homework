# ----Video link: https://youtu.be/I3Jgx6Qxrq0 -----

# Project -- Extract + Load Pipeline
# This script:
# 1. Extracts weather data from Open-Meteo API
# 2. Serializes it into JSON bytes
# 3. Uploads it to Azure Blob Storage
# 4. Verifies upload by listing blobs
# 5. Downloads blob, loads into pandas, and saves local copy 

import requests
import json
import pandas as pd
from datetime import date 
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

# Setup

ACCOUNT_URL = "https://aiperictd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

# Step 1: Extract

url = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude=35.2271&longitude=-80.8431"
    f"&hourly=temperature_2m,precipitation"
    f"&forecast_days=7"
)

response = requests.get(url)
response.raise_for_status()
data = response.json()

# Step 2: Serialize

payload = json.dumps(data).encode("utf-8")

# Step 3: Load

today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

credential = DefaultAzureCredential()
container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)
container.upload_blob(blob_path, payload, overwrite=True)
print(f"Uploaded {len(payload)} bytes to {blob_path}")

# Step 4: Verify

print("\n Blobs in container:")

for blob in container.list_blobs():
    print(f"{blob.name} ({blob.size} bytes)")

# Step 5: Read Back

raw = container.download_blob(blob_path).readall()
df = pd.DataFrame(json.loads(raw.decode("utf-8"))["hourly"])
print(f"\n First 5 rows")
print(df.head())

# Save raw JSON locally
with open("outputs/weather_raw.json", "wb") as f:
    f.write(raw)