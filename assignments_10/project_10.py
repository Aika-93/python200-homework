# --- Part 2: Project -- LLM Transform Pipeline ---

# Video link : https://youtu.be/p8CDiLSgf_k

# Reflect

"""
I think an LLM can classify weather conditions for outdoor running, but it is not the best use case for an LLM because 
the rules are fairly simple.
Deterministic code could probably do this task better.
Simple rules based on temperature and precipitation would be faster, cheaper, and give consistent results.
The advantage of an LLM is that it can handle more complex situations and make decisions that are closer to human judgment.

"""

import json
import pandas as pd
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from datetime import date
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Setup

ACCOUNT_URL = "https://aiperictd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

# Step 1: Read

today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

credential = DefaultAzureCredential()
container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

raw = container.download_blob(blob_path).readall()
data = json.loads(raw.decode("utf-8"))

# Reshape 
hourly = data["hourly"]
records = []
for i in range(len(hourly["time"])):
    record = {
        "time": hourly["time"][i],
        "temperature_2m": hourly["temperature_2m"][i],
        "precipitation": hourly["precipitation"][i],
    }
    records.append(record)
print(f"Loaded {len(records)} hourly records")


# Step 2: Transform

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

def make_user_message(record):
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

VALID_LABELS = {"good", "marginal", "bad"}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enriched = []
for i, record in enumerate(records[:24]):
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(record)},
        ]
    )
    raw_label = response.choices[0].message.content.strip().lower()
    label = raw_label if raw_label in VALID_LABELS else "unknown"
    enriched.append({**record, "conditions": label})
    if (i + 1) % 6 == 0:
        print(f"Processed {i+1} records ...")


# Step 3: Write

processed_path = f"processed/{today}/weather_classified.json"
container.upload_blob(processed_path, json.dumps(enriched).encode("utf-8"), overwrite=True)
print(f"Uploaded to {processed_path}")


# Step 4: Spot-Check

raw_processed = container.download_blob(processed_path).readall()
processed_data = json.loads(raw_processed.decode("utf-8"))
df = pd.DataFrame(processed_data)

print("\nLabel distribution:")
print(df["conditions"].value_counts())

print(df.head())


# Step 5: Save Output

with open("outputs/first_10_records.json", "w", encoding="utf-8") as f:
    json.dump(enriched[:10], f, indent=2)