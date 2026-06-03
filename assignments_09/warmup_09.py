
# --- Azure Authentication ---

# Q1

"""
When running locally, DefaultAzureCredential relies on an existing Azure login session.
You must run 'az login' first.
DefaultAzureCredential checks several authentication methods and automatically detects the Azure CLI credential if an active
Azure CLI session exists. 

"""

# Q2

"""
A deployed pipeline cannot use 'az login' because Azure VMs and containers run without an interactive user to sign in manually.
Instead, it uses Managed Identity, which provides secure automatic authentication.
DefaultAzureCredential automatically detects the available authentication method,
so the same Python code works locally with 'az login' and in Azure with Managed Identity.

"""

# Q3

"""
Two common causes of AuthenticationError are:
1. No valid authentication session exists, such as forgetting to run 'az login' or having an expired Azure CLI session.
    I encountered this when DefaultAzureCredential could not find an active Azure CLI login.
    I would diagnose this by running 'az account show' or 'az login' again.
2. A credential source is configured incorrectly, such as environment variables or Managed Identity settings being missing or invalid.
    I would diagnose this by reading the DefaultAzureCredential error details to see which credential failed and why.

"""


# ----- Blob Storage ------

# Q1

"""
Azure Blob Storage has three levels:
1. Storage Account - the main storage (like a computer or cabinet).
2. Container -  like a folder inside it.
3. Blob - the actual file (data) inside the folder.

"""

# Q2

"""
1. Blob Storage is used because we only need to store raw JSON API responses for later reprocessing.
2. A relational database like Azure SQL is needed because we need to run queries, extract customer IDs, and perform daily analytics 
    on large structured data.
3. Blob Storage is used because we only need to store image embeddings (NumPy arrays) between pipeline runs.

"""

# Q3

def list_container(container_client):
    for blob in container_client.list_blobs():
        print(f"{blob.name} - {blob.size} bytes")


# Q4

def upload_text(container_client, blob_name, text):
    container_client.upload_blob(
        name = blob_name, 
        data = text.encode("utf-8"),
        overwrite = True
    )