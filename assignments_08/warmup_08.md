----------- Cloud Concepts ---------------- 

Cloud Concepts Q1:

What is the core economic model of cloud computing, and how does it differ from owning your own servers?

The core economic model of cloud computing is pay-as-you-go, meaning you only pay for the resources you actually use.
This differs from owning your own servers, where you must invest upfront in hardware and maintenance costs even if the resources are not fully used.


Cloud Concepts Q2:

What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

- Vertical scaling means increasing the resources of a single machine (for example, upgrading RAM or CPU).
- Horizontal scaling means adding more machines to handle the load.

I would choose vertical scaling for small applications that need simple upgrades, and horizontal scaling for large systems that need high availability and can distribute traffic accross multiple servers.

Scenarios:

    1. A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch. 

This is horizontal scaling because the application can handle more traffic by adding more servers.

    2. A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.

This is vertical scaling because the machine needs more computing power, RAM, and a faster GPU.

    3. A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be 
        split across machines.

This is horizontal scaling because the file processing can be distributed across multiple machines.


Cloud Concepts Q3:

Gmail - SaaS
Gmail is SaaS because it is ready to use application that users access through the internet without managing infrastructure.

Azure Virtual Machines - IaaS
Azure Virtual Machines are IaaS because they provide infrastructure while the users manages the operating system and software.

Azure App Service - PaaS
Azure App Service is PaaS because developers can deploy application without managing the underlying servers.

AWS S3 (Simple Storage Service) - IaaS
AWS S3 (Simple Storage Service) is IaaS because it provides cloud storage infrastructure that users manage for their own data.

GitHub Codespaces - PaaS
GitHub Codespaces is PaaS because it provides a cloud development environment without requiring developers to manage infrastructure.

Snowflake - PaaS
Snowflake is PaaS because it provides a managed platform for storing and analyzing data without managing servers.

- IaaS ( Infrastructure as a Service) provides infrastructure like virtual machines, storage, and networking. 
    As a developer, I am responsible for managing the operating system, software, and my application. 
        Example: Azure Virtual Machine.

- PaaS (Platform as a Service) provides a platform to build and deploy applications without managing the 
    underlying infrastructure. 
    As a developer, I manage my code and application settings, while the provider manages the servers and operating system.
        Example: Azure App Service.

- SaaS (Software as a Service) is a ready to use application delivered over the internet. 
    As a user, I do not manage infrastructure or software updates and simply use the application.
        Example: Gmail.


Cloud Concepts Q4:

What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?

A manged data platform like Snowflake or Databricks provides ready-made tools for data processing and analytics, while a cloud provider like Azure gives direct access to infrastructure and cloud services. 
Using a managed platform makes setup and scaling easier and requires less infrastructure managment, but it offers less control and flexibility compared to working directly with Azure.


Cloud Concepts Q5:

The lesson names two situations where the cloud is probably not the right choice. What are they?

The cloud may not be right choice when a dataset fits on a single machine and does not require large computing power, since local processing can be faster and cheaper.
It may also not be the best choice when cloud costs become too expensive or resources are not managed carefully.


-------------- Azure Basics ---------------------

Azure Basics Q1:

What is the difference between an Azure subscription and a resource group? Which one is yours alone, and which one does CTD share?

An Azure subscription -  is the top-level account that manages billing, access, and cloud resources.
A resource group - is a container inside a subscription used to organize related resources.

The subscription is shared by CTD, while my resource group is assigned to me for my own work.


Azure Basics Q2:

Azure Cloud Shell is ephemeral by default. What does that mean in practice, and what does your course setup use to make it persistent?

Ephemeral means the Cloud Shell is temporary and does not automatically keep files or changes between sessions. 
In our course setup, Azure Storage is used to make environment persistant so files and work can be saved.


Azure Basics Q3:

What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?

The SSH private key is secret and stays on my computer, while the SSH public key can be shared.
The public key is uploaded to the remote system because it allows the server to verify mmy identity without exposing the private key. This is safe because the public key alone cannot be used to access the system.


Azure Basics Q4:

az account show

{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "mail#amanturova93@gmail.com",
    "type": "user"
  }
}

az account show --output table  

EnvironmentName    HomeTenantId                          IsDefault    Name                       State    TenantId
-----------------  ------------------------------------  -----------  -------------------------  -------  ------------------------------------
AzureCloud         0f040ddd-301f-4665-8677-7b21f129d605  True         CTD Nonprofit Sponsorship  Enabled  0f040ddd-301f-4665-8677-7b21f129d605


Adding --output table changes the output from detailed JSON format to a simpler table format that is easier to read but contains less information.