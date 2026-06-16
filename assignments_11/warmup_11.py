# ----- Prefect Orchestration -----

# Q1

"""
@flow is used to organize and orchestrate the pipeline workflow.
It controls the execution order of tasks and the overall pipeline logic.

@task is used for individual units of work inside the flow, especially for steps with I/O, API calls, retries, logging,
or state tracking.

I would not decorate a simple Celsius to Fahrenheit helper function with @task because it is a small in memory calculation
with no I/O or external dependencies.

"""

# Q2

@task(retries=3, retry_delay_seconds=30)
def call_api():
    # Code
    pass


# Q3

"""
I would open the failed flow run in the Prefect UI and look at the task states.
Then I would open the Logs tab to inspect the error details and any print or
log output captured during the run.
I expect to find information such as the exception message, stack trace, retry attempts, timestamps, and
logs showing where the pipeline failed.
Since the transform task failed, the load task never ran because the flow after the failure.

"""

# ----- Production Patterns -----

# Q1

"""
raise_for_status() raises an exception for any HTTP error response, automatically failing the task when the API request
is not successful.
This is better than manually checking status_code because it ensures all error cases are caught and the pipeline fails 
immediately instead of continuing with invalid data.
If the API returns a 500 error, raise_for_status() will raise an exception, the current task will fail, and 
downstream tasks will not execute.
If we only print an error when status_code != 200, the task may continue running, and downstream tasks could process missing
or incorrect data, leading to unreliable pipeline outputs.

"""

# Q2

"""
overwrite=True ensures that the output file in Blob Storage is replaced every time the pipeline runs.
In this scenario, after a failed run and a successful rerun, overwrite=True guarantees that the final output reflects knly the 
latest successful execution, not partial or outdated data from previous attempts.
Without overwrite=True, the pipeline could either fail when trying to write an existing file or result in keeping outdated or 
incomplete data, which would make the stored data output unreliable.

"""

# Q3

@task
def load(records: list, blob_path: str) -> None:
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")