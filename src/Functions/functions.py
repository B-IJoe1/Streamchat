# functions.py
import os
import time
import requests
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

def check_status(job_id: str) -> dict:
    """
    Poll the RunPod status endpoint for a given job.
    """
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def inference(prompt: str) -> dict:
    """
    Fire off an asynchronous job to the RunPod endpoint and
    poll until completion, then return the 'output' field.
    """
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    payload = {
        "input": {
            "prompt": prompt,
            "sampling_params": {      # adjust keys to match your worker’s signature
                "max_new_tokens": 512,
                "temperature": 0.2,
                "top_p": 0.1
            }
        }
    }

    # Start the job
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    result = r.json()
    job_id = result.get("id")
    if not job_id:
        raise RuntimeError(f"No job ID returned: {result}")

    # Poll until complete (or error)
    while True:
        status_json = check_status(job_id)
        status = status_json.get("status")
        if status == "COMPLETED":
            break
        if status in ("FAILED", "TIMED_OUT"):
            raise RuntimeError(f"RunPod job {job_id} failed with status '{status}'")
        time.sleep(5)

    # Return the output blob
    output = status_json.get("output")
    if output is None:
        raise RuntimeError(f"No output in status response: {status_json}")
    return output
