# File: tests/test_initial.py

import requests
import datetime
from helpers import BASE_URL, wait_for_startup, gather_endpoints

def test_endpoints():
    """Verify each endpoint returns HTTP 200."""
    print("\n--- Starting Initial Endpoint Wake Test ---")
    ENDPOINTS = gather_endpoints()

    payload = {
        'email' : 'test.sorea@gmail.com',
        'message' : f'Hello, this is a test message.{datetime.datetime.now()}'
    }

    # Use a simple endpoint as healthcheck
    healthcheck_url = f"{BASE_URL}{ENDPOINTS[0] if ENDPOINTS else '/api/health'}"
    assert wait_for_startup(healthcheck_url), "Azure Functions host didn't start."

    for ep in ENDPOINTS:
        url = BASE_URL + ep
        print(f"Testing: {url}")
        res = requests.get(url, json=payload)
        assert res.status_code == 200, f"Endpoint {ep} failed → {res.status_code}"
    
    print("--- Initial Endpoint Wake Test Passed ---")