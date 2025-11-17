# File: tests/helpers.py

import requests
import time
import os
import re
import datetime

BASE_URL = "http://localhost:7071"

def wait_for_startup(url):
    """Wait for Azure Functions host to start responding."""
    print(f"Waiting for host to start at {url}...")
    for _ in range(30):
        try:
            response = requests.get(url, timeout=2)
            if response.ok:
                print("Host is up!")
                return True
        except requests.ConnectionError:
            time.sleep(1)
    return False

def gather_endpoints():
    """Extract @app.route(route='...') routes from function_app.py."""
    print("Gathering endpoints from function_app.py...")
    target_file = os.path.join("function", "function_app.py")
    if not os.path.exists(target_file):
        raise FileNotFoundError("function/function_app.py not found")

    endpoints = []
    pattern = re.compile(r'@app\.route\(.*route="([^"]+)"')
    with open(target_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                route = match.group(1)
                if "{" in route or "}" in route:
                    continue
                endpoints.append(f"/api/{route}")
    print(f"Found endpoints: {endpoints}")
    return endpoints