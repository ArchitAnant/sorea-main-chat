import requests
import time
import os
import re

BASE_URL = "http://localhost:7071"


def wait_for_startup(url):
    """Wait for Azure Functions host to start responding."""
    for _ in range(30):
        try:
            requests.get(url, timeout=2)
            return True
        except:
            time.sleep(1)
    return False


def gather_endpoints():
    """Extract @app.route(route='...') routes from function_app.py."""
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

                # Skip parameterized routes like hello/{name}
                if "{" in route or "}" in route:
                    continue

                endpoints.append(f"/api/{route}")

    return endpoints


def test_endpoints():
    """Verify each endpoint returns HTTP 200."""
    ENDPOINTS = gather_endpoints()

    # Use a simple endpoint as healthcheck if available
    healthcheck = f"{BASE_URL}{ENDPOINTS[0] if ENDPOINTS else '/api/health'}"

    assert wait_for_startup(healthcheck), "Azure Functions host didn't start."

    for ep in ENDPOINTS:
        url = BASE_URL + ep
        print(f"Testing: {url}")

        res = requests.get(url, timeout=5)

        assert res.status_code == 200, f"Endpoint {ep} failed → {res.status_code}"
