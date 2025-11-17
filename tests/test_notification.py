# File: tests/test_notification.py

import requests
from helpers import BASE_URL

def test_notification():
    """Test for notification endpoint."""
    print("\n--- Starting Notification API Test ---")
    url = f"{BASE_URL}/api/notification"
    payload = {
        'email' : 'test.sorea@gmail.com'
    }
    print(f"Testing: {url}")
    res = requests.post(url, json=payload)
    return_json = res.json()
    assert return_json['notification'] == "[TEST NOTIFICATION SUCCESS]", f"Notification endpoint failed → {res.status_code}"
    print("--- Notification API Test Passed ---")