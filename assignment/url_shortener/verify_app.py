import requests
import subprocess
import time
import sys
import os

def run_test():
    app_path = r"c:\Users\LENOVO\Desktop\innomatic internship\assignment\url_shortener\app.py"
    # Ensure DB is created
    if os.path.exists("urls.db"):
        os.remove("urls.db")
        
    process = subprocess.Popen([sys.executable, app_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Starting Flask app...")
    time.sleep(5)

    try:
        base_url = "http://127.0.0.1:5001"
        
        # Test 1: Home page loads
        resp = requests.get(base_url)
        assert resp.status_code == 200, "Home page failed to load"
        print("PASS: Home page loaded")

        # Test 2: Shorten a URL
        long_url = "https://www.google.com"
        resp = requests.post(base_url, data={"original_url": long_url})
        assert resp.status_code == 200, "Shorten request failed"
        if "Your Shortened URL" not in resp.text:
            print(f"DEBUG: Response text: {resp.text}")
        assert "Your Shortened URL" in resp.text, "Shortened URL not displayed"
        print("PASS: URL Shortened")
        
        # Extract short ID (hacky parsing for test)
        # Assuming format: value="http://127.0.0.1:5001/AbCdEf"
        import re
        match = re.search(r'value="(http://127.0.0.1:5001/([a-zA-Z0-9]+))"', resp.text)
        if match:
            short_url = match.group(1)
            short_id = match.group(2)
            print(f"Generated Short URL: {short_url}")
            
            # Test 3: Redirect
            resp = requests.get(short_url, allow_redirects=False)
            assert resp.status_code == 302, f"Redirect failed: {resp.status_code}"
            assert resp.headers['Location'] == long_url, "Redirect location incorrect"
            print("PASS: Redirect works")
            
            # Test 4: History
            resp = requests.get(f"{base_url}/history")
            assert resp.status_code == 200, "History page failed"
            assert long_url in resp.text, "Original URL not in history"
            assert short_id in resp.text, "Short ID not in history"
            print("PASS: History verified")
            
        else:
            print("FAIL: Could not extract short URL")

    except Exception as e:
        print(f"FAIL: {e}")
    finally:
        process.terminate()

if __name__ == "__main__":
    run_test()
