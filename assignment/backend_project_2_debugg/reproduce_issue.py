import requests
import subprocess
import time
import sys
import os

def run_test():
    # Start the Flask app
    app_path = r"c:\Users\LENOVO\Desktop\innomatic internship\assignment\backend_project_2_debugg\note_taking_app\app.py"
    process = subprocess.Popen([sys.executable, app_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print("Starting Flask app...")
    time.sleep(5)  # Wait for app to start

    try:
        # Test 1: GET request to root (should fail with 405 currently)
        try:
            print("Sending GET request to /...")
            response = requests.get("http://127.0.0.1:5000/")
            print(f"GET / Response Code: {response.status_code}")
            if response.status_code == 405:
                print("Confirmed: GET request failed with 405 Method Not Allowed.")
            else:
                print(f"Unexpected status code for GET: {response.status_code}")
        except Exception as e:
            print(f"GET request failed: {e}")

        # Test 2: POST request (simulating the current form behavior which is GET, but testing POST per code)
        # The current code expects args (query string) even in POST? No, usually POST uses form data.
        # But let's see what happens if we POST with data.
        try:
            print("Sending POST request to / with data...")
            response = requests.post("http://127.0.0.1:5000/", data={"note": "Test Note"})
            print(f"POST / Response Code: {response.status_code}")
            if "Test Note" in response.text:
                 print("POST worked, but did it add the note?")
            else:
                 print("Note 'Test Note' NOT found in response.")
                 
            # The current code uses request.args.get("note"). 
            # Requests.post with data puts it in body. args gets query params.
            # So this should fail to add the note.
            
            print("Sending POST request to / with query params (simulating weird usage)...")
            response = requests.post("http://127.0.0.1:5000/?note=QueryNote")
            if "QueryNote" in response.text:
                print("Confirmed: App uses query params (request.args) even for POST.")
            else:
                print("QueryNote not found.")

        except Exception as e:
            print(f"POST request failed: {e}")

    finally:
        process.terminate()
        print("Stopped Flask app.")

if __name__ == "__main__":
    run_test()
