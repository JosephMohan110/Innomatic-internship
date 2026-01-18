import requests
import subprocess
import time
import sys

def run_test():
    app_path = r"c:\Users\LENOVO\Desktop\innomatic internship\assignment\backend_project_2_debugg\note_taking_app\app.py"
    process = subprocess.Popen([sys.executable, app_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Starting Flask app...")
    time.sleep(5)

    try:
        # Test 1: GET / (Should be 200)
        resp_get = requests.get("http://127.0.0.1:5000/")
        assert resp_get.status_code == 200, f"GET failed: {resp_get.status_code}"
        print("PASS: GET / returned 200 OK")

        # Test 2: POST / with data (Should add note)
        resp_post = requests.post("http://127.0.0.1:5000/", data={"note": "MySecretNote"})
        assert resp_post.status_code == 200, f"POST failed: {resp_post.status_code}"
        assert "MySecretNote" in resp_post.text, "POST did not add note!"
        print("PASS: POST / added note successfully")
        
    except Exception as e:
        print(f"FAIL: {e}")
    finally:
        process.terminate()

if __name__ == "__main__":
    run_test()
