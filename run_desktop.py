import sys
import os
import time
import threading
import webbrowser
try:
    from streamlit.web import cli as stcli
except ImportError:
    from streamlit import cli as stcli
if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        os.path.join(os.path.dirname(__file__), "app.py"),
        "--global.developmentMode=false"
    ]
    sys.exit(stcli.main())

def run_server():
    # Resolve the absolute path of app.py whether running as script or frozen exe
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    app_path = os.path.join(base_dir, "app.py")
    
    # Configure Streamlit arguments for standalone desktop execution
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.port=8501",
        "--browser.serverAddress=localhost",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]
    stcli.main()

if __name__ == "__main__":
    # Start the Streamlit server in a background daemon thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Give the local server a second to initialize, then launch the interface
    time.sleep(2.5)
    webbrowser.open("http://localhost:8501")
    
    # Keep the main process alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sys.exit(0)
