import subprocess
import sys
import time
import os
import signal
from pathlib import Path

# Force UTF-8 stdout if needed
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def start_tyretwin():
    print("=" * 60)
    print("   [F1] TYRETWIN: F1 TYRE DEGRADATION INTELLIGENCE PLATFORM")
    print("=" * 60)
    
    root_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(root_dir))
    
    # 1. Start FastAPI Backend
    print("[1/2] Launching FastAPI Backend on http://localhost:8000 ...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", "0.0.0.0", "--port", "8000"
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(root_dir))
    time.sleep(2)
    
    # 2. Start Streamlit Frontend
    print("[2/2] Launching Streamlit Pit Wall Dashboard on http://localhost:8501 ...")
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(root_dir / "app" / "dashboard.py"),
        "--server.port", "8501",
        "--server.headless", "true"
    ]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(root_dir))
    
    print("-" * 60)
    print(">> TyreTwin is live and running!")
    print("   * Frontend Dashboard: http://localhost:8501")
    print("   * Backend API Docs:   http://localhost:8000/docs")
    print("Press Ctrl+C to stop services.")
    print("-" * 60)
    
    try:
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down TyreTwin services...")
        backend_proc.terminate()
        frontend_proc.terminate()

if __name__ == "__main__":
    start_tyretwin()
