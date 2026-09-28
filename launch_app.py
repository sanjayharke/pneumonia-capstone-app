
"""Start both services; stop both if either fails. Run: python launch_app.py"""
from pathlib import Path
import subprocess,sys,time,signal

def main():
    root=Path(__file__).resolve().parent
    processes=[]
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        processes.append(subprocess.Popen([sys.executable,'-m','waitress','--listen=127.0.0.1:5000','backend:app'],cwd=root))
        processes.append(subprocess.Popen([sys.executable,'-m','streamlit','run','app.py','--server.address=0.0.0.0','--server.port=8501','--server.fileWatcherType=none','--browser.serverAddress=localhost'],cwd=root))
        while all(p.poll() is None for p in processes):time.sleep(.5)
        return next((p.returncode or 1 for p in processes if p.poll() is not None),1)
    except KeyboardInterrupt:
        return 0
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
if __name__=='__main__':sys.exit(main())
