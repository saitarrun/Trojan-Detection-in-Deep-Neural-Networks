import os
import sys
import time
import subprocess

try:
    import redislite
except ImportError:
    print("Error: redislite not found. Please run: pip install redislite")
    sys.exit(1)

def start_redis():
    print("--- 🛡️ Starting Portable Redis Server ---")
    db_file = os.path.abspath("redis.db")
    
    # We use redislite to start a server on the standard port 6379
    # This ensures Celery and FastAPI can connect without changing code.
    try:
        # Configuration for the server
        config = {
            'port': '6379',
            'bind': '127.0.0.1',
            'daemonize': 'yes',
            'loglevel': 'notice'
        }
        
        # Instantiate Redis with config results in starting the server
        r = redislite.Redis(db_file, serverconfig=config)
        
        print(f"✅ Redis started successfully!")
        print(f"📍 Database file: {db_file}")
        print(f"🔌 Port: 6379 (Localhost)")
        print(f"🚀 Status: {r.ping()}")
        
    except Exception as e:
        print(f"❌ Failed to start Redis: {e}")
        print("\nAttempting direct binary execution...")
        
        # Fallback: Find the binary inside the package
        pkg_path = os.path.dirname(redislite.__file__)
        redis_bin = os.path.join(pkg_path, 'bin', 'redis-server')
        
        if os.path.exists(redis_bin):
            print(f"Found binary at {redis_bin}. Starting manually...")
            subprocess.Popen([redis_bin, "--port", "6379", "--bind", "127.0.0.1", "--daemonize", "yes"])
            time.sleep(2)
            print("✅ Manual start attempted.")
        else:
            print(f"CRITICAL: Could not find redis-server binary at {redis_bin}")

if __name__ == "__main__":
    start_redis()
