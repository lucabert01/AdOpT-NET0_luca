import subprocess, os, sys

mains = ["main_cement.py", "main_WtE_CaL.py", "main_WtE_MEA.py", "main_WtE_MEA_timeless.py", "main_WtE_technology_selection.py"]
mains = ["main_cement_oxy_only.py", "main_cement.py"]#, "main_cement_flexible_ops.py"]
processes = []
env = os.environ.copy()


for script in mains:
    print(f"Starting {script}...")
    # run() waits for the process to complete before moving to the next iteration
    subprocess.run([sys.executable, script], check=True)
    print(f"Finished {script}\n")

print("All sequential simulations finished.")