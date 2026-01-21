import subprocess

scripts = ["main_cement.py", "main_WtE_CaL.py", "main_WtE_MEA.py", "main_WtE_MEA_timeless.py", "main_WtE_technology_selection.py"]
processes = []

# Start all scripts
for script in scripts:
    p = subprocess.Popen(["python", script])
    processes.append(p)
    print(f"Launched {script}")

# Wait for all scripts to finish
for p in processes:
    p.wait()

print("All parallel simulations finished.")