import subprocess, os, sys

mains = ["main_cement.py", "main_WtE_CaL.py", "main_WtE_MEA.py", "main_WtE_MEA_timeless.py", "main_WtE_technology_selection.py"]
mains = ["main_cement.py", "main_cement_flexible_ops.py"]
plots = ["plot_results_CaL.py","plot_results_MEA.py","plot_results_WtE_technology_selection.py","plot_results_cement.py",]
processes = []
env = os.environ.copy()


# Start all scripts
for script in mains:
    p = subprocess.Popen(
    [sys.executable, script],
    env=env
)
    processes.append(p)
    print(f"Launched {script}")

# Wait for all scripts to finish
for p in processes:
    p.wait()

print("All parallel simulations finished.")