import subprocess

mains = ["main_cement.py", "main_WtE_CaL.py", "main_WtE_MEA.py", "main_WtE_MEA_timeless.py", "main_WtE_technology_selection.py"]
plots = ["plot_results_CaL.py","plot_results_MEA.py","plot_results_WtE_technology_selection.py","plot_results_cement.py",]
processes = []

# Start all scripts
for script in mains:
    p = subprocess.Popen(["python", script])
    processes.append(p)
    print(f"Launched {script}")

# Wait for all scripts to finish
for p in processes:
    p.wait()

print("All parallel simulations finished.")