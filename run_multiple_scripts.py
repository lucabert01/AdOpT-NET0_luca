import subprocess, os, sys

mains = ["main_cement.py", "main_WtE_CaL.py", "main_WtE_MEA.py", "main_WtE_MEA_timeless.py", "main_WtE_technology_selection.py"]
mains = ["main_WtE_CaL.py", "main_WtE_MEA.py", "main_WtE_MEA_timeless.py", "main_WtE_technology_selection.py",
         "main_WtE_withoutCCS.py"]
plots = ["plot_results_CaL.py","plot_results_MEA.py","plot_results_WtE_technology_selection.py","plot_results_cement.py",]
processes = []
env = os.environ.copy()


for script in mains:
    print(f"Starting {script}...")
    # run() waits for the process to complete before moving to the next iteration
    subprocess.run([sys.executable, script], check=True)
    print(f"Finished {script}\n")

print("All sequential simulations finished.")