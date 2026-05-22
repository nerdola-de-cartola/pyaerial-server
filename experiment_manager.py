import subprocess

N = 5  # número de processos

processes = []

for _ in range(N):
    p = subprocess.Popen(["python", "base_experiment.py"])
    processes.append(p)

for p in processes:
    p.wait()