from ldpc import ldpc_stack
from time import time

def main():
    print("Initialization test")
    ldpc_stack(esno_db=8.4, num_layers=2, num_prb=1000)

    start = time()
    print("Stress test started for 10 seconds")

    i = 0
    while time()-start < 10:
        ldpc_stack(esno_db=8.4, num_layers=2, num_prb=200)
        i += 1

    print(f"Test finished with {i} executions")
    

if __name__ == "__main__":
    main()