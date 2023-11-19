import time
import numpy as np
import pandas as pd
import os
import subprocess
import string
import random


def run_fuzzer(fuzzer_sc, seeds):
    start_time = time.time()
    process = subprocess.run(['python', fuzzer_sc, seeds], capture_output= True)
    end_time = time.time()
    execution_time = end_time - start_time
    exit_code = process.returncode
    return execution_time, exit_code

def fuzzer_script(fuzzer_sc, trails, seed_inputs):
    execution_time = []
    for i in range(trails):
        seeds = seed_inputs[i]
        exec_time, exit_code = run_fuzzer(fuzzer_sc, seeds)
        format_time = f"{exec_time:.4f}"

        if exit_code == 219:
            execution_time.append(format_time)
        else:
            execution_time.append(format_time + "*")

    return execution_time


def generate_inputs(trails):
    random.seed(20)
    return [''.join(random.choices(string.ascii_lowercase, k=20))for i in range(trails)]


def save_to_csv(fuzzer_name, execution_time):
    df = pd.DataFrame({'ExecutionTime': execution_time})
    csv_filename = f'{fuzzer_name}_results.csv'
    df.to_csv(csv_filename, index = False)
    return csv_filename

def main(baseline_script, stu_script, trails):

    seed_inputs = generate_inputs(trails)

    # baseline_fuzzer testing
    baseline_times = fuzzer_script(baseline_script, trails, seed_inputs)
    baseline_csv = save_to_csv('baseline', baseline_times)
    print(f"baseline_fuzzer has been saved to {baseline_csv}")

    # student_fuzzer testing
    stu_times = fuzzer_script(stu_script, trails, seed_inputs)
    stu_csv = save_to_csv('stu', stu_times)
    print(f"student_fuzzer has been saved to {stu_csv}")




if __name__ == '__main__':
    trails = 50
    main('baseline_fuzzer_test.py', 'student_fuzzer_test.py', trails)



