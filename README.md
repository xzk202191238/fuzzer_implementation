# Student Fuzzer
Baseline\_fuzzer, student\_fuzzer and the experiments are included.  
This project is the optimization of **GreyBoxFuzzer** to more effectively detect cases with nested structures.

## Layout
**The directory structure of this project is as follows:**  
*examples* (it contains different types of test cases)  
*test* (used for making comparison between baseline_fuzzer & student_fuzzer)  
— baseline_fuzzer_test.py  
— student_fuzzer_test.py  
— bug.py (example cases for testing)  
— experiment.py
*bug.py*  
*Dockfile*  
*README.md*  
*requirements.txt*  
*student_fuzzer.py*  

## Instructions
To compare the performance of **baseline_fuzzer** and **student_fuzzer** repeatedly in the experiment, an `experiment.py ` script was created in the ‘test’ folder to iterate multiple times and record the executing time each time. The script uses the **exit(219)** as the standard for finding bugs. The execution time will be recorded regardless of whether a bug is found. The execution time for those that failed to find bugs will be marked with ‘\*’ for identification.  
To execute the experiment on the bug in `/test/bug.py`, just run:  

	python experiment.py

After running, you will get the results in two CSVs, `baseline_results.csv ` and `stu_results.csv` respectively.

## Setup
Install all dependencies needed by the Fuzzing Book baseline fuzzer with:

	pip install -r requirements.txt

You may want to do this in a Python **virtual environment** to avoid global dependency conflicts.

## Usage

The fuzzer expects a file named `bug.py` to be *in the same directory as the fuzzer file* (`student-fuzzer.py`).
This `bug.py` file should have two functions: an `entrypoint` that is fuzzed by the fuzzer and `get_initial_corpus` function which returns a list of initial inputs for the fuzzer.
To execute the fuzzer on the bug in `bug.py`, just run:

	python student_fuzzer.py

Several example bugs are included in the `examples` directory.
To run the fuzzer on an example bug, copy e.g. `examples/0/bug.py` to the base directory of this repository before running the fuzzer with the command above.  


**The student\_fuzzer optimization are implemented by XUE ZEKAI**  
**StudentID: A0276538N**  
**NUSNET ID: E1132271**  