import inspect

from fuzzingbook import GreyboxFuzzer as gbf
from fuzzingbook import Coverage as cv
from fuzzingbook import MutationFuzzer as mf

from collections import deque
from difflib import SequenceMatcher
from typing import List, Set, Any, Tuple, Optional, Callable, Dict, Union
from types import FrameType

import traceback
import numpy as np
import time
import sys
import string
import random
import re
import pickle
import hashlib

from bug import entrypoint
from bug import get_initial_corpus






class MyCoverage(cv.Coverage):
    def __init__(self, n_gram=2):
        super().__init__()
        self.n_gram = n_gram
        self.ngram_queue = deque(maxlen=n_gram)
        self.ngrams = set()
        self.depth = 0

    def traceit(self, frame: FrameType, event: str, arg: Any) -> Optional[Callable]:

        # Trace execution depth
        if event == 'call':
            self.depth += 1
        elif event == 'return':
            self.depth -= 1

        # For function calls or returns, we add to the n-gram queue
        if event == "line":
            function_name = frame.f_code.co_name
            lineno = frame.f_lineno
            location = (function_name, lineno)

            # Update the n-gram queue with the current location
            self.ngram_queue.append(location)
            # Once we have a full n-gram, add it to the set of n-grams
            if len(self.ngram_queue) == self.n_gram:
                self.ngrams.add((tuple(self.ngram_queue), self.depth))
                # 移除最早的元素以保持队列大小
                self.ngram_queue.popleft()


        # Returning the base class response ensures that we continue tracing as before
        return self.traceit





    def coverage(self) -> Set[Tuple[str, int]]:
        # Return the line coverage
        # line_coverage = set(self.trace())
        return self.ngrams

    def ngram_coverage(self) -> Set[Tuple[Tuple[str, int]]]:
        # Return the n-gram coverage
        return self.ngrams

class MyFunctionCoverageRunner(mf.FunctionRunner):
    def __init__(self, function):
        super().__init__(function)
        self._coverage = set()
        self._ngram_coverage = set()

    def run_function(self, inp: str) -> Any:
        with MyCoverage() as cov:
            try:
                result = super().run_function(inp)
            except Exception as exc:
                print(f"Exception caught: {exc}") # print error
                # Save both line coverage and n-gram coverage
                # self._coverage = cov.coverage()
                self._ngram_coverage = cov.ngram_coverage()
                raise exc

            # Save both line coverage and n-gram coverage
            # self._coverage = cov.coverage()
            self._ngram_coverage = cov.ngram_coverage()

        return result

    def coverage(self) -> Set[Tuple[str, int]]:
        return self._ngram_coverage

    def ngram_coverage(self) -> Set[Tuple[Tuple[str, int]]]:
        # Return the n-gram coverage
        return self._ngram_coverage






class CountingGreyboxFuzzer(gbf.GreyboxFuzzer):

    def reset(self):
        super().reset()
        self.schedule.path_frequency = {}

    def run(self, runner: MyFunctionCoverageRunner) -> Tuple[Any, str]:
        """Inform scheduler about n-gram path frequency"""
        result, outcome = super().run(runner)

        # 改用n-gram覆盖率
        ngram_path_id = getNgramPathID(runner.ngram_coverage())
        if ngram_path_id not in self.schedule.path_frequency:
            self.schedule.path_frequency[ngram_path_id] = 1
        else:
            self.schedule.path_frequency[ngram_path_id] += 1

        return result, outcome




class MyMutator(gbf.Mutator):
    def __init__(self):
        super().__init__()
        self.ngram_data = []
        self.depth_threshold = 2

    def update_ngram(self, ngram_coverage):
        self.ngram_data = ngram_coverage


    # Erase a byte from the data
    def mutate_erase_bytes(self, s: str) -> str:
        idx = random.randrange(len(s))
        return s[idx:random.randrange(idx, len(s))]

    # Insert one or more random bytes into the data
    def mutate_insert_bytes(self, s: str) -> str:
        idx = random.randrange(len(s))
        # Insert a larger block of random bytes to cause overflow
        new_bytes = self.get_random_bytes(random.randrange(10, 100))
        return s[:idx] + new_bytes + s[idx:]

    # Insert repeated bytes into the data
    def mutate_insert_repeated_bytes(self, s: str) -> str:
        idx = random.randrange(len(s))
        new_byte = self.get_random_byte()
        # Repeat N times and inserts it into the data
        N = random.randrange(16)
        s[idx:idx + N] = bytearray(new_byte) * N
        return s

    @staticmethod
    def get_random_bytes(size):
        return bytearray(random.getrandbits(8) for i in range(size))

    @staticmethod
    def get_random_byte():
        return random.getrandbits(8)


    # Input reduction
    # For dealing with example 2
    def mutate_input_reduce(self, s: str) -> str:
        # Regular expression matching one or more consecutive repeated non-empty character sequences
        pattern = re.compile(r'(.)\1{1,}')

        # Replace matched substrings with "substring{count}"
        def replace_func(match):
            return '{}{{{}}}'.format(match.group(1), len(match.group(0)))

        compressed_string = pattern.sub(replace_func, s)
        return compressed_string




    def mutate_depth_exe(self, s: str) -> str:
        tracer = ExecutionDepthTracer()
        # 跟踪执行深度和运行时间
        start_time = time.time()

        def run_trace(s):
            # start tracing
            sys.settrace(tracer.trace)
            try:
                entrypoint(s)
            finally:
                # end tracing
                sys.settrace(None)

        run_trace(s)
        execution_depth = tracer.get_max_depth()

        duration = time.time() - start_time

        self.max_depth = max(self.max_depth, execution_depth)
        self.max_duration = max(self.max_duration, duration)

        # 根据执行深度和运行时间调整变异策略
        if execution_depth > self.max_depth * 0.8 or duration > self.max_duration * 0.8:
            # 如果执行深度或运行时间接近最大值，尝试特定的变异策略
            index = random.randrange(len(s))
            char_to_insert = random.choice(string.ascii_letters + string.digits + string.punctuation)
            s = s[:index] + char_to_insert + s[index:]

        return s

    def generate_input(self) -> str:

        chosen_ngram, chosen_depth = random.choice(list(self.ngrams))

        data = bytearray(random.choice(self.seed))

        num_mutations = random.randrange(1, 6)
        for i in range(num_mutations):
            if not data:
                return ""

            # 根据执行深度选择变异策略
            if chosen_depth > self.depth_threshold:
                # 对于较深的执行深度，倾向于使用更复杂的变异
                choice = random.choice([3, 4])
            else:
                # 对于较浅的执行深度，使用较简单的变异
                choice = random.randrange(3)

            # 应用选定的变异方法
            data = self.apply_mutation(choice, data)

        return data.decode('utf-8', errors='ignore')

    def apply_mutation(self, choice, data):
        if choice == 0:
            return self.mutate_erase_bytes(data)
        elif choice == 1:
            return self.mutate_insert_bytes(data)
        elif choice == 2:
            return self.mutate_insert_repeated_bytes(data)
        elif choice == 3:
            return self.mutate_depth_exe(data)
        elif choice == 4:
            return self.mutate_input_reduce(data)
        else:
            assert False


# 计算执行深度--跟踪函数调用的嵌套级别
class ExecutionDepthTracer:
    def __init__(self):
        self.depth = 0
        self.max_depth = 0

    def trace(self, frame, event, arg):
        if event == 'call':
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
        elif event == 'return':
            self.depth -= 1

    def get_max_depth(self):
        return self.max_depth




def getNgramPathID(ngram_coverage: Set[Tuple[Tuple[str, int]]]) -> str:
    """Returns a unique hash for the covered n-grams"""
    # 将n-gram覆盖率转换成排序后的列表形式以保证一致性
    sorted_ngrams = sorted(ngram_coverage)
    # 序列化排序后的n-grams
    pickled = pickle.dumps(sorted_ngrams)
    # 返回序列化数据的MD5哈希值
    return hashlib.md5(pickled).hexdigest()




## You can re-implement the fuzzer class to change your
## fuzzer's overall structure

# class MyFuzzer(gbf.GreyboxFuzzer):
#
#     def reset(self):
#           <your implementation here>
#
#     def run(self, runner: gbf.FunctionCoverageRunner):
#           <your implementation here>
#   etc...


## The Mutator and Schedule classes can also be extended or
## replaced by you to create your own fuzzer!



if __name__ == "__main__":
    seed_inputs = get_initial_corpus()
    fast_schedule = gbf.AFLFastSchedule(5)
    line_runner = MyFunctionCoverageRunner(entrypoint)


    fast_fuzzer = gbf.CountingGreyboxFuzzer(seed_inputs, MyMutator(), fast_schedule)

    # for check
    print("Starting fuzzing loop...")
    fast_fuzzer.runs(line_runner, trials=9999999)
    print("Max trials end.")