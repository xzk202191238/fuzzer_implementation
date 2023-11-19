import inspect

from fuzzingbook import GreyboxFuzzer as gbf
from fuzzingbook import Coverage as cv
from fuzzingbook import MutationFuzzer as mf

from collections import deque
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
        self.max_depth = 0

    def traceit(self, frame: FrameType, event: str, arg: Any) -> Optional[Callable]:

        # Trace execution depth
        if event == 'call':
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
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
                # remove oldest elements to maintain queue size
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

    def get_ngram_data(self):
        return [(ngram, depth) for ngram, depth in self.ngrams]

    def get_max_depth(self):
        return self.max_depth




class MyFunctionCoverageRunner(mf.FunctionRunner):
    def __init__(self, function):
        super().__init__(function)
        self._coverage = set()
        self._ngram_coverage = set()
        self._max_depth = 0

    def run_function(self, inp: str) -> Any:
        with MyCoverage() as cov:
            try:
                result = super().run_function(inp)
                self._max_depth = cov.max_depth
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

    def get_max_depth(self):
        return self._max_depth

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
        ngram_path_id = getNgramPathID(runner.ngram_coverage())

        if isinstance(self.mutator, MyMutator):
            self.mutator.chosen_depth = runner.get_max_depth()

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
        self.max_depth = 0
        self.max_duration = 0
        self.chosen_depth = 0

        self.phase = 0
        self.phase_limit = 800
        self.phase_count = 0



        self.custom_mutators = [
            self.delete_random_character,
            self.insert_random_character,
            self.flip_random_character,
            self.mutate_input_reduce,
            self.duplicated_mutate,
            self.mutate_depth_exe
        ]

    def next_phase(self):
        self.phase += 1
        self.phase_count = 0



    # Erase a byte from the data
    def insert_random_character(self, s: str) -> str:
        """Returns s with a random character inserted"""
        pos = random.randint(0, len(s))
        random_character = chr(random.randrange(32, 127))
        return s[:pos] + random_character + s[pos:]

    # Insert one or more random bytes into the data
        # Insert one or more random bytes into the data
    def delete_random_character(self, s: str) -> str:
        """Returns s with a random character deleted"""
        if s == "":
            return self.insert_random_character(s)

        pos = random.randint(0, len(s) - 1)
        return s[:pos] + s[pos + 1:]

    # Insert repeated bytes into the data
    def flip_random_character(self, s: str) -> str:
        """Returns s with a random bit flipped in a random position"""
        if s == "":
            return self.insert_random_character(s)

        pos = random.randint(0, len(s) - 1)
        c = s[pos]
        bit = 1 << random.randint(0, 6)
        new_c = chr(ord(c) ^ bit)
        return s[:pos] + new_c + s[pos + 1:]




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


    def duplicated_mutate(self, s: str) -> str:
        prefix_length = random.randint(1, 3)
        suffix_length = random.randint(8, 11)

        prefix = ''.join(random.choices(string.ascii_letters, k=prefix_length))
        suffix = ''.join(random.choices(string.ascii_letters, k=suffix_length))

        # generate duplicated part
        repeating_part = 'really'
        repeat_times = 6

        new_s = prefix + (repeating_part * repeat_times) + suffix
        return new_s


    def mutate_depth_exe(self, s: str) -> str:
        # trace execution depth and time
        start_time = time.time()
        try:
            entrypoint(s)
        except Exception as e:
            pass

        duration = time.time() - start_time

        self.max_duration = max(self.max_duration, duration)

        # Adjust mutation strategy based on execution depth and run time
        if self.chosen_depth > self.max_depth * 0.8 or duration > self.max_duration * 0.8:
            # try specific mutation strategy if execution depth or runtime is near max
            index = random.randrange(len(s))
            char_to_insert = random.choice(string.ascii_letters + string.digits + string.punctuation)
            new_s = s[:index] + char_to_insert + s[index:]
        return new_s


    def mutate(self, inp: Any) -> Any:

        self.phase_count += 1
        if self.phase_count > self.phase_limit:
            self.next_phase()

        # based on shell sort
        if self.phase == 0:
            mutation_methods = [self.delete_random_character, self.insert_random_character]

        elif self.phase == 1:
            mutation_methods = [self.flip_random_character]

        elif self.phase == 2:
            mutation_methods = [self.mutate_input_reduce]

        # based on execution depth
        else:
            if self.chosen_depth > self.depth_threshold:
                mutation_methods = [self.mutate_depth_exe, self.mutate_input_reduce]
            else:
                mutation_methods = [self.delete_random_character, self.insert_random_character,
                                    self.flip_random_character]

        mutator = random.choice(mutation_methods)
        mutated_inp = mutator(inp)

        if isinstance(mutated_inp, bytes):
            return mutated_inp.decode('utf-8', errors='ignore')
        return mutated_inp


    # def mutate(self, inp: Any) -> Any:
    #     mutator = random.choice(self.custom_mutators)
    #     return mutator(inp)




# For mutation strategy -- mutate_depth_exe
# Calculate execution depth -- trace the nesting level of function calls
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
    # convert n-gram coverage into sorted list form to ensure consistency
    sorted_ngrams = sorted(ngram_coverage)
    # serialize sorted n-grams
    pickled = pickle.dumps(sorted_ngrams)
    # return MD5 hash of the serialized data
    return hashlib.md5(pickled).hexdigest()




if __name__ == "__main__":
    seed_inputs = get_initial_corpus()
    fast_schedule = gbf.AFLFastSchedule(5)
    my_mutator = MyMutator()
    line_runner = MyFunctionCoverageRunner(entrypoint)


    fast_fuzzer = gbf.CountingGreyboxFuzzer(seed_inputs, my_mutator, fast_schedule)

    # for checking
    print("Starting fuzzing loop...")
    fast_fuzzer.runs(line_runner, trials=9999999)
    print("Max trials end.")