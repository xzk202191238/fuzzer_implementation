import sys

from fuzzingbook import GreyboxFuzzer as gbf
from fuzzingbook import Coverage as cv
from fuzzingbook import MutationFuzzer as mf

from typing import List, Set, Any, Tuple, Dict, Union

import traceback
import numpy as np
import time

from bug import entrypoint
from bug import get_initial_corpus



class MyCoverage(cv.Coverage):
    def coverage(self) -> Set[cv.Location]:
        """The set of executed lines, as (function_name, line_number) pairs"""
        # print(self.trace())
        return self.trace()



class MyFunctionCoverageRunner(mf.FunctionRunner):
    def run_function(self, inp: str) -> Any:
        with MyCoverage() as cov:
            try:
                result = super().run_function(inp)
            except Exception as exc:
                self._coverage = cov.coverage()
                raise exc

        self._coverage = cov.coverage()
        return result

    def coverage(self) -> Set[cv.Location]:
        return self._coverage



if __name__ == "__main__":
    seed_inputs = sys.argv[1]
    print(seed_inputs)
    fast_schedule = gbf.AFLFastSchedule(5)
    line_runner = MyFunctionCoverageRunner(entrypoint)

    fast_fuzzer = gbf.CountingGreyboxFuzzer(seed_inputs, gbf.Mutator(), fast_schedule)
    fast_fuzzer.runs(line_runner, trials=999999999)
