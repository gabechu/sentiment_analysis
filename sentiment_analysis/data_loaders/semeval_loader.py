import logging
from typing import Dict

from pandas import DataFrame

logger = logging.getLogger(__name__)


class SemEvalLoader(object):
    # map column and indices for subtasks
    _subtasks = {
        # 3-point scale: POSITIVE, NEGATIVE or NEUTRAL
        "A": {"id": 0, "label": 1, "tweet": 2},
        # 2-point scale: POSITIVE or NEGATIVE
        "B": {"id": 0, "topic": 1, "label": 2, "tweet": 3},
        # 5-point scale: STRONGLYPOSITIVE, WEAKLYPOSITIVE, NEUTRAL, WEAKLYNEGATIVE, and
        # STRONGLYNEGATIVE
        "C": {"id": 0, "topic": 1, "label": 2, "tweet": 3},
    }

    def __init__(self, input_file: str, language: str, task_name: str):
        self.input_file = input_file
        # users should be responsible of assigning the language,
        # we will not be validating it
        self.language = language

        # use list because order matters for unittest
        subtask_set = ["A", "B", "C"]
        if task_name in subtask_set:
            self.task_name = task_name
        else:
            raise ValueError(
                f"Do not support subtask {task_name}, "
                f"available subtasks are {subtask_set}"
            )

    def _parse_line(self, line: str, task_fields: Dict[str, int]) -> Dict[str, str]:
        """Parse a line read from a SemEval file."""
        splits = line.rstrip().split("\t")

        if len(splits) != len(task_fields):
            # Parse path to get file name only
            file_name = self.input_file.split("/")[-1]
            logger.warning(f"Found invalid record: {line} in file {file_name}.")
            return None

        return {
            field_name: splits[field_index]
            for field_name, field_index in task_fields.items()
        }

    def load_test_data(self) -> DataFrame:
        """Load data to pandas DataFrame on columns of label and text."""
        lines = open(self.input_file).read().splitlines()
        task_fields = self._subtasks[self.task_name]

        data = []
        for line in lines:
            record = self._parse_line(line, task_fields)
            if record:
                data.append(record)
        return DataFrame(data)
