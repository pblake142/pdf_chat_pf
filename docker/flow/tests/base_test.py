import unittest
import os
import time
import traceback

class BaseTest(unittest.TestCase):
    def setUp(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
        self.flow_path = os.path.join(root, "chat-with-pdf")
        self.data_path = os.path.join(
            self.flow_path, "data/bert-paper-qna-3-line.jsonl"
        )
        self.eval_groundedness_flow_path = os.path.join(
            root, "../evaluation/eval-groundedness"
        )

# development of script paused as I interrogate the evaluatuon process 