"""CPU simulation verifies graph buffer updates and independent branch slots."""

from contextlib import contextmanager, ExitStack
from typing import NamedTuple
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_gen_graph import run_full_gen_graph


class Prep(NamedTuple):
    hidden_gen: object
    s_video: int = 2
    s_action: int = 0
    has_action: bool = True
    use_multi_control_attention: bool = False
    ulysses_size: int = 1
    has_control: bool = False
    has_sound: bool = False


class Stream:
    def wait_stream(self, other):
        pass


class Graph:
    def replay(self):
        self.output.copy_(self.compute())


class BufferTest(unittest.TestCase):
    def test_branch_buffers_refresh_and_postprocess_output_survives(self):
        active = []
        calls = []

        @contextmanager
        def context(*args, **kwargs):
            yield

        @contextmanager
        def graph_context(graph, **kwargs):
            active.append(graph)
            try:
                yield
            finally:
                active.pop()

        model = SimpleNamespace(gen_layers=[None] * 2)

        def original(prep):
            hidden = prep.hidden_gen
            kv = model.cached_kv
            rope = model.cached_freqs_gen

            def compute():
                return (
                    hidden
                    + sum(k.sum() + v.sum() for k, v in kv)
                    + sum(x.sum() for x in rope)
                )

            out = compute()
            calls.append(1)
            if active:
                active[-1].compute, active[-1].output = compute, out
            return out

        def invoke(length, value):
            model.cached_kv = [
                (
                    torch.full((1, length, 1, 1), value),
                    torch.full((1, length, 1, 1), value + 1),
                )
                for _ in range(2)
            ]
            model.cached_freqs_gen = (
                torch.full((1, 2, 1, 1), value + 2),
                torch.full((1, 2, 1, 1), value + 3),
            )
            prep = Prep(torch.full((1, 2, 2), value + 4))
            expected = original(prep)
            calls.pop()
            kv = model.cached_kv
            rope = model.cached_freqs_gen
            result = run_full_gen_graph(model, prep, original)
            self.assertIs(model.cached_kv, kv)
            self.assertIs(model.cached_freqs_gen, rope)
            self.assertTrue(torch.equal(result, expected))
            return result * 2  # models graph-external allocating postprocess

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"WAM_FULL_GEN_AUDIT": ""}))
            for name, value in {
                "CUDAGraph": Graph,
                "Stream": Stream,
                "current_stream": lambda: Stream(),
                "stream": context,
                "graph": graph_context,
                "synchronize": lambda: None,
            }.items():
                stack.enter_context(patch.object(torch.cuda, name, value))
            a = invoke(3, 1.0)
            frozen = a.clone()
            invoke(1, 5.0)
            invoke(3, 9.0)
            again = invoke(3, 1.0)
            self.assertTrue(torch.equal(a, frozen))
            self.assertTrue(torch.equal(again, frozen))
            self.assertEqual(len(model._wam_gen_graph_states), 2)
            self.assertEqual(len(calls), 8)  # three warmups + one capture, two branches


if __name__ == "__main__":
    unittest.main()
