"""0.9.0 Inspect adapter: native .eval reading, message-text length, item grouping, scorer keys, drop accounting.
Fixture: tests/fixtures/inspect_chat_3epochs.eval, made by tests/fixtures/make_inspect_chat_fixture.py
(inspect_ai 0.3.292, mockllm, no API keys): 30 samples x 3 epochs, chat-list and long string inputs,
scorers match (C/I), risk (numeric) and multi (dict)."""
import json
import math
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from baseline_increment import adapters
from baseline_increment.adapters import _correctness, _text, from_inspect, list_inspect

FIX = Path(__file__).parent / "fixtures" / "inspect_chat_3epochs.eval"


def _true_words(sample):
    inp = sample["input"]
    if isinstance(inp, str):
        txt = inp
    else:
        txt = " ".join(m["content"] if isinstance(m["content"], str) else " ".join(c.get("text", "") for c in m["content"])
                       for m in inp)
    return round(math.log1p(len(txt.split())), 4)


def test_native_eval_reader_reproduces_inspect_accuracy():
    with zipfile.ZipFile(FIX) as z:
        head = json.loads(z.read("header.json"))
    acc = [m["value"] for s in head["results"]["scores"] if s["name"] == "match"
           for k, m in s["metrics"].items() if k == "accuracy"][0]
    df = from_inspect([FIX], "match", score_scorer="risk")
    assert len(df) == 90 and abs((1 - df.wrong.mean()) - acc) < 1e-12


def test_native_equals_inspect_ai_reader(monkeypatch):
    pytest.importorskip("inspect_ai")
    native = from_inspect([FIX], "match", score_scorer="multi:confidence")
    monkeypatch.setattr(adapters, "_read_eval_zip", lambda p: (_ for _ in ()).throw(RuntimeError("force fallback")))
    via_lib = from_inspect([FIX], "match", score_scorer="multi:confidence")
    key = ["sample_id", "epoch"]
    pd.testing.assert_frame_equal(native.sort_values(key).reset_index(drop=True), via_lib.sort_values(key).reset_index(drop=True))


def test_input_length_counts_message_text_only():
    df = from_inspect([FIX], "match", score_scorer="risk")
    with zipfile.ZipFile(FIX) as z:
        samples = {(s["id"], s["epoch"]): s for s in (json.loads(z.read(n)) for n in z.namelist() if n.startswith("samples/"))}
    exp = [_true_words(samples[(i, e)]) for i, e in zip(df.sample_id, df.epoch)]
    assert list(df.log_input_words) == exp
    assert any(isinstance(s["input"], list) for s in samples.values())        # the fixture really has chat inputs


def test_items_group_across_epochs():
    df = from_inspect([FIX], "match", score_scorer="risk")
    assert df.group.nunique() == 30 and (df.groupby("group").size() == 3).all()
    old = from_inspect([FIX], "match", score_scorer="risk", group_by="file")
    assert old.group.nunique() == 30 and old.group.iloc[0].startswith(FIX.stem)


def test_dict_scorer_needs_key_and_key_works():
    with pytest.raises(ValueError, match="multi:<key>"):
        from_inspect([FIX], "multi", score_scorer="risk")
    df = from_inspect([FIX], "multi:exact", score_scorer="multi:confidence")
    assert len(df) == 90 and set(df.wrong) <= {0, 1}


def test_list_inspect_reports_scorers_and_keys():
    info = list_inspect([FIX.parent])
    assert info["scorers"]["multi"]["keys"] == ["confidence", "exact"]
    assert "difficulty" in info["metadata_fields"]


def test_partial_and_noanswer_policies():
    assert _correctness("P") is None and _correctness("P", partial_as="wrong") == 1
    assert _correctness("N") == 1 and _correctness("N", noanswer_as="drop") is None
    assert _correctness({"a": 1}) is None and _correctness("C") == 0 and _correctness(0.2) == 1


def test_text_resolves_attachments_and_content_parts():
    att = {"abc": "long system prompt here"}
    msgs = [{"role": "system", "content": "attachment://abc", "id": "x1"},
            {"role": "user", "content": [{"type": "text", "text": "what is two"}, {"type": "image", "image": "data:..."}]}]
    assert _text(msgs, att) == "long system prompt here what is two"


def _mini_log(path, inputs, scores):
    samples = [{"id": f"s{i}", "epoch": 1, "input": inp, "output": {"completion": "x"},
                "scores": {"m": {"value": v}, "r": {"value": 0.5}}} for i, (inp, v) in enumerate(zip(inputs, scores))]
    path.write_text(json.dumps({"eval": {"model": "m/" + path.stem, "task": "t"}, "samples": samples}))


def test_id_mismatch_across_logs_is_an_error(tmp_path):
    _mini_log(tmp_path / "a.json", ["q one", "q two"], ["C", "I"])
    _mini_log(tmp_path / "b.json", ["q one", "a different question"], ["C", "I"])
    with pytest.raises(ValueError, match="different input text"):
        from_inspect([tmp_path], "m", score_scorer="r")


def test_dropped_rows_are_counted(tmp_path):
    _mini_log(tmp_path / "a.json", ["q1", "q2", "q3"], ["C", "P", "I"])
    df = from_inspect([tmp_path / "a.json"], "m", score_scorer="r")
    assert len(df) == 2 and sum(df.attrs["dropped"].values()) == 1
