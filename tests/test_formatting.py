from data_pipeline.cleaning.deduplicator import deduplicate
from data_pipeline.cleaning.quality_filter import is_good
from data_pipeline.formatting.chat_template import build_user_message, to_messages
from data_pipeline.formatting.dataset_splitter import split_records
from data_pipeline.formatting.instruction_formatter import format_record


def test_format_support_record():
    rec = {"domain": "support", "ticket": " Where is my order? ", "resolution": "It ships today."}
    out = format_record(rec)
    assert out["input"] == "Where is my order?"
    assert out["output"] == "It ships today."
    assert "support reply" in out["instruction"]


def test_unknown_domain_returns_none():
    assert format_record({"domain": "cooking"}) is None


def test_messages_structure():
    ex = {"instruction": "Do X.", "input": "data", "output": "done"}
    msgs = to_messages(ex)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert build_user_message(ex) == "Do X.\n\ndata"


def test_dedup():
    rows = [{"domain": "a", "q": "Hello  World"}, {"domain": "a", "q": "hello world"}]
    assert len(deduplicate(rows)) == 1


def test_quality_filter():
    assert not is_good({"domain": "a", "q": "hi"})
    assert is_good({"domain": "a", "q": "This is a reasonably long and clean example sentence."})


def test_split_sizes_and_no_overlap():
    rows = [{"domain": "legal", "id": i} for i in range(100)]
    train, val, test = split_records(rows, 0.8, 0.1, seed=1)
    assert len(train) == 80 and len(val) == 10 and len(test) == 10
    ids = [r["id"] for r in train + val + test]
    assert len(set(ids)) == 100