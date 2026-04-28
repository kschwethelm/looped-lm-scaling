from scripts.base_eval import _filter_tasks

def test_filter_tasks_allowlist():
    tasks = [{"label": "arc_easy"}, {"label": "piqa"}, {"label": "openbook_qa"}]
    out = _filter_tasks(tasks, allowlist={"arc_easy", "openbook_qa"})
    assert [t["label"] for t in out] == ["arc_easy", "openbook_qa"]

def test_filter_tasks_none_passes_through():
    tasks = [{"label": "arc_easy"}, {"label": "piqa"}]
    assert _filter_tasks(tasks, allowlist=None) == tasks

def test_filter_tasks_empty_allowlist_yields_empty():
    tasks = [{"label": "arc_easy"}]
    assert _filter_tasks(tasks, allowlist=set()) == []
