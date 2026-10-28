from pathlib import Path

from eval.run_eval import Result, evaluate, load_items, summarize

from rag.pdf import read_pdf

QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.sample.json"


def test_the_sample_questions_load():
    items = load_items(QUESTIONS)
    assert len(items) == 6
    assert all(i.question and i.expect and i.page for i in items)


def test_the_sample_questions_match_the_sample_pdf(sample_bytes):
    """If someone edits the sample PDF or the questions, this keeps the two in step."""
    pages = {p.number: " ".join(p.text.split()).lower() for p in read_pdf(sample_bytes)}
    for item in load_items(QUESTIONS):
        assert item.expect.lower() in pages[item.page], item.question


def test_evaluate_runs_every_question(pipeline, sample_bytes):
    doc = pipeline.ingest(sample_bytes, "how-rag-works.pdf")
    results = evaluate(pipeline, doc.namespace, load_items(QUESTIONS))
    assert len(results) == 6
    assert all(r.retrieved for r in results)  # the sample is small enough that vector search returns every chunk
    assert any(r.reranked for r in results)
    assert all(r.page is not None and r.answered is not None for r in results)


def test_summary_counts_only_the_questions_that_have_the_field():
    results = [
        Result("a", True, True, True, True, ""),
        Result("b", True, False, None, None, ""),
        Result("c", False, False, False, False, ""),
    ]
    assert summarize(results) == {"retrieved": "2/3", "reranked": "1/3", "page": "1/2", "answered": "1/2"}
    assert summarize([]) == {"retrieved": "n/a", "reranked": "n/a", "page": "n/a", "answered": "n/a"}
