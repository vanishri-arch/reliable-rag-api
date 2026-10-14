from app.answering import generate_answer, NO_ANSWER

def test_no_evidence_abstains_without_api_call():
    assert generate_answer("What?", [], "", "unused") == NO_ANSWER
