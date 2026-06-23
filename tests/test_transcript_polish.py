from jake_tools.hermes import Reply
from jake_tools.transcripts.polish import polish_transcript


class FakeHermes:
    def __init__(self, reply: Reply) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def run(self, prompt: str) -> Reply:
        self.prompts.append(prompt)
        return self.reply


def test_polish_transcript_uses_run_text_response() -> None:
    hermes = FakeHermes(Reply(text="Polished transcript"))

    polished = polish_transcript(hermes, "raw transcript text")

    assert polished == "Polished transcript"
    assert "raw transcript text" in hermes.prompts[0]


def test_polish_transcript_raises_when_reply_has_no_text() -> None:
    hermes = FakeHermes(Reply(text=None))

    try:
        polish_transcript(hermes, "raw transcript text")
    except ValueError as exc:
        assert str(exc) == "No response from Hermes"
    else:
        raise AssertionError("expected ValueError")
