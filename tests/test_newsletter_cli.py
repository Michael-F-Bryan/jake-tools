import importlib
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from jake_tools.cli.newsletter import newsletter
from jake_tools.newsletters import (
    NewsletterAttachment,
    NewsletterItem,
    body_from_html,
    body_to_html,
)

newsletter_cli = importlib.import_module("jake_tools.cli.newsletter")


class TtyStdin:
    """A stdin stand-in that reports as an interactive terminal.

    Reading from it raises, so any test that reaches ``read()`` on this
    object demonstrates the bug this fixture guards against: blocking
    forever waiting for EOF on a real terminal.
    """

    def isatty(self) -> bool:
        return True

    def read(self) -> str:
        raise AssertionError("must not read from stdin when it is a tty")


class FakeNewsletterClient:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.items = [
            NewsletterItem(
                id="295",
                title="Support requested: Net Control station course at Cockburn SES",
                body="Cockburn SES are running a Net Control station course.",
                created="2026-06-20T10:52:41Z",
                modified="2026-06-20T10:52:41Z",
                url="https://example.test/295",
            )
        ]

    def list_items(self, *, limit: int):
        return self.items[:limit]

    def create_item(
        self, *, title: str, body: str, attachments: list[NewsletterAttachment]
    ):
        self.created.append({"title": title, "body": body, "attachments": attachments})
        return NewsletterItem(
            id="296",
            title=title,
            body=body,
            created="2026-06-21T01:00:00Z",
            modified="2026-06-21T01:00:00Z",
            url="https://example.test/296",
        )

    def update_item(
        self,
        item_id: str,
        *,
        title: str | None,
        body: str | None,
        attachments: list[NewsletterAttachment],
    ):
        self.updated.append(
            {
                "item_id": item_id,
                "title": title,
                "body": body,
                "attachments": attachments,
            }
        )
        return NewsletterItem(
            id=item_id,
            title=title or "Existing title",
            body=body or "Existing body",
            created="2026-06-20T10:52:41Z",
            modified="2026-06-21T01:00:00Z",
            url=f"https://example.test/{item_id}",
        )


def test_list_newsletter_items(monkeypatch) -> None:
    client = FakeNewsletterClient()
    monkeypatch.setattr(newsletter_cli, "NewsletterClient", lambda: client)
    runner = CliRunner()

    result = runner.invoke(
        newsletter,
        ["list", "--limit", "1", "--body"],
    )

    assert result.exit_code == 0
    assert "ID: 295" in result.output
    assert "Net Control station course" in result.output


def test_add_newsletter_item_reads_body_from_stdin_and_accepts_attachments(
    tmp_path: Path, monkeypatch
) -> None:
    attachment = tmp_path / "flyer.pdf"
    attachment.write_bytes(b"fake pdf")
    client = FakeNewsletterClient()
    monkeypatch.setattr(newsletter_cli, "NewsletterClient", lambda: client)
    runner = CliRunner()

    result = runner.invoke(
        newsletter,
        ["add", "Gosnells GPS Workshop", "--attach", str(attachment)],
        input="Gosnells SES are running their GPS workshop again this year.\n",
    )

    assert result.exit_code == 0
    assert "ID: 296" in result.output
    assert client.created == [
        {
            "title": "Gosnells GPS Workshop",
            "body": "Gosnells SES are running their GPS workshop again this year.\n",
            "attachments": [NewsletterAttachment(attachment)],
        }
    ]


def test_add_requires_body_on_stdin() -> None:
    runner = CliRunner()

    result = runner.invoke(newsletter, ["add", "Empty item"], input="")

    assert result.exit_code != 0
    assert "newsletter body is required on stdin" in result.output


def test_edit_updates_title_body_and_attachments(tmp_path: Path, monkeypatch) -> None:
    attachment = tmp_path / "map.png"
    attachment.write_bytes(b"fake image")
    client = FakeNewsletterClient()
    monkeypatch.setattr(newsletter_cli, "NewsletterClient", lambda: client)
    runner = CliRunner()

    result = runner.invoke(
        newsletter,
        ["edit", "295", "--title", "Updated title", "--attach", str(attachment)],
        input="Updated body\n",
    )

    assert result.exit_code == 0
    assert "Updated title" in result.output
    assert client.updated == [
        {
            "item_id": "295",
            "title": "Updated title",
            "body": "Updated body\n",
            "attachments": [NewsletterAttachment(attachment)],
        }
    ]


def test_edit_requires_at_least_one_change() -> None:
    runner = CliRunner()

    result = runner.invoke(newsletter, ["edit", "295"], input="")

    assert result.exit_code != 0
    assert "nothing to update" in result.output


def test_edit_rejects_non_numeric_item_id_before_any_client_call(
    monkeypatch,
) -> None:
    # item_id is interpolated directly into Graph/SharePoint URLs, so a
    # malformed id must fail at argument parsing, before NewsletterClient
    # is ever constructed or called.
    def fail_if_constructed() -> None:
        raise AssertionError("NewsletterClient must not be constructed")

    monkeypatch.setattr(newsletter_cli, "NewsletterClient", fail_if_constructed)
    runner = CliRunner()

    result = runner.invoke(
        newsletter,
        ["edit", "not-an-id", "--title", "x"],
    )

    assert result.exit_code != 0
    assert "not a valid newsletter item id" in result.output


def test_edit_title_only_succeeds_with_no_stdin_body(monkeypatch) -> None:
    client = FakeNewsletterClient()
    monkeypatch.setattr(newsletter_cli, "NewsletterClient", lambda: client)
    runner = CliRunner()

    result = runner.invoke(
        newsletter,
        ["edit", "295", "--title", "Updated title"],
        input="",
    )

    assert result.exit_code == 0
    assert client.updated == [
        {
            "item_id": "295",
            "title": "Updated title",
            "body": None,
            "attachments": [],
        }
    ]


def test_optional_stdin_body_skips_read_on_a_tty(monkeypatch) -> None:
    monkeypatch.setattr(newsletter_cli.sys, "stdin", TtyStdin())

    assert newsletter_cli._optional_stdin_body() is None


def test_require_stdin_body_errors_without_reading_on_a_tty(monkeypatch) -> None:
    # `newsletter add ID` on a real terminal used to call sys.stdin.read()
    # unconditionally and hang forever waiting for EOF. TtyStdin.read() raises
    # instead of blocking, so reaching it would fail this test loudly.
    monkeypatch.setattr(newsletter_cli.sys, "stdin", TtyStdin())

    with pytest.raises(click.UsageError, match="required on stdin"):
        newsletter_cli._require_stdin_body()


def test_body_html_round_trip_preserves_paragraphs() -> None:
    body = "First paragraph\nwith a line break.\n\nSecond paragraph."

    html = body_to_html(body)

    assert (
        html == "<p>First paragraph<br>with a line break.</p><p>Second paragraph.</p>"
    )
    assert body_from_html(html) == body
