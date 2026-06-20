from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli.newsletter import newsletter
from jake_tools.newsletters import NewsletterAttachment, NewsletterItem, body_from_html, body_to_html


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

    def create_item(self, *, title: str, body: str, attachments: list[NewsletterAttachment]):
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
            {"item_id": item_id, "title": title, "body": body, "attachments": attachments}
        )
        return NewsletterItem(
            id=item_id,
            title=title or "Existing title",
            body=body or "Existing body",
            created="2026-06-20T10:52:41Z",
            modified="2026-06-21T01:00:00Z",
            url=f"https://example.test/{item_id}",
        )


def test_list_newsletter_items() -> None:
    client = FakeNewsletterClient()
    runner = CliRunner()

    result = runner.invoke(
        newsletter,
        ["list", "--limit", "1", "--body"],
        obj={"newsletter_client": client},
    )

    assert result.exit_code == 0
    assert "ID: 295" in result.output
    assert "Net Control station course" in result.output


def test_add_newsletter_item_reads_body_from_stdin_and_accepts_attachments(tmp_path: Path) -> None:
    attachment = tmp_path / "flyer.pdf"
    attachment.write_bytes(b"fake pdf")
    client = FakeNewsletterClient()
    runner = CliRunner()

    result = runner.invoke(
        newsletter,
        ["add", "Gosnells GPS Workshop", "--attach", str(attachment)],
        input="Gosnells SES are running their GPS workshop again this year.\n",
        obj={"newsletter_client": client},
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


def test_edit_updates_title_body_and_attachments(tmp_path: Path) -> None:
    attachment = tmp_path / "map.png"
    attachment.write_bytes(b"fake image")
    client = FakeNewsletterClient()
    runner = CliRunner()

    result = runner.invoke(
        newsletter,
        ["edit", "295", "--title", "Updated title", "--attach", str(attachment)],
        input="Updated body\n",
        obj={"newsletter_client": client},
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


def test_body_html_round_trip_preserves_paragraphs() -> None:
    body = "First paragraph\nwith a line break.\n\nSecond paragraph."

    html = body_to_html(body)

    assert html == "<p>First paragraph<br>with a line break.</p><p>Second paragraph.</p>"
    assert body_from_html(html) == body
