import json

from jake_tools import newsletters
from jake_tools.newsletters import (
    AzureCliTokenProvider,
    GraphListItem,
    GraphListItemsResponse,
    NewsletterItem,
)


def test_graph_list_item_parses_all_fields() -> None:
    item = GraphListItem.model_validate(
        {
            "id": "295",
            "webUrl": "https://example.test/295",
            "fields": {
                "Title": "Support requested",
                "Body": "<p>Cockburn SES are running a course.<br>Bring a radio.</p>",
                "Created": "2026-06-20T10:52:41Z",
                "Modified": "2026-06-21T01:00:00Z",
            },
        }
    ).to_newsletter_item()

    assert item == NewsletterItem(
        id="295",
        title="Support requested",
        body="Cockburn SES are running a course.\nBring a radio.",
        created="2026-06-20T10:52:41Z",
        modified="2026-06-21T01:00:00Z",
        url="https://example.test/295",
    )


def test_graph_list_item_uses_defaults_for_missing_fields() -> None:
    item = GraphListItem.model_validate({"id": "295"}).to_newsletter_item()

    assert item == NewsletterItem(
        id="295",
        title="",
        body="",
        created="",
        modified="",
        url="",
    )


def test_graph_list_response_parses_items() -> None:
    response = GraphListItemsResponse.model_validate(
        {
            "value": [
                {
                    "fields": {
                        "id": "field-id",
                        "Title": "Title from SharePoint fields",
                    }
                }
            ]
        }
    )

    assert [item.to_newsletter_item() for item in response.value] == [
        NewsletterItem(
            id="field-id",
            title="Title from SharePoint fields",
            body="",
            created="",
            modified="",
            url="",
        )
    ]


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _az_output(token: str, expires_on: float | None) -> str:
    payload: dict[str, object] = {"accessToken": token, "tokenType": "Bearer"}
    if expires_on is not None:
        payload["expires_on"] = int(expires_on)
    return json.dumps(payload)


def test_azure_cli_token_provider_reuses_a_token_only_until_it_expires(
    monkeypatch,
) -> None:
    # `az account get-access-token` spawns a subprocess (~1s) and one command
    # asks for the same resource's token repeatedly, so a live token is
    # reused; but never past the expiry the CLI reports.
    clock = _Clock()
    calls: list[list[str]] = []

    def fake_check_output(command: list[str], **kwargs: object) -> str:
        calls.append(command)
        return _az_output(f"token-{len(calls)}", clock.now + 3600)

    monkeypatch.setattr(newsletters.subprocess, "check_output", fake_check_output)
    provider = AzureCliTokenProvider(clock=clock)

    assert provider("https://graph.microsoft.com") == "token-1"
    assert provider("https://graph.microsoft.com") == "token-1"
    assert len(calls) == 1

    clock.now += 3600  # a long-running process must not keep the expired token
    assert provider("https://graph.microsoft.com") == "token-2"
    assert len(calls) == 2


def test_azure_cli_token_without_a_usable_expiry_is_never_reused(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_check_output(command: list[str], **kwargs: object) -> str:
        calls.append(command)
        return _az_output(f"token-{len(calls)}", None)

    monkeypatch.setattr(newsletters.subprocess, "check_output", fake_check_output)
    provider = AzureCliTokenProvider()

    assert provider("https://graph.microsoft.com") == "token-1"
    assert provider("https://graph.microsoft.com") == "token-2"


def test_azure_cli_token_provider_fetches_each_resource_separately(
    monkeypatch,
) -> None:
    clock = _Clock()

    def fake_check_output(command: list[str], **kwargs: object) -> str:
        resource = command[command.index("--resource") + 1]
        return _az_output(f"token-for-{resource}", clock.now + 3600)

    monkeypatch.setattr(newsletters.subprocess, "check_output", fake_check_output)
    provider = AzureCliTokenProvider(clock=clock)

    graph_token = provider("https://graph.microsoft.com")
    sharepoint_token = provider("https://csuses.sharepoint.com")

    assert graph_token == "token-for-https://graph.microsoft.com"
    assert sharepoint_token == "token-for-https://csuses.sharepoint.com"
