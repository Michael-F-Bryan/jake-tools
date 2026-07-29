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


def test_azure_cli_token_provider_caches_token_for_the_process_lifetime(
    monkeypatch,
) -> None:
    # `az account get-access-token` spawns a subprocess (~1s). A single
    # command (e.g. create_item with attachments) can ask for the same
    # resource's token several times; only the first ask should shell out.
    calls: list[list[str]] = []

    def fake_check_output(command: list[str], **kwargs: object) -> str:
        calls.append(command)
        return "token-value\n"

    monkeypatch.setattr(newsletters.subprocess, "check_output", fake_check_output)
    provider = AzureCliTokenProvider()

    first = provider("https://graph.microsoft.com")
    second = provider("https://graph.microsoft.com")

    assert first == "token-value"
    assert second == "token-value"
    assert len(calls) == 1


def test_azure_cli_token_provider_fetches_each_resource_separately(
    monkeypatch,
) -> None:
    def fake_check_output(command: list[str], **kwargs: object) -> str:
        resource = command[command.index("--resource") + 1]
        return f"token-for-{resource}\n"

    monkeypatch.setattr(newsletters.subprocess, "check_output", fake_check_output)
    provider = AzureCliTokenProvider()

    graph_token = provider("https://graph.microsoft.com")
    sharepoint_token = provider("https://csuses.sharepoint.com")

    assert graph_token == "token-for-https://graph.microsoft.com"
    assert sharepoint_token == "token-for-https://csuses.sharepoint.com"
