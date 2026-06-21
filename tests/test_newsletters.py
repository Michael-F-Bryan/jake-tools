from jake_tools.newsletters import GraphListItem, GraphListItemsResponse, NewsletterItem


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
