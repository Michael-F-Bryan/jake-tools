from __future__ import annotations

from jake_tools.ai_watch.cleanup import strip_page_chrome


def test_keeps_article_body() -> None:
    content = """# Skill engineering

Paul Bakaus thinks the emerging discipline of "skill engineering" can make AI agents more capable."""
    result = strip_page_chrome(content)
    assert result == content


def test_removes_substack_subscribe_banner() -> None:
    content = """## **There will be no auto mode**

The AI industry often treats complete automation as the natural endpoint.

#### Subscribe to Latent.Space

Thousands of paid subscribers

The AI Engineer newsletter"""
    result = strip_page_chrome(content)
    assert "#### Subscribe to Latent.Space" not in result
    assert "Thousands of paid subscribers" not in result
    assert "## **There will be no auto mode**" in result
    assert "The AI industry often treats" in result


def test_removes_substack_discussion_section() -> None:
    content = """Impeccable's core innovation.

#### Discussion about this post

CommentsRestacks

TopLatestDiscussions

[The 2025 AI Engineer Reading List](https://example.com)"""
    result = strip_page_chrome(content)
    assert "Impeccable's core innovation" in result
    assert "Discussion about this post" not in result
    assert "CommentsRestacks" not in result
    assert "TopLatestDiscussions" not in result
    assert "Reading List" not in result


def test_removes_github_ui_chrome() -> None:
    content = """Hi MCP maintainers — opening this as a constructive heads-up.

| Package | Days |
| --- | --- |
| `@modelcontextprotocol/server-postgres` | 541 |

## Activity

[![avatar]fjcobu14](https://github.com/fjcobu14)

mentioned this 3w ago

## Metadata

### Assignees

No one assigned

### Labels

No labels

### Issue actions

You can't perform that action at this time."""
    result = strip_page_chrome(content)
    assert "Hi MCP maintainers" in result
    assert "server-postgres" in result
    assert "## Activity" not in result
    assert "mentioned this" not in result
    assert "## Metadata" not in result
    assert "### Assignees" not in result
    assert "### Labels" not in result
    assert "## Issue actions" not in result


def test_removes_share_like_count_lines() -> None:
    content = """## **There will be no auto mode**

64

1

Share

Bakaus rejects that premise."""
    result = strip_page_chrome(content)
    assert "## **There will be no auto mode**" in result
    assert "Bakaus rejects that premise" in result
    assert "Share" not in result
    assert "64" not in result  # standalone number line


def test_removes_avatar_image_lines() -> None:
    content = """# Title

[![Richard's avatar](https://example.com/avatar.png)](https://example.com)

Article text here."""
    result = strip_page_chrome(content)
    assert "# Title" in result
    assert "Article text here" in result
    assert "avatar.png" not in result


def test_removes_hero_image_lines() -> None:
    content = """# Title

[![](https://example.com/hero.jpeg)](https://example.com) Impeccable's Paul Bakaus.

Article text."""
    result = strip_page_chrome(content)
    assert "# Title" in result
    assert "Article text" in result


def test_preserves_table_content() -> None:
    content = """Two questions that would help downstream users:

1. **Are these packages intentionally archived?**

| Package | Days |
| --- | --- |
| `@modelcontextprotocol/server-postgres` | 541 |

Thanks for everything you build."""
    result = strip_page_chrome(content)
    assert "server-postgres" in result
    assert "| Package | Days |" in result
    assert "Thanks for everything" in result


def test_removes_sign_in_notification_lines() -> None:
    content = """Issue text.

You signed in with another tab or window. Reload to refresh your session.

More issue text."""
    result = strip_page_chrome(content)
    assert "Issue text" in result
    assert "You signed in with another tab or window" not in result


def test_collapses_excessive_blank_lines() -> None:
    content = """Paragraph one.



Paragraph two.

"""
    result = strip_page_chrome(content)
    assert "Paragraph one." in result
    assert "Paragraph two." in result
    assert "\n\n\n" not in result


def test_empty_content() -> None:
    assert strip_page_chrome("") == ""
    assert strip_page_chrome("   \n  \n") == ""


def test_removes_substack_subscribe_sign_in() -> None:
    content = """# Title

SubscribeSign in

[AINews: Weekday Roundups](https://example.com)

Article body."""
    result = strip_page_chrome(content)
    assert "# Title" in result
    assert "Article body" in result
    assert "SubscribeSign in" not in result
    assert "AINews" not in result


def test_removes_github_fork_star_notifications() -> None:
    content = """Issue body.

[Notifications](https://github.com/login) You must be signed in to change notification settings
[Fork\\\\11.1k](https://github.com/login)
[Star\\\\88k](https://github.com/login)"""
    result = strip_page_chrome(content)
    assert "Issue body" in result
    assert "Notifications" not in result
    assert "Fork" not in result
    assert "Star" not in result


def test_removes_new_issue_and_copy_link() -> None:
    content = """Issue description.

[New issue](https://github.com/login)

Copy link

Open

More issue description."""
    result = strip_page_chrome(content)
    assert "Issue description" in result
    assert "More issue description" in result
    assert "New issue" not in result
    assert "Copy link" not in result
    assert "Open" not in result
