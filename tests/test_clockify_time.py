from __future__ import annotations

from typing import Any

import pytest
import requests
from test_clockify import FakeResponse, FakeSession

from jake_tools.clockify import ClockifyClient, ClockifyError
from jake_tools.clockify_time import (
    CreateTimeEntryError,
    create_time_entry,
    time_entries,
    workspace_catalogue,
)

START = "2026-01-02T00:00:00Z"
END = "2026-01-03T00:00:00Z"


def entry(
    id: str = "e-1", start: str = START, end: str | None = END, **changes: Any
) -> dict[str, Any]:
    return {
        "id": id,
        "userId": "u-1",
        "workspaceId": "ws-1",
        "description": "work",
        "billable": False,
        "projectId": None,
        "taskId": None,
        "timeInterval": {"start": start, "end": end},
        **changes,
    }


def client(*payloads: object) -> tuple[ClockifyClient, FakeSession]:
    session = FakeSession(*(FakeResponse(200, payload) for payload in payloads))
    return ClockifyClient(api_key="secret", session=session), session


def test_all_history_overlap_running_boundaries_and_original_offsets() -> None:
    rows = [
        entry("ends-at-start", "2020-01-01T00:00:00Z", START),
        entry("starts-at-end", END, "2026-01-04T00:00:00Z"),
        entry("old-running", "2020-01-01T00:00:00+08:00", None),
        entry("straddles", "2026-01-01T23:00:00Z", "2026-01-02T01:00:00Z"),
        entry("inside", "2026-01-02T03:00:00Z", "2026-01-02T04:00:00Z"),
    ]
    api, session = client({"id": "u-1"}, rows)
    result = time_entries(api, "ws-1", start=START, end=END, timezone="Australia/Perth")
    assert [r.entry.id for r in result.entries] == [
        "old-running",
        "straddles",
        "inside",
    ]
    assert result.entries[0].entry.time_interval.start == "2020-01-01T00:00:00+08:00"
    assert result.entries[0].running and result.entries[0].display_end is None
    assert result.entries[1].display_start == "2026-01-02T07:00:00+08:00"
    assert session.requests[1]["params"] == {"page": 1, "page-size": 5000}
    assert result.coverage.reads[0]["pages"] == 1
    assert result.total == result.count == 3


def test_paginated_history_and_query_bound_cursor() -> None:
    rows = [entry("b"), entry("a")]
    api, session = client({"id": "u-1"}, rows, [], {"id": "u-1"}, rows, [])
    api._PAGE_SIZE = 2
    first = time_entries(api, "ws-1", start=START, end=END, limit=1)
    second = time_entries(
        api, "ws-1", start=START, end=END, limit=1, cursor=first.next_cursor
    )
    assert [first.entries[0].entry.id, second.entries[0].entry.id] == ["a", "b"]
    assert second.observed_at == first.observed_at
    assert second.next_cursor is None
    assert first.coverage.reads[0]["pages"] == 2
    assert [r["params"]["page"] for r in session.requests if "params" in r] == [
        1,
        2,
        1,
        2,
    ]
    api, _ = client({"id": "u-1"})
    with pytest.raises(ValueError, match="different query"):
        time_entries(
            api, "ws-1", start=START, end=END, limit=2, cursor=first.next_cursor
        )
    api, _ = client({"id": "u-1"}, [entry("changed")])
    with pytest.raises(ValueError, match="History changed"):
        time_entries(
            api, "ws-1", start=START, end=END, limit=1, cursor=first.next_cursor
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": "2026-01-02", "end": END},
        {"start": END, "end": START},
        {"start": START, "end": END, "limit": 0},
        {"start": START, "end": END, "timezone": "Not/AZone"},
    ],
)
def test_invalid_history_arguments_touch_nothing(kwargs: dict[str, Any]) -> None:
    api, session = client()
    with pytest.raises(ValueError):
        time_entries(api, "ws-1", **kwargs)
    assert session.requests == []


@pytest.mark.parametrize("id", ["../p", "p/q", "p?x=y", "p%2fsecret", "", " p", "."])
def test_ids_cannot_interpolate_paths(id: str) -> None:
    api, session = client()
    with pytest.raises(ValueError):
        create_time_entry(
            api,
            "ws-1",
            start=START,
            end=END,
            description="work",
            billable=False,
            project_id=id,
        )
    assert session.requests == []


def test_pagination_bounds_and_repeated_pages_fail_not_truncate() -> None:
    api, _ = client([entry()], [entry()])
    api._PAGE_SIZE = 1
    with pytest.raises(ClockifyError, match="repeated"):
        api.get_time_entries("ws-1", "u-1")
    api, _ = client([entry()])
    api._PAGE_SIZE = api._MAX_PAGES = 1
    with pytest.raises(ClockifyError, match="page bound"):
        api.get_time_entries("ws-1", "u-1")


def test_catalogue_includes_archived_and_completed_records() -> None:
    api, session = client(
        [{"id": "c-active", "name": "Active", "archived": False}],
        [{"id": "c-old", "name": "Old", "archived": True}],
        [{"id": "p-active", "name": "Active", "archived": False}],
        [{"id": "p-old", "name": "Old", "archived": True}],
        [{"id": "done", "name": "Done", "projectId": "p-active", "status": "DONE"}],
        [
            {
                "id": "active",
                "name": "Active",
                "projectId": "p-active",
                "status": "ACTIVE",
            }
        ],
        [],
        [],
    )
    result = workspace_catalogue(api, "ws-1")
    assert [c.archived for c in result.clients] == [False, True]
    assert [p.archived for p in result.projects] == [False, True]
    assert [t.status for t in result.tasks] == ["ACTIVE", "DONE"]
    assert [t.completed for t in result.tasks] == [False, True]
    assert len(result.coverage.reads) == 8
    assert {r["params"]["archived"] for r in session.requests[:4]} == {"false", "true"}


def test_create_posts_once_then_gets_and_verifies_every_field() -> None:
    api, session = client({"id": "u-1"}, {"id": "created"}, entry("created"))
    result = create_time_entry(
        api,
        "ws-1",
        start="2026-01-02T08:00:00+08:00",
        end="2026-01-03T08:00:00+08:00",
        description="work",
        billable=False,
    )
    assert result.verified
    assert session.requests[1]["allow_redirects"] is False
    assert [r["method"] for r in session.requests] == ["GET", "POST", "GET"]
    assert session.requests[1]["json"] == {
        "start": "2026-01-02T00:00:00+00:00",
        "end": "2026-01-03T00:00:00+00:00",
        "description": "work",
        "billable": False,
        "projectId": None,
        "taskId": None,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"description": "different"},
        {"billable": True},
        {"projectId": "unexpected"},
        {"userId": "someone-else"},
        {"workspaceId": "elsewhere"},
        {"timeInterval": {"start": START, "end": None}},
    ],
)
def test_verification_failure_keeps_created_id(change: dict[str, Any]) -> None:
    api, session = client({"id": "u-1"}, {"id": "created"}, entry("created", **change))
    with pytest.raises(CreateTimeEntryError) as exc:
        create_time_entry(
            api, "ws-1", start=START, end=END, description="work", billable=False
        )
    assert exc.value.outcome == "verification_failed"
    assert exc.value.entry_id == "created"
    assert len([r for r in session.requests if r["method"] == "POST"]) == 1


def test_invalid_known_created_id_is_preserved_but_never_interpolated() -> None:
    api, session = client({"id": "u-1"}, {"id": "../known-id"})
    with pytest.raises(CreateTimeEntryError) as exc:
        create_time_entry(
            api, "ws-1", start=START, end=END, description="work", billable=False
        )
    assert (exc.value.outcome, exc.value.entry_id) == (
        "verification_unavailable",
        "../known-id",
    )
    assert [r["method"] for r in session.requests] == ["GET", "POST"]


def test_failed_get_keeps_created_id_and_does_not_retry() -> None:
    session = FakeSession(
        FakeResponse(200, {"id": "u-1"}),
        FakeResponse(201, {"id": "created"}),
        FakeResponse(503, {}),
    )
    api = ClockifyClient(api_key="secret", session=session)
    with pytest.raises(CreateTimeEntryError) as exc:
        create_time_entry(
            api, "ws-1", start=START, end=END, description="work", billable=False
        )
    assert (exc.value.outcome, exc.value.entry_id) == (
        "verification_unavailable",
        "created",
    )
    assert len(session.requests) == 3


def test_uncertain_post_never_retries() -> None:
    class TimeoutSession(FakeSession):
        def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            if method == "POST":
                self.requests.append({"method": method})
                raise requests.Timeout("upstream might have committed")
            return super().request(method, url, **kwargs)

    session = TimeoutSession(FakeResponse(200, {"id": "u-1"}))
    with pytest.raises(CreateTimeEntryError) as exc:
        create_time_entry(
            ClockifyClient(api_key="secret", session=session),
            "ws-1",
            start=START,
            end=END,
            description="work",
            billable=False,
        )
    assert (exc.value.outcome, exc.value.entry_id) == ("post_uncertain", None)
    assert [r["method"] for r in session.requests] == ["GET", "POST"]


@pytest.mark.parametrize(
    "kwargs", [{"end": START}, {"start": "2026-01-02T00:00:00"}, {"task_id": "t"}]
)
def test_create_validation_never_posts(kwargs: dict[str, Any]) -> None:
    api, session = client()
    args = {
        "start": START,
        "end": END,
        "description": "work",
        "billable": False,
        **kwargs,
    }
    with pytest.raises(ValueError):
        create_time_entry(api, "ws-1", **args)
    assert not session.requests


def test_name_resolution_failure_is_an_error_not_silent_gap() -> None:
    session = FakeSession(
        FakeResponse(200, {"id": "u-1"}),
        FakeResponse(200, [entry(projectId="p")]),
        FakeResponse(404, {"message": "removed project"}),
    )
    with pytest.raises(ClockifyError, match="404"):
        time_entries(
            ClockifyClient(api_key="secret", session=session),
            "ws-1",
            start=START,
            end=END,
        )


def test_catalogue_pages_to_exhaustion_in_each_filtered_collection() -> None:
    api, _ = client(
        [{"id": "c", "name": "Client", "archived": False}],
        [],
        [],
        [{"id": "p", "name": "Project", "archived": False}],
        [],
        [],
        [{"id": "t", "name": "Task", "projectId": "p", "status": "DONE"}],
        [],
        [],
    )
    api._PAGE_SIZE = 1
    result = workspace_catalogue(api, "ws-1")
    assert len(result.clients) == len(result.projects) == len(result.tasks) == 1
    assert [read["pages"] for read in result.coverage.reads] == [2, 1, 2, 1, 2, 1]


def test_post_redirect_and_invalid_response_are_uncertain_and_not_retried() -> None:
    for response in (FakeResponse(307, {"id": "created"}), FakeResponse(201, {})):
        session = FakeSession(FakeResponse(200, {"id": "u-1"}), response)
        with pytest.raises(CreateTimeEntryError, match="post_uncertain"):
            create_time_entry(
                ClockifyClient(api_key="secret", session=session),
                "ws-1",
                start=START,
                end=END,
                description="work",
                billable=False,
            )
        assert [r["method"] for r in session.requests] == ["GET", "POST"]
        assert session.requests[1]["allow_redirects"] is False


def test_task_project_validation_before_write() -> None:
    api, session = client(
        {"id": "u-1"},
        {"id": "p", "name": "Project"},
        {"id": "t", "name": "Task", "projectId": "other", "status": "ACTIVE"},
    )
    with pytest.raises(ValueError, match="does not belong"):
        create_time_entry(
            api,
            "ws-1",
            start=START,
            end=END,
            description="work",
            billable=True,
            project_id="p",
            task_id="t",
        )
    assert all(r["method"] == "GET" for r in session.requests)


def test_zero_length_entries_do_not_overlap() -> None:
    api, _ = client(
        {"id": "u-1"}, [entry("zero", "2026-01-02T12:00:00Z", "2026-01-02T12:00:00Z")]
    )
    assert time_entries(api, "ws-1", start=START, end=END).total == 0


@pytest.mark.parametrize("start", ["1", "2026-01-02T00:00:00"])
def test_timestamp_requires_explicit_timezone(start: str) -> None:
    api, session = client()
    with pytest.raises(ValueError, match="timezone-aware"):
        time_entries(api, "ws-1", start=start, end=END)
    assert session.requests == []


def test_running_future_entries_and_other_users_are_not_included() -> None:
    api, _ = client({"id": "u-1"}, [entry("future", "9998-01-01T00:00:00Z", None)])
    assert (
        time_entries(
            api, "ws-1", start="9998-01-01T00:00:00Z", end="9999-01-01T00:00:00Z"
        ).entries
        == []
    )
    api, _ = client({"id": "u-1"}, [entry(userId="someone-else")])
    with pytest.raises(ClockifyError, match="outside"):
        time_entries(api, "ws-1", start=START, end=END)
