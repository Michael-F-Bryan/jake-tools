"""Exact work-item lookup through the real ClockifyClient and a routing session."""

from __future__ import annotations

import json

import pytest
from clockify_fixtures import API_KEY, RoutingSession, Workspace, sunfish_workspace

from jake_tools.clockify import ClockifyClient, ClockifyError
from jake_tools.clockify_time import (
    MAX_CANDIDATES_PER_KEY,
    MAX_UNREAD_PROJECTS,
    lookup_work_items,
    workspace_catalogue,
)


def client_for(ws: Workspace, page_size: int | None = None) -> ClockifyClient:
    api = ClockifyClient(
        api_key=API_KEY, base_url="http://clockify.test", session=RoutingSession(ws)
    )
    if page_size is not None:
        api._PAGE_SIZE = page_size  # a short page is the last, so shrink to page
    return api


def test_resolves_ambiguous_missing_and_archived_across_projects() -> None:
    ws = sunfish_workspace()
    result = lookup_work_items(
        client_for(ws),
        "ws-1",
        keys=["SF-101", "SF-301", "SF-999", "SF-102", "SF-77", "SF-100"],
    )
    by_key = {r.key: r for r in result.results}
    assert [r.key for r in result.results] == [
        "SF-101",
        "SF-301",
        "SF-999",
        "SF-102",
        "SF-77",
        "SF-100",
    ]
    assert result.coverage.complete

    resolved = by_key["SF-101"]
    assert resolved.outcome == "resolved"
    (only,) = resolved.candidates
    assert only.kind == "task"
    assert only.task is not None
    assert (only.task.id, only.task.name) == ("t-101", "SF-101 Wire up auth")
    assert (only.task.status, only.task.completed) == ("ACTIVE", False)
    assert (only.project.id, only.project.name) == ("p-acme", "Acme Phase 1")
    assert only.project.archived is False
    assert only.client is not None
    assert (only.client.id, only.client.name) == ("c-acme", "Acme Robotics")

    done = by_key["SF-102"].candidates[0].task
    assert done is not None and (done.status, done.completed) == ("DONE", True)

    ambiguous = by_key["SF-301"]
    assert ambiguous.outcome == "ambiguous"
    assert sorted(c.task.id for c in ambiguous.candidates if c.task) == [
        "t-301a",
        "t-301b",
    ]
    assert {c.client.id for c in ambiguous.candidates if c.client} == {
        "c-acme",
        "c-globex",
    }

    assert by_key["SF-999"].outcome == "not_found"
    assert by_key["SF-999"].candidates == []

    legacy = by_key["SF-77"].candidates[0]
    assert by_key["SF-77"].outcome == "resolved"
    assert legacy.project.archived is True
    assert legacy.client is not None and legacy.client.archived is True
    assert legacy.task is not None and legacy.task.completed is True

    project_owned = by_key["SF-100"].candidates[0]
    assert (by_key["SF-100"].outcome, project_owned.kind) == ("resolved", "project")
    assert project_owned.task is None and project_owned.project.id == "p-acme"


def test_matches_only_the_exact_key_never_a_similar_one() -> None:
    result = lookup_work_items(
        client_for(sunfish_workspace(0)),
        "ws-1",
        keys=["sf-101", "SF-10", "SF-555", "SF-1010"],
    )
    by_key = {r.key: r for r in result.results}
    # Lower case is normalised; a prefix of a longer key is not a match; a key
    # mentioned mid-name does not make a task that key's owner.
    assert by_key["SF-101"].outcome == "resolved"
    assert by_key["SF-101"].candidates[0].task.id == "t-101"  # type: ignore[union-attr]
    assert by_key["SF-10"].outcome == "not_found"
    assert by_key["SF-555"].outcome == "not_found"
    assert by_key["SF-1010"].candidates[0].task.id == "t-1010"  # type: ignore[union-attr]


def test_keys_are_validated_before_any_upstream_request() -> None:
    ws = sunfish_workspace(0)
    for keys in (["not a key"], [], [f"SF-{n}" for n in range(1, 52)]):
        with pytest.raises(ValueError):
            lookup_work_items(client_for(ws), "ws-1", keys=keys)
    assert ws.requests == []


def test_duplicate_keys_are_answered_once() -> None:
    result = lookup_work_items(
        client_for(sunfish_workspace(0)), "ws-1", keys=["SF-101", "sf-101", "SF-101"]
    )
    assert [r.key for r in result.results] == ["SF-101"]


def test_client_scope_limits_the_search_and_is_reported() -> None:
    ws = sunfish_workspace()
    result = lookup_work_items(
        client_for(ws), "ws-1", keys=["SF-201", "SF-101"], client_ids=["c-globex"]
    )
    by_key = {r.key: r.outcome for r in result.results}
    assert by_key == {"SF-201": "resolved", "SF-101": "not_found"}
    scope = result.coverage.scope
    assert scope.client_ids == ["c-globex"] and scope.project_ids is None
    assert not scope.workspace_wide
    assert result.coverage.complete
    assert result.coverage.projects_in_scope < result.coverage.projects_total


def test_project_scope_and_both_scopes_intersect() -> None:
    ws = sunfish_workspace()
    only_project = lookup_work_items(
        client_for(ws), "ws-1", keys=["SF-301"], project_ids=["p-globex"]
    )
    (result,) = only_project.results
    assert result.outcome == "resolved"
    assert result.candidates[0].task.id == "t-301b"  # type: ignore[union-attr]

    both = lookup_work_items(
        client_for(ws),
        "ws-1",
        keys=["SF-301"],
        client_ids=["c-acme"],
        project_ids=["p-globex"],
    )
    assert both.results[0].outcome == "not_found"
    assert both.coverage.projects_in_scope == 0


def test_unknown_scope_ids_are_rejected_not_reported_as_not_found() -> None:
    ws = sunfish_workspace(0)
    with pytest.raises(ValueError, match="client"):
        lookup_work_items(
            client_for(ws), "ws-1", keys=["SF-101"], client_ids=["c-nope"]
        )
    with pytest.raises(ValueError, match="project"):
        lookup_work_items(
            client_for(ws), "ws-1", keys=["SF-101"], project_ids=["p-nope"]
        )
    with pytest.raises(ValueError, match="letters, digits"):
        lookup_work_items(client_for(ws), "ws-1", keys=["SF-101"], project_ids=["../x"])


def test_unreadable_project_makes_unfound_keys_incomplete_not_absent() -> None:
    ws = sunfish_workspace(0)
    ws.unstable.add("/workspaces/ws-1/projects/p-globex/tasks")
    result = lookup_work_items(
        client_for(ws, page_size=1),
        "ws-1",
        keys=["SF-101", "SF-301", "SF-999", "SF-201"],
    )
    by_key = {r.key: r for r in result.results}
    assert not result.coverage.complete
    (unread,) = result.coverage.unread
    assert unread.project_id == "p-globex"
    assert unread.failure is not None and unread.failure.reason == "repeated_page"
    # Absent from what was read, but not definitively absent.
    assert by_key["SF-999"].outcome == "incomplete"
    assert by_key["SF-201"].outcome == "incomplete"
    # One candidate seen, uniqueness unproven: evidence kept, not "resolved".
    assert by_key["SF-101"].outcome == "incomplete"
    assert [c.task.id for c in by_key["SF-101"].candidates if c.task] == ["t-101"]
    # Only one SF-301 task is readable (the other project is unread), so
    # that is not yet ambiguity: incomplete, with the one candidate seen.
    assert by_key["SF-301"].outcome == "incomplete"


def test_ambiguity_is_final_even_when_part_of_the_scope_is_unread() -> None:
    ws = sunfish_workspace(0)
    ws.add_task("t-301c", "SF-301 Third copy", "p-acme")
    ws.unstable.add("/workspaces/ws-1/projects/p-globex/tasks")
    result = lookup_work_items(client_for(ws, page_size=1), "ws-1", keys=["SF-301"])
    assert result.results[0].outcome == "ambiguous"
    assert not result.coverage.complete


def test_failed_project_read_is_reported_without_upstream_content() -> None:
    ws = sunfish_workspace(0)
    ws.failing.add("/workspaces/ws-1/projects/p-old/tasks")
    result = lookup_work_items(client_for(ws), "ws-1", keys=["SF-77", "SF-101"])
    by_key = {r.key: r.outcome for r in result.results}
    assert by_key == {"SF-77": "incomplete", "SF-101": "incomplete"}
    (unread,) = result.coverage.unread
    assert unread.project_id == "p-old" and unread.failure is None
    assert unread.error == "GET /workspaces/ws-1/projects/p-old/tasks returned HTTP 503"
    assert "must not leak" not in result.model_dump_json()


def test_primary_list_failure_raises_rather_than_reporting_a_scope() -> None:
    ws = sunfish_workspace(0)
    ws.failing.add("/workspaces/ws-1/projects")
    # The prefix also fails task reads, but projects fail first and are fatal.
    with pytest.raises(ClockifyError, match="503"):
        lookup_work_items(client_for(ws), "ws-1", keys=["SF-101"])


def test_response_volume_and_upstream_requests_are_measured_separately() -> None:
    """Compact output shrinks the answer; only scope shrinks upstream work."""
    ws = sunfish_workspace(60)
    keys = ["SF-101", "SF-301", "SF-999", "SF-102", "SF-77"]

    ws.reset_counters()
    full = workspace_catalogue(client_for(ws), "ws-1")
    full_requests, full_wire = len(ws.requests), ws.response_bytes
    full_response = len(full.model_dump_json())

    ws.reset_counters()
    unscoped = lookup_work_items(client_for(ws), "ws-1", keys=keys)
    unscoped_requests, unscoped_wire = len(ws.requests), ws.response_bytes
    unscoped_response = len(unscoped.model_dump_json())

    ws.reset_counters()
    scoped = lookup_work_items(
        client_for(ws),
        "ws-1",
        keys=["SF-101", "SF-301"],
        project_ids=["p-acme", "p-globex"],
    )
    scoped_requests = len(ws.requests)

    # The caller-visible answer is far smaller...
    assert unscoped_response * 10 < full_response
    # ...but an unscoped lookup reads exactly what the full catalogue reads.
    assert unscoped_requests == full_requests
    assert unscoped.coverage.upstream_requests == unscoped_requests
    # Declaring scope is what removes upstream requests (2 task reads/project).
    assert scoped_requests < full_requests // 4
    assert scoped.coverage.upstream_requests == scoped_requests
    # What crossed the wire is the same too: only the projection shrank.
    assert unscoped_wire == full_wire
    print(
        json.dumps(
            {
                "full": {"requests": full_requests, "response_chars": full_response},
                "lookup_unscoped": {
                    "requests": unscoped_requests,
                    "response_chars": unscoped_response,
                },
                "lookup_scoped": {
                    "requests": scoped_requests,
                    "response_chars": len(scoped.model_dump_json()),
                },
            }
        )
    )


def test_full_catalogue_is_unchanged_by_the_lookup_feature() -> None:
    ws = sunfish_workspace(2)
    full = workspace_catalogue(client_for(ws), "ws-1")
    assert len(full.projects) == 5 and len(full.tasks) == 8 + 8
    assert {t.status for t in full.tasks} == {"ACTIVE", "DONE"}
    assert "results" not in full.model_dump()


def test_same_task_seen_in_both_status_scans_is_not_false_ambiguity() -> None:
    ws = sunfish_workspace(0)
    ws.echoed_tasks.add("t-101")  # identical record in the ACTIVE and DONE scans
    result = lookup_work_items(client_for(ws), "ws-1", keys=["SF-101"])
    (found,) = result.results
    assert found.outcome == "resolved" and len(found.candidates) == 1
    assert result.coverage.complete


def test_contradictory_status_observations_are_incomplete_not_a_silent_pick() -> None:
    ws = sunfish_workspace(0)
    # SF-101 reads ACTIVE in one scan and DONE in the other: its status moved
    # mid-read. Neither is a trustworthy answer, and neither may be chosen.
    ws.flipping_tasks.add("t-101")
    result = lookup_work_items(client_for(ws), "ws-1", keys=["SF-101", "SF-201"])
    by_key = {r.key: r for r in result.results}
    assert by_key["SF-101"].outcome == "incomplete"
    assert by_key["SF-101"].candidates == []
    assert by_key["SF-201"].outcome == "incomplete"  # unproven: p-acme unread
    assert not result.coverage.complete
    (unread,) = result.coverage.unread
    assert unread.project_id == "p-acme"
    assert unread.error is not None and "contradict" in unread.error


def test_contradictory_client_observations_fail_rather_than_pick_one() -> None:
    ws = sunfish_workspace(0)
    ws.flipping_clients.add("c-acme")
    with pytest.raises(ClockifyError, match="contradictory"):
        lookup_work_items(client_for(ws), "ws-1", keys=["SF-101"])


def test_candidates_per_key_are_bounded_and_say_so() -> None:
    ws = sunfish_workspace(0)
    for n in range(14):
        ws.add_task(f"t-400-{n}", f"SF-400 Copy {n}", "p-acme")
    result = lookup_work_items(client_for(ws), "ws-1", keys=["SF-400", "SF-101"])
    many, one = result.results
    assert many.outcome == "ambiguous"
    assert len(many.candidates) == MAX_CANDIDATES_PER_KEY
    assert many.candidate_count == 14 and many.truncated
    assert (one.candidate_count, one.truncated) == (1, False)


def test_a_failing_upstream_stops_after_a_bounded_number_of_unread_projects() -> None:
    ws = sunfish_workspace(0)
    for n in range(MAX_UNREAD_PROJECTS + 15):
        ws.add_project(f"p-down-{n}", f"Down {n}", "c-acme")
    ws.failing.add("/workspaces/ws-1/projects/p-down-")
    result = lookup_work_items(client_for(ws), "ws-1", keys=["SF-101", "SF-999"])
    coverage = result.coverage
    assert len(coverage.unread) == MAX_UNREAD_PROJECTS
    # p-acme reads, 20 down projects fail, then 15 more down plus p-globex
    # and p-old (ID order) are never attempted.
    assert (coverage.projects_read, coverage.unattempted) == (1, 17)
    assert (
        coverage.projects_read + len(coverage.unread) + coverage.unattempted
        == coverage.projects_in_scope
    )
    assert {r.outcome for r in result.results} == {"incomplete"}
    # 4 list scans, 2 task reads for p-acme, then one failed request per unread
    # project and nothing for the unattempted.
    assert coverage.upstream_requests == 4 + 2 + MAX_UNREAD_PROJECTS


def test_project_note_owns_only_its_first_key_and_task_key_must_lead() -> None:
    ws = sunfish_workspace(0)
    ws.add_project("p-multi", "Multi", "c-acme", "Jira: SF-500 and also SF-501")
    ws.add_task("t-ws", " SF-600 leading space", "p-multi")
    result = lookup_work_items(
        client_for(ws), "ws-1", keys=["SF-500", "SF-501", "SF-600"]
    )
    by_key = {r.key: r for r in result.results}
    assert by_key["SF-500"].outcome == "resolved"
    assert by_key["SF-500"].candidates[0].project.id == "p-multi"
    assert by_key["SF-501"].outcome == "not_found"
    assert by_key["SF-600"].outcome == "not_found"
