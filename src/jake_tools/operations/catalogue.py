"""Every operation, in the order adapters register them."""

from __future__ import annotations

from . import (
    Operation,
    claude_cancel,
    claude_list,
    claude_poll,
    claude_start,
    claude_status,
    claude_steer,
    claude_transcript,
    claude_update,
    clockify_create_time_entry,
    clockify_jira_sync,
    clockify_time_entries,
    clockify_workspace_catalogue,
    polish_transcript,
    transcribe,
)

OPERATIONS: tuple[Operation, ...] = (
    clockify_jira_sync.OPERATION,
    clockify_time_entries.OPERATION,
    clockify_workspace_catalogue.OPERATION,
    clockify_create_time_entry.OPERATION,
    claude_start.OPERATION,
    claude_status.OPERATION,
    claude_poll.OPERATION,
    claude_list.OPERATION,
    claude_transcript.OPERATION,
    claude_steer.OPERATION,
    claude_update.OPERATION,
    claude_cancel.OPERATION,
    # Sends transcript text to Claude, so it is a `claude` capability.
    polish_transcript.OPERATION,
    # Unavailable unless transcription.api_base_url is configured.
    transcribe.OPERATION,
)
