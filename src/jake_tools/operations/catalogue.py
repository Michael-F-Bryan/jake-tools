"""Every operation, in the order adapters register them."""

from __future__ import annotations

from . import (
    Operation,
    artifact_publish,
    artifact_read,
    artifact_update,
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
    newsletter_create,
    newsletter_get,
    newsletter_list,
    newsletter_update,
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
    # Read-only; always listed, and `missing_credentials` until Microsoft is configured.
    newsletter_list.OPERATION,
    newsletter_get.OPERATION,
    # Publishes to, or changes, a list its readers can see.
    newsletter_create.OPERATION,
    newsletter_update.OPERATION,
    # Sends transcript text to Claude, so it is in the `claude` capability.
    polish_transcript.OPERATION,
    # Unavailable unless transcription.api_base_url is configured.
    transcribe.OPERATION,
    # Unavailable unless artifacts.enabled: an undocumented Claude Code mechanism.
    artifact_publish.OPERATION,
    artifact_read.OPERATION,
    # Overwrites an existing artifact's live content.
    artifact_update.OPERATION,
)
