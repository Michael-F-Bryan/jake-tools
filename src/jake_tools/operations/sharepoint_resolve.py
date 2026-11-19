"""sharepoint_resolve: a SharePoint link, or a Teams channel, to a drive item."""

from __future__ import annotations

from ..config import Config
from ..file_transfers import ResolveResult, resolve_channel, resolve_link
from ..sharepoint_files import DriveClient
from . import Handler, Operation
from ._files import ChannelId, TeamId, call_files
from .errors import ToolError

DESCRIPTION = (
    "Read-only. Turn either a SharePoint link (`url`) or a Teams location "
    "(`team_id` + `channel_id`) into the drive item it points at: `drive_id`, "
    "`item_id`, name, kind (file/folder), web URL, size and version (`etag`, `ctag`, "
    "modified). Use those IDs with the other sharepoint_* tools; links and names "
    "are never accepted anywhere else. A link must be on a SharePoint host the "
    "operator configured (files.sharepoint_hosts); another host is refused. A "
    "channel resolves to its files folder (a private or shared channel has its own "
    "site, which the account must be able to open). Uses the server's configured "
    "Microsoft account; never another."
)


def build(config: Config) -> Handler:
    async def sharepoint_resolve(
        url: str | None = None,
        team_id: TeamId | None = None,
        channel_id: ChannelId | None = None,
    ) -> ResolveResult:
        by_channel = team_id is not None or channel_id is not None
        if (url is not None) == by_channel or (
            by_channel and (team_id is None or channel_id is None)
        ):
            raise ToolError(
                "invalid_argument",
                "supply either url, or both team_id and channel_id",
            )
        hosts = config.files_sharepoint_hosts.value

        def run(client: DriveClient) -> ResolveResult:
            if url is not None:
                return resolve_link(client, url, hosts=hosts)
            assert team_id is not None and channel_id is not None
            return resolve_channel(client, team_id=team_id, channel_id=channel_id)

        return await call_files(config, run)

    return sharepoint_resolve


OPERATION = Operation(
    name="sharepoint_resolve",
    skill="sharepoint-files",
    description=DESCRIPTION,
    build=build,
    omit_none=True,
)
