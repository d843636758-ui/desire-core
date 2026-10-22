# desire-core

Desire Core exposes the intention engine over HTTP and MCP, synchronizes the
configured Eventide state, and can optionally observe Phosphene tasks.

## Phosphene bridge

Configure the bridge with these environment variables:

```env
PHOSPHENE_ENABLED=true
PHOSPHENE_MCP_URL=https://your-phosphene.example/mcp
PHOSPHENE_MCP_TOKEN=phosphene_ai_your_token
PHOSPHENE_MODE=shadow
PHOSPHENE_POLL_SECONDS=900
```

`PHOSPHENE_MCP_TOKEN` accepts either the raw AI token or a value prefixed with
`Bearer `. Keep this token private.

- `shadow` caches task attention for inspection but does not change Desire.
- `in_session` also ingests visible pending/submitted tasks into Desire. When
  `DESIRE_DRIVEN=true`, the MCP state may mark them as eligible for a reminder
  during a conversation started by the user.

The public `/state` route exposes only bridge health and queue counts. Task
titles, descriptions, and the Phosphene overview are returned only through the
authenticated Desire MCP interface.

## Private deployment

The included `Dockerfile` runs the HTTP dashboard and MCP server. Mount a
persistent volume at `/data`; the only mutable state file is
`/data/desire_state.json`. Copy that file without transforming it when moving
between hosts, and keep the old service online until the new state and MCP
connection have been verified.

Production environment variables:

| Variable | Purpose |
|---|---|
| `DESIRE_PUBLIC_HOST` | Public MCP host, defaults to `desire.xunzhou.space` |
| `EVENTIDE_MCP_URL` | Full private Eventide MCP URL, including its path token |
| `EVENTIDE_API_TOKEN` | Optional bearer token for Eventide MCP |
| `EVENTIDE_TOOL_NAME` | Eventide state tool, defaults to `get_full_state` |
| `PHOSPHENE_ENABLED` | Enable the Phosphene bridge |
| `PHOSPHENE_MCP_URL` | Full Phosphene MCP URL |
| `PHOSPHENE_MCP_TOKEN` | Phosphene bearer token |
| `PHOSPHENE_MODE` | `shadow` or `in_session` |
| `DATA_DIR` | Persistent state directory, defaults to `/data` |
| `PORT` | HTTP port, defaults to `8000` |

During migration, point `EVENTIDE_MCP_URL` at the Alibaba Cloud Eventide
service only after its restored state has passed health and MCP checks.
