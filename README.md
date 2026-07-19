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
