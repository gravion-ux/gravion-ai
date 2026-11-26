# Drop-in bot slot

Gravion can sit beside a bot the operator already owns. It does not
recreate, rewrite or execute that bot. The bot stays as-is; Gravion
only needs a small HTTP bridge to it.

## Slot layout

```text
<deployment-home>/bots/
  bot.json
  ... the operator's bot files, unchanged ...
```

Example `bot.json`:

```json
{
  "name": "My existing bot",
  "kind": "http",
  "endpoint_url": "http://127.0.0.1:9000/reply",
  "health_path": "/health",
  "timeout_seconds": 60,
  "description": "Operator-owned bot running beside Gravion"
}
```

Check the slot without running bot code:

```bash
gravion bot status
gravion bot check
```

## HTTP contract

Gravion posts:

```json
{
  "messages": [{"role": "user", "content": "hello"}],
  "files": [{"name": "notes.md", "text_preview": "..."}]
}
```

The bot endpoint returns either:

```json
{"reply": "answer text"}
```

or a chat-completions-shaped response with
`choices[0].message.content`.

## Input needed from the bot owner

To connect an existing self-chat bot unchanged, Gravion needs:

1. The bot code, link, or export — as-is.
2. How it runs today: language/runtime and the exact start command
   or endpoint.
3. Its input/output format: what a message looks like in, and what
   a reply looks like out.
4. Names of any environment variables it needs. Secret values stay
   in the operator's environment, never in chat or in `bot.json`.
5. Whether it can expose a local HTTP endpoint, or only runs inside
   a chat app. If it only runs inside a chat app, the owner must
   provide the bridge/export route; Gravion will not invent one.
