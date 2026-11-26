# Messaging Channels (`sofuu.channels`)

Sofuu agents can reach people where they already are — Telegram, WhatsApp,
Slack/Discord/ntfy-style webhooks, or the desktop — and accept commands back
without every agent re-implementing HTTP clients, polling loops, signature
verification, credential storage and retry queues.

The channel layer is **domain-agnostic by governance** (`AGENTS.md`,
`RULES.md`): an adapter transports the text it is handed. No adapter contains a
market ticker, a security prompt, a support schema or a business endpoint. The
runtime ships four transports: `telegram`, `whatsapp`, `webhook`, `desktop`.

```js
// One line to notify through the active channel:
await sofuu.channels.send({
  title: 'Build finished',
  text: 'main @ 4f2a1c is green.',
  severity: 'info',
});

// Two-way: verified inbound messages, per-sender sessions, no transport code.
const sub = sofuu.channels.listen({
  agent: 'ops_assistant',          // runs with a tagged external-message prompt
  permissionProfile: 'plan',      // remote senders get no writes/shell by default
});
```

---

## 1. Configuration (`channels.json`)

The runtime resolves a per-agent channel file with the same discovery order as
`models.json`:

1. `--channels-config <path>` or `SOFUU_CHANNELS_CONFIG`
2. `<cwd>/<binary-stem>.channels.json`, then `<binary-dir>/<binary-stem>.channels.json`
   (for compiled binaries)
3. For a script, `<script-stem>.channels.json`, `<script-dir>/channels.json`,
   then `<cwd>/channels.json` (`index.js` skips the stem form)

Schema:

```json
{
  "active": "team-telegram",
  "channels": {
    "team-telegram": {
      "type": "telegram",
      "token": "env:TELEGRAM_BOT_TOKEN",
      "default_recipient": "123456789",
      "allowed_senders": ["123456789"],
      "network_mode": "outbound_polling"
    },
    "team-whatsapp": {
      "type": "whatsapp",
      "phone_number_id": "109876543210",
      "access_token": "env:WHATSAPP_ACCESS_TOKEN",
      "app_secret": "env:WHATSAPP_APP_SECRET",
      "verify_token": "env:WHATSAPP_VERIFY_TOKEN",
      "default_recipient": "+9198XXXXXXXX",
      "allowed_senders": ["+9198XXXXXXXX"],
      "network_mode": "gateway_webhook"
    },
    "team-webhook": {
      "type": "webhook",
      "url": "env:ALERT_WEBHOOK_URL",
      "format": "slack",
      "network_mode": "outbound_only"
    },
    "desktop": {
      "type": "desktop",
      "network_mode": "local"
    }
  }
}
```

Rules that hold for every adapter:

- `"env:VAR_NAME"` is resolved at load time for every secret field. A literal
  secret is accepted but written with `0600` permissions by the setup wizard and
  masked by `--list-channels`.
- `allowed_senders` is **fail-closed**. An empty list drops every inbound
  message unless the profile sets `allow_anonymous: true` or policy sets
  `channels.inbound.allow_anonymous: true`; policy
  `channels.inbound.require_allowlist: true` wins over either anonymous flag.
- `network_mode` is descriptive (`outbound_polling`, `outbound_only`,
  `gateway_webhook`, `local`); the enforced boundary is the bind rule in §4.

### CLI

```bash
sofuu run agent.js --setup-channel       # guided wizard; --type enables headless setup
sofuu run agent.js --edit-channel        # switch / edit / add / delete
sofuu run agent.js --list-channels       # masked table + network mode + drop warning
sofuu run agent.js --channel team-whatsapp   # per-run active-profile override
sofuu run agent.js --channels-config ./channels.json

# Same flags on a compiled standalone binary:
./my-agent --setup-channel --type telegram --channel-name alerts \
  --token env:TELEGRAM_BOT_TOKEN --default-recipient 123456789 \
  --allowed-senders 123456789
```

---

## 2. Outbound sends

```ts
await sofuu.channels.send({
  channel?: string,       // profile name; defaults to the active profile
  to?: string,            // override default_recipient
  title?: string,
  text: string,
  severity?: 'info' | 'warn' | 'critical',
  actions?: Array<{ id: string; label: string }>,
  dryRun?: boolean,       // validate the profile without a network call
});
```

| Adapter | Outbound behaviour |
| :--- | :--- |
| `telegram` | `POST /bot{token}/sendMessage`; splits long text at the platform limit; `actions` become an inline keyboard; `callback_data` carries each action id. |
| `whatsapp` | `POST /v21.0/{phone_number_id}/messages` with `Bearer`; up to three `actions` become interactive reply buttons, more become a numbered text list. |
| `webhook` | JSON, Slack (`{text}`), Discord (`{content}`), ntfy headers, or plain text; retries `429`/`5xx` with capped exponential backoff. |
| `desktop` | `osascript` on macOS, `notify-send` on Linux, PowerShell toast on Windows. Arguments are passed as an argv array, never through a shell. |

Send errors are `SofuuError` with `E_CHANNEL_CONFIG`, `E_CHANNEL_AUTH`,
`E_CHANNEL_NETWORK` or `E_POLICY`; `err.status`, `err.retryable` and
`err.retryAfterMs` are populated where the platform supplies them.

---

## 3. Inbound (`sofuu.channels.listen`)

```ts
const sub = sofuu.channels.listen({
  channel?: string | string[],   // omitted = every configured inbound profile
  onMessage?: async (msg) => {
    const answer = await sofuu.agent.run('ops_assistant', msg.prompt, {
      sessionKey: `channel:${msg.channel}:${msg.senderId}`,
      permissionProfile: 'plan',
    });
    await msg.reply(answer.answer);   // same profile + verified sender
  },
  onAction?: async (action) => {
    // `action.data` is the Telegram callback payload / WhatsApp reply id.
    // Async HITL: continue a parked run from a button press.
    await sofuu.agent.resume(action.data, { allow: true });
  },
  actionAck?: 'Working…',       // Telegram answerCallbackQuery text
  pollTimeout?: 25,             // Telegram long-poll seconds
});

sub.close();
```

`listen({ agent: 'name' })` is the zero-boilerplate form: the driver runs the
agent with `msg.prompt`, a per-sender `sessionKey`, a bounded per-sender history
in `sofuu.state`, a default `plan` permission profile, and replies with
`result.answer`. `msg.prompt` wraps the incoming text in
`<<<EXTERNAL_MESSAGE>>> … <<<END_EXTERNAL_MESSAGE>>>` and states that the content
is untrusted, so an inbound message cannot silently rewrite the agent's role,
tool permissions or instructions.

Inbound message shape:

```ts
interface InboundMessage {
  channel: string;
  type: 'telegram' | 'whatsapp' | 'webhook';
  senderId: string;        // Telegram chat id / WhatsApp phone (E.164 normalised)
  senderName: string;
  text: string;
  prompt: string;          // tagged, safe to pass as the agent task
  messageId: string;
  timestamp: number;
  raw: unknown;
  reply(text: string, opts?): Promise<ChannelSendResult>;
  run(agent: string, opts?): Promise<unknown>;
}
```

### Telegram

- Default inbound is **outbound long polling** (`getUpdates`) with the offset
  persisted in `sofuu.state` under `telegram:offset:<profile>`; no public port,
  no webhook.
- `callback_query` events answer the callback and route `data` to `onAction`.
- Optional webhook mode: `POST /webhooks/telegram` verifies
  `X-Telegram-Bot-Api-Secret-Token` against `verify_token` (or `app_secret`)
  before dispatching.

### WhatsApp Cloud API

- `GET /webhooks/whatsapp` validates `hub.mode=subscribe` and constant-time
  matches `hub.verify_token`, then returns `hub.challenge`.
- `POST /webhooks/whatsapp` caps the body, verifies
  `X-Hub-Signature-256` over the raw body with `app_secret`, and drops duplicate
  `message.id` retries before dispatch. It fails closed with `503` when no
  secret is configured.
- Include `?channel=<profile>` when more than one WhatsApp profile shares a
  gateway; otherwise the payload's `metadata.phone_number_id` selects the
  profile.

### Generic webhook

`POST /webhooks/<profile>` (or `?channel=<profile>`) accepts
`{ senderId, senderName?, text, messageId?, timestamp? }` and verifies either
`X-Sofuu-Signature: sha256=<hmac>` with `app_secret`, or
`Authorization: Bearer <verify_token>` / `X-Sofuu-Token`. It is fail-closed
when the profile has neither secret.

All inbound paths cap the body at 64 KiB, apply a per-sender rate limit
(`channels.inbound.max_msgs_per_minute`, default 20) and log dropped senders to
stderr.

---

## 4. Local vs. public network boundary

```text
UNTRUSTED PUBLIC EDGE
  Meta / Telegram / external webhook sender
        │ HTTPS POST + signature
        ▼
  cloudflared / Tailscale Funnel / nginx TLS
        │ forward to loopback
        ▼
  sofuu gateway  127.0.0.1:8788
    1. body cap
    2. constant-time HMAC / bearer / verify-token check
    3. sender allowlist + rate limit
    4. dispatch to sofuu.channels.listen subscribers
        ▼
  PRIVATE PLANE  agent runtime (fs, shell, MCP, memory)
```

```bash
sofuu gateway                       # 127.0.0.1:8788
sofuu gateway --port 9000 --channels-config ./channels.json
sofuu gateway --policy ./policy.json
```

A non-loopback `--host` (or `startGateway({host})`) throws `E_POLICY` unless
policy opts in:

```json
{
  "channels": {
    "allow": ["telegram", "whatsapp", "desktop"],
    "inbound": {
      "require_allowlist": true,
      "allow_anonymous": false,
      "bind": "127.0.0.1",
      "allow_public_bind": false,
      "max_msgs_per_minute": 20
    }
  }
}
```

`channels.allow`, when present, is a transport allowlist checked before any
adapter runs (including outbound sends). `--policy` is applied before the
gateway starts; without it, the layered `~/.sofuu/policy.json` and
`./.sofuu/policy.json` are loaded.

The same routes are available on an existing edge server: `sofuu serve` hands
`/webhooks/*` to `sofuu.channels.handleWebhook` before its bearer-token gate,
because providers authenticate with their own HMAC/verify token and cannot
present the serve bearer. A path the channel driver does not own falls through
to the normal authenticated surface and its 404. Hosts can embed the handler
directly:

```js
const handler = sofuu.channels.createWebhookHandler();
server.on('request', (req, res) => handler(req, res, req.url));
```

---

## 5. Security defaults

| Control | Behaviour |
| :--- | :--- |
| Sender allowlist | Empty `allowed_senders` drops inbound unless anonymous access is explicitly enabled. |
| Signature checks | HMAC-SHA256 and bearer/verify tokens are compared with `sofuu.crypto.timingSafeEqual` (constant-time in Rust, local fallback in JS). |
| Fail-closed config | Missing WhatsApp `app_secret` or Telegram webhook secret answers `503`; missing signatures answer `401`. |
| Body cap | Inbound bodies above 64 KiB answer `413` before parsing. |
| Replay protection | WhatsApp `message.id` and generic `messageId` are deduplicated for seven days. |
| Prompt injection | Inbound text is tagged as untrusted before it reaches an agent; remote sessions default to `plan` permissions. |
| Bind rule | Loopback by default; public binds require explicit policy opt-in. |
| Reconnect | Telegram polling backs off exponentially and resumes from the persisted offset. |

---

## 6. Tests

`tests/channels_test.js` runs on loopback fixtures (Telegram Bot API, WhatsApp
Graph API, a flaky webhook) and covers outbound sends and splitting, long-poll
offset persistence, allowlist drops, callback routing, WhatsApp verification and
HMAC rejection, duplicate suppression, webhook retries, prompt tagging,
per-sender agent routing, the loopback bind rule and a full Async HITL flow
where a Telegram button resumes a parked agent run. It is registered with
`tests/run_js_tests.sh`; it needs no network and no API keys.

```bash
./sofuu run tests/channels_test.js
```
