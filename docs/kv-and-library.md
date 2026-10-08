# How to store data in MemMachine

The backend forwards requests. MemMachine is the store. A project has four kinds of data. Pick the one that matches what you are saving. `org_id` and `project_id` come from the logged-in project. `role_id` is set by the caller and decides which bucket the data lands in.

Deleting a project removes all four: episodes, semantic features, key-value rows, and library files.

| What you are saving | Where it goes | `role_id` | Shows up in memory search |
|---|---|---|---|
| Facts about the user, from conversation | Normal memory | omit it | yes |
| How this agent should behave | Agent personality | the agent's id | no |
| One call, kept as written | KV | `call_log` | no |
| A document the user filed | Library | `library` | no |

## 1. Normal memory

Use this for what the user said about themselves. Do not send `role_id`.

`POST /api/v2/memories` writes the message into episodic memory and queues semantic extraction. Extraction uses the user-profile prompts. The semantic set is `mem_user_<project_id>`. If the message also has `session_id`, a session set is updated too.

`POST /api/v2/memories/search` reads that user profile and episodic memory. It does not read agent personality, KV, or Library.

```python
memory.add(
    content="I prefer morning visits.",
    role="user",
)
memory.search(query="When does the user like to be visited?")
```

```bash
curl -X POST "http://localhost:8080/api/v2/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project",
    "messages": [
      {"content": "I prefer morning visits.", "role": "user"}
    ]
  }'
```

## 2. Agent personality

Use this for how the agent should act. Put the agent's id in `metadata.role_id`. The same id is reused for that agent inside this project.

When `role_id` is present, semantic extraction uses the role prompts (`agent_personality`) and does not write the user profile. The set id is `mem_role_<org_id>/<project_id>/<role_id>`. Memory search does not return this set. Read it with `POST /api/v2/memories/list` and the same `role_id`.

A settings screen can also append one instruction directly with `POST /api/v2/memories/semantic`, `isolation` set to `role`, and the same `role_id`. That path does not create an episode.

```python
memory.add(
    content="Keep answers short and confirm the appointment time before ending the call.",
    role="user",
    metadata={"role_id": "front-desk"},
)
```

```bash
curl -X POST "http://localhost:8080/api/v2/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "my-org",
    "project_id": "my-project",
    "messages": [
      {
        "content": "Keep answers short and confirm the appointment time before ending the call.",
        "role": "user",
        "metadata": {"role_id": "front-desk"}
      }
    ]
  }'
```

## 3. KV

Use this for a call record that must stay out of the user profile. `role_id` is `call_log`. One phone number is one key. Each finished call appends a new value. Values are not updated and single rows are not deleted.

Lookup is an exact key match, newest first. `limit` defaults to 5. Send `null` to return every value for that key. One value is at most 8000 characters. An unknown key returns an empty list.

```python
memory.append_kv(
    key="+14155550142",
    value="2026-09-07, caller said they were from Dr Smith's office and confirmed Tuesday at 10.",
    role_id="call_log",
)
memory.get_kv(key="+14155550142", role_id="call_log", limit=5)
```

| Operation | Path | Success |
|---|---|---|
| Append | `POST /api/v2/memories/kv` | 201 |
| Read | `POST /api/v2/memories/kv/get` | 200 |

Write the call here. `POST /api/v2/memories` would run extraction and can put the caller's phone and name into the user profile.

## 4. Library — backend

MemMachine stores the file. The backend owns the file page and all writes. Scope is `org_id` + `project_id` + `role_id` (`library`).

Library is **not** memory search and **not** semantic extraction. Do not send these documents through `POST /api/v2/memories`.

There is no `always_loaded`. Do not send it.

### Writes (backend only)

| | Path | Body | Success | Failure |
|---|---|---|---|---|
| Create | `POST /api/v2/memories/library` | `org_id`, `project_id`, `role_id`, `name`, `content`, `description`, `category` | **201**, full file + `id` | 422 invalid or category full; **409** title taken |
| Update | `POST /api/v2/memories/library/update` | `org_id`, `project_id`, `role_id`, `id`, `name`, `content`, `description` | **200**, same `id` and `category` | **404** unknown id; 422 invalid; **409** title taken |
| Delete | `POST /api/v2/memories/library/delete` | `org_id`, `project_id`, `role_id`, `id` | **204** | missing id is still 204 |

Update overwrites title, body, and summary in one call. Send all three every time. Category cannot be changed after create.

Removed (do not call): `/library/content`, `/library/rename`, `/library/category`, `/library/description`, `/library/always_loaded`.

### Fields

| Field | Rules |
|---|---|
| `id` | UUID from create. Use it for update, get, delete. |
| `name` | Display title. Unique in this project + role. Max 256. Trimmed. MemMachine does not add suffixes. |
| `content` | Markdown stored as plain text. Required, max 40000. |
| `description` | One line, required, max 512, no newlines. |
| `category` | `personal` or `business`, **create only**. 20 files per category per project + role. |

On 409, ask the user for a different title. Nothing was written.

### Reads

| | Path | Returns |
|---|---|---|
| List | `POST /api/v2/memories/library/list` | `files`: `id`, `name`, `description`, `category`, `updated_at`. No body. Empty is `[]`. |
| Get | `POST /api/v2/memories/library/get` | `id`, `name`, `content`, `description`, `category`, `created_at`, `updated_at`. Unknown id: **404**. |

### Agent

Agent **does not write**. Read path is unchanged: list once per turn (title, category, summary); get by `id` only when the body is needed. If a client typed `always_loaded` as required, drop that field.
