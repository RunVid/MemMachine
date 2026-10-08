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

## 4. Library

Use this for a document the user files. `role_id` is `library`. MemMachine assigns an `id` at create time. The title is a display name and is unique for that project and role. Read, replace the body, rename, and delete all use the `id`.

The body is stored as written, including Markdown. It is plain text. The maximum length is 40000 characters. It is not extracted and it is not searchable.

**Create requires `name`, `content`, `description`, and `category`.** Send the title, body, one-line summary, and category. `category` must be `personal` or `business`. The body may be Markdown pasted as plain text (stored as written). Titles are unique for that project and role. If the same title already exists, create returns **409** and nothing is written. MemMachine does not rename files or add numeric suffixes. One project and role may hold at most **50** files; a 51st create returns **422**.

**`description`** is a single line, at most 512 characters. **`category`** is stored as written at create and is not changed by rename or replace-body.

```python
created = memory.create_library(
    content="# Service area\n\nWeekday coverage for the office and the lobby.",
    name="Service area",
    description="Weekday lobby and office coverage",
    category="business",
    role_id="library",
)
memory.update_library_content(
    file_id=created.id,
    content="Updated body",
    role_id="library",
)
memory.rename_library(
    file_id=created.id,
    name="Business hours",
    role_id="library",
)
memory.get_library(file_id=created.id, role_id="library")
memory.delete_library(file_id=created.id, role_id="library")
memory.list_library(role_id="library")
```

| Operation | Path | Success | Failure |
|---|---|---|---|
| Create | `POST /api/v2/memories/library` | 201, returns `id`, title, body, description, and category | Missing or invalid fields (including `category` not `personal`/`business`), or 50 files already in this scope: 422. Title already exists: 409 |
| Replace body | `POST /api/v2/memories/library/content` | 200, title unchanged | Unknown `id`: 404, no file is created |
| Rename | `POST /api/v2/memories/library/rename` | 200, `id` unchanged | Unknown `id`: 404. Title already exists: 409, previous title kept |
| Delete | `POST /api/v2/memories/library/delete` | 204 | A missing `id` still returns 204 |
| Read | `POST /api/v2/memories/library/get` | 200, returns `name`, `content`, `description`, and `category` | Unknown `id`: 404 |
| List | `POST /api/v2/memories/library/list` | 200, each file has `id`, `name`, `description`, `category`, `updated_at` | No files: `files` is `[]` |

List entries are `id`, `name`, `description`, `category`, and `updated_at`. They do not include the body.

The file page collects title, Markdown body, one-line summary, and category (`personal` or `business`), then creates with those fields. On 409, prompt for a different title. Replace, rename, and delete use the returned `id`. Facts about the user still go through normal memory. The assistant can call list once per turn to show each file's title, category, and summary in tool text, then call get with `id` only when it needs the Markdown body.
