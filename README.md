# MemMachine

This repository is an independent memory layer for AI agents, based on the
original [MemMachine](https://github.com/MemMachine/MemMachine) project
(Apache 2.0). It is not a drop-in replacement for upstream MemMachine, and it
does not track or contribute back to that project.

The design is narrower and more operational:

- **Episodic memory stores claims**, not full chat transcripts.
- **Semantic memory is organized by our category and tag schema.**
- **Ingestion and consolidation are safe to run across multiple pods.**

## What is different

### Claims as episodic memory

Upstream MemMachine treats conversational episodes (messages, sessions) as the
unit of episodic memory. This implementation stores **claims**: discrete,
self-contained statements about a user, an agent, or a setting.

A claim is the input to extraction. Semantic prompts expect a stated fact or
preference, not a multi-turn conversation. Chat history, if you have it, should
be reduced to claims before it is written here.

### Category and tag schema

Semantic features are not a flat bag of facts. Each feature belongs to a
**category** (extraction domain) and a **tag** (allowed facet inside that
domain).

```
claim → category → tag → feature_name + value
```

Built-in categories include:

| Category | Purpose | Example tags |
| --- | --- | --- |
| `agent_personality` | Stable agent behavior settings | `tone`, `persona`, `style`, `boundaries` |
| `life_context` | Long-lived personal context | `interests`, `lifestyle`, `goals`, `personality`, `life_situation`, `general_preference` |
| `task_assistant` | Structured facts for task completion | `basics`, `contacts`, `identities`, `accounts`, `preferences`, `relationships`, `services` |

Tags are defined per category. Extractors must pick an existing tag rather than
inventing new ones. Manual writes (for example agent personality) also validate
against the allowed tag set.

### Multi-pod deployment

The server is meant to run as several replicas behind a load balancer, sharing
PostgreSQL (and Neo4j when used). Duplicate work is avoided in storage, not in
process-local queues:

- Uningested claims are **claimed atomically** (`SELECT FOR UPDATE SKIP LOCKED`
  on PostgreSQL; an equivalent ingest mark on Neo4j) so only one pod processes
  a given history row.
- Consolidation takes a **per-`set_id` lock** with expiry and cleanup, so two
  pods do not consolidate the same set at once.

Horizontal scale is therefore a deployment choice: add pods, point them at the
same databases and config.

## Architecture

1. Clients write **claims** through the REST API, Python SDK, or MCP.
2. Claims are stored as episodic history.
3. Background ingestion on any pod extracts semantic features into the
   configured **category / tag** schema.
4. Search returns episodic claims and/or consolidated semantic features.

MemMachine is **not a hosted service**. You run the server yourself.

## Quick start

Docker is the usual path:

```bash
./memmachine-compose.sh
```

The script checks Docker, creates `.env` if needed, writes `cfg.yml` (or
`configuration.yml`), and starts MemMachine plus PostgreSQL and Neo4j.

```bash
./memmachine-compose.sh stop
./memmachine-compose.sh restart
./memmachine-compose.sh logs
./memmachine-compose.sh clean   # removes data and volumes
```

Python install:

```bash
pip install memmachine
# or separately:
pip install memmachine-client
pip install memmachine-server
```

Run the server:

```bash
memmachine-server --config cfg.yml
```

Default API base: `http://localhost:8080/api/v2`.

## Usage sketch

```python
from memmachine import MemMachineClient

client = MemMachineClient(base_url="http://localhost:8080")
project = client.create_project(org_id="my-org", project_id="my-project")

memory = project.memory(user_id="user123", agent_id="agent456")

# Write a claim, not a chat turn
memory.add(
    content="Notify about emails from Peter",
    role="user",
    metadata={"type": "claim"},
)

results = memory.search(query="notification scope", limit=10)
```

v2 APIs are scoped by `org_id` and `project_id`. See `AGENTS.md` for endpoint
lists and more examples.

## Configuration

YAML (`cfg.yml` / `configuration.yml`) covers:

- `resources.databases` — PostgreSQL (semantic) and Neo4j (episodic, if used)
- `resources.embedders` / `resources.language_models`
- `episodic_memory` / `semantic_memory` — including which semantic categories
  are enabled for a project

## License

Apache 2.0. See [LICENSE](LICENSE). This codebase includes work derived from
the original MemMachine project.
