# Database — Supabase / Postgres

Persistence layer for the ML Training Playground. Replaces the flat
`backend/leaderboard.json` file once the v1 engine lands.

- **`schema.sql`** — full DDL: tables, indexes, triggers, leaderboard views, RLS policies, and seed
  rows for the dataset/algorithm catalogues.
- Scoring rules that the `runs` columns encode are specified in **[`../docs/SCORING.md`](../docs/SCORING.md)**.

## What gets stored

| Table | Holds |
|-------|-------|
| `students` | One row per player. Anonymous arcade tag, or linked to Supabase Auth via `auth_user_id`. |
| `datasets` | Dataset catalogue + the split contract (`test_size`, `split_seed`) every score depends on. |
| `algorithms` | Algorithm catalogue + `hyperparameter_spec` JSONB — the single source of truth the frontend renders sliders from and the backend clamps against. |
| `runs` | **The core table.** Which dataset, which algorithm, the exact tuned `hyperparams`, every metric, the score, and the overfit verdict. |
| `run_hyperparams` | Normalized mirror of `runs.hyperparams`, kept in sync by trigger. Makes "which `max_depth` do students pick, and does it help?" a plain `GROUP BY`. |
| `run_snapshots` | Per-step training trace, so a run can be replayed on the results page without retraining. |
| `roast_templates` | The overfit comment lines, editable in the DB without a redeploy. |

Leaderboard positions are **not stored** — they are computed by the
`leaderboard_by_algorithm`, `leaderboard_by_dataset`, and `leaderboard_global` views. Storing a rank
would go stale the moment anyone else submits; a `rank()` window over the best run per student is
always correct and is fast against the partial indexes on `runs`.

Each view takes each student's **best** run (not all of them), so one player cannot flood the top ten.

## Applying it

**Supabase SQL Editor:** paste `schema.sql`, Run. It is idempotent — safe to re-run.

**psql:**

```bash
psql "$DATABASE_URL" -f db/schema.sql
```

**Supabase CLI:**

```bash
supabase db push --db-url "$DATABASE_URL"
```

## Backend configuration (not yet wired)

The backend will read these from the environment — never commit them:

```
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role key>   # server-side only
# or, for direct psql access:
DATABASE_URL=postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres
```

## Security model

RLS is enabled on every table.

- **Reads are public** on all tables — a leaderboard nobody can see is not a leaderboard.
- **No insert/update/delete policy is granted to `anon` or `authenticated`.** All writes go through
  the FastAPI backend using the **service-role key**, which bypasses RLS. This is deliberate: if the
  browser could insert into `runs` directly, a student could POST a fabricated 1000-point run without
  training anything. The score must only ever be computed server-side, from a model the server fitted.

Consequently the service-role key must live on the server only — never in `frontend/app.js`.

## Status

⚠️ **This schema has not been applied to a live database yet, and has not been executed against a
Postgres instance.** There is no Supabase connection available from the development session (no
Supabase MCP integration, no `supabase` CLI, no `psql`, no credentials in the environment), and no
local Postgres to dry-run the DDL against. Treat the first `Run` in the Supabase SQL Editor as the
validation step, and report any syntax error back.
