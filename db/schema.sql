-- =====================================================================
-- ARIES ML Training Playground — Postgres / Supabase schema
--
-- Stores every student submission: which dataset, which algorithm, which
-- hyperparameters they tuned and to what values, the resulting scores,
-- the overfit verdict, and the leaderboard position derived from them.
--
-- Apply with:  psql "$DATABASE_URL" -f db/schema.sql
--         or:  paste into the Supabase SQL Editor and Run.
--
-- Idempotent: safe to re-run.
-- =====================================================================

create extension if not exists "pgcrypto";   -- gen_random_uuid()

-- ---------------------------------------------------------------------
-- Enums
-- ---------------------------------------------------------------------

do $$ begin
  create type overfit_verdict as enum (
    'underfit', 'clean', 'slight_overfit', 'overfit', 'hard_overfit'
  );
exception when duplicate_object then null; end $$;

do $$ begin
  create type algo_family as enum (
    'tree', 'bagging', 'boosting', 'distance', 'linear', 'neural'
  );
exception when duplicate_object then null; end $$;

do $$ begin
  create type run_status as enum ('running', 'complete', 'failed');
exception when duplicate_object then null; end $$;


-- ---------------------------------------------------------------------
-- students — one row per player
-- ---------------------------------------------------------------------

create table if not exists students (
  id            uuid primary key default gen_random_uuid(),
  handle        text not null unique
                  check (handle ~ '^[A-Za-z0-9_]{2,24}$'),
  display_name  text,
  initials      char(3),                    -- arcade-style tag shown on the board
  -- Set when the player signs in through Supabase Auth; null for anonymous
  -- arcade players. Left as a bare uuid (not a FK to auth.users) so this file
  -- also applies cleanly to a plain Postgres instance.
  auth_user_id  uuid unique,
  created_at    timestamptz not null default now()
);

comment on table students is 'Players. auth_user_id links to auth.users when signed in; null = anonymous.';


-- ---------------------------------------------------------------------
-- datasets — the catalogue the backend exposes
-- ---------------------------------------------------------------------

create table if not exists datasets (
  id             text primary key,                  -- 'iris_2f', 'wine_2f', ...
  name           text not null,
  description    text,
  source         text,                              -- 'sklearn.datasets.load_iris'
  feature_names  text[] not null default '{}',
  class_names    text[] not null default '{}',
  n_samples      integer not null check (n_samples > 0),
  n_features     integer not null check (n_features > 0),
  n_classes      integer not null check (n_classes >= 2),
  -- Split contract the backend guarantees. Recorded so a future change to the
  -- split is visible as a schema-level fact rather than a silent score shift.
  test_size      numeric(4,3) not null default 0.300,
  split_seed     integer      not null default 42,
  is_active      boolean      not null default true,
  created_at     timestamptz  not null default now()
);


-- ---------------------------------------------------------------------
-- algorithms — the catalogue + the hyperparameter contract
-- ---------------------------------------------------------------------

create table if not exists algorithms (
  id                  text primary key,             -- 'decision_tree', 'lightgbm', ...
  name                text not null,
  family              algo_family not null,
  library             text not null default 'scikit-learn',
  blurb               text,
  tier                smallint not null default 1,  -- unlock ordering in the UI
  is_active           boolean  not null default true,
  -- Declarative spec the frontend renders sliders/selects from and the backend
  -- validates against. One object per tunable parameter, e.g.
  --   {"max_depth": {"type":"int","min":1,"max":20,"default":3,"step":1,
  --                  "label":"Max depth","overfit_lever":"up"}}
  hyperparameter_spec jsonb not null default '{}'::jsonb,
  created_at          timestamptz not null default now()
);

comment on column algorithms.hyperparameter_spec is
  'Per-parameter UI + validation spec. Single source of truth for the sliders and for backend clamping.';


-- ---------------------------------------------------------------------
-- runs — the core submission table
-- ---------------------------------------------------------------------

create table if not exists runs (
  id                 uuid primary key default gen_random_uuid(),
  student_id         uuid not null references students(id)   on delete cascade,
  dataset_id         text not null references datasets(id)   on delete restrict,
  algorithm_id       text not null references algorithms(id) on delete restrict,

  -- Exactly what the student tuned, verbatim, after backend clamping.
  -- e.g. {"max_depth": 4, "min_samples_split": 6, "criterion": "gini"}
  hyperparams        jsonb not null default '{}'::jsonb,

  -- --- metrics -------------------------------------------------------
  train_macro_f1     numeric(6,5) check (train_macro_f1 between 0 and 1),
  test_macro_f1      numeric(6,5) check (test_macro_f1  between 0 and 1),
  train_accuracy     numeric(6,5) check (train_accuracy between 0 and 1),
  test_accuracy      numeric(6,5) check (test_accuracy  between 0 and 1),
  test_roc_auc       numeric(6,5) check (test_roc_auc   between 0 and 1),

  -- Stored for the stability badge; deliberately NOT part of the score.
  cv_macro_f1_mean   numeric(6,5) check (cv_macro_f1_mean between 0 and 1),
  cv_macro_f1_std    numeric(6,5) check (cv_macro_f1_std  >= 0),

  -- --- scoring (see docs/SCORING.md) ---------------------------------
  generalization_gap numeric(7,5),          -- train_macro_f1 - test_macro_f1
  overfit_penalty    numeric(7,5) not null default 0,
  simplicity_bonus   numeric(7,5) not null default 0,
  complexity_ratio   numeric(6,5) check (complexity_ratio between 0 and 1),
  aries_score        integer      check (aries_score between 0 and 1000),
  verdict            overfit_verdict,

  -- --- provenance ----------------------------------------------------
  status             run_status  not null default 'complete',
  error_message      text,
  train_duration_ms  integer     check (train_duration_ms >= 0),
  model_seed         integer     not null default 42,
  engine_version     text        not null default 'v1',
  client_ip_hash     text,                  -- hashed, for light abuse control
  created_at         timestamptz not null default now()
);

comment on table runs is 'One row per training submission. hyperparams is the source of truth for what was tuned.';

create index if not exists runs_board_idx
  on runs (dataset_id, algorithm_id, aries_score desc, test_macro_f1 desc, created_at asc)
  where status = 'complete';

create index if not exists runs_dataset_board_idx
  on runs (dataset_id, aries_score desc, created_at asc)
  where status = 'complete';

create index if not exists runs_student_idx    on runs (student_id, created_at desc);
create index if not exists runs_created_at_idx on runs (created_at desc);
create index if not exists runs_hyperparams_idx on runs using gin (hyperparams);


-- ---------------------------------------------------------------------
-- run_hyperparams — normalized mirror of runs.hyperparams
--
-- The JSONB column is the source of truth; this table exists so questions
-- like "what max_depth do students actually pick, and does it correlate
-- with score?" are a plain SQL GROUP BY instead of JSON gymnastics.
-- ---------------------------------------------------------------------

create table if not exists run_hyperparams (
  run_id      uuid not null references runs(id) on delete cascade,
  param_name  text not null,
  value_num   numeric,      -- populated for numeric parameters
  value_text  text,         -- populated for categorical parameters ('gini')
  primary key (run_id, param_name)
);

create index if not exists run_hyperparams_param_idx on run_hyperparams (param_name, value_num);

-- Keeps run_hyperparams in sync automatically — the app only ever writes JSONB.
create or replace function sync_run_hyperparams() returns trigger
language plpgsql as $$
begin
  delete from run_hyperparams where run_id = new.id;
  insert into run_hyperparams (run_id, param_name, value_num, value_text)
  select new.id,
         kv.key,
         case when jsonb_typeof(kv.value) = 'number'
              then (kv.value #>> '{}')::numeric end,
         case when jsonb_typeof(kv.value) in ('string', 'boolean')
              then kv.value #>> '{}' end
  from jsonb_each(new.hyperparams) as kv;
  return new;
end $$;

drop trigger if exists runs_sync_hyperparams on runs;
create trigger runs_sync_hyperparams
  after insert or update of hyperparams on runs
  for each row execute function sync_run_hyperparams();


-- ---------------------------------------------------------------------
-- run_snapshots — the per-step training trace
--
-- One row per animation frame, so a run can be replayed on the results
-- page (or shared) without retraining. Boundary grids stay out of the DB;
-- they are large and cheap to recompute.
-- ---------------------------------------------------------------------

create table if not exists run_snapshots (
  run_id         uuid    not null references runs(id) on delete cascade,
  step_index     integer not null check (step_index >= 0),
  step_label     text    not null,          -- 'Depth 3', '40 trees'
  train_macro_f1 numeric(6,5),
  test_macro_f1  numeric(6,5),
  train_accuracy numeric(6,5),
  test_accuracy  numeric(6,5),
  extras         jsonb not null default '{}'::jsonb,   -- n_leaves, learning_rate, ...
  primary key (run_id, step_index)
);


-- ---------------------------------------------------------------------
-- roast_templates — the comment engine, editable without a redeploy
-- ---------------------------------------------------------------------

create table if not exists roast_templates (
  id           bigserial primary key,
  verdict      overfit_verdict not null,
  algorithm_id text references algorithms(id) on delete cascade,  -- null = applies to all
  text         text not null,
  weight       smallint not null default 1 check (weight > 0),
  is_active    boolean  not null default true,
  unique (verdict, text)
);

create index if not exists roast_templates_lookup_idx
  on roast_templates (verdict, algorithm_id) where is_active;


-- =====================================================================
-- Leaderboard views
-- =====================================================================

-- Per dataset + algorithm: every student's best run, ranked.
create or replace view leaderboard_by_algorithm as
select
  r.dataset_id,
  r.algorithm_id,
  a.name  as algorithm_name,
  s.id    as student_id,
  s.handle,
  s.initials,
  r.id    as run_id,
  r.aries_score,
  r.test_macro_f1,
  r.test_accuracy,
  r.generalization_gap,
  r.verdict,
  r.hyperparams,
  r.created_at,
  rank() over (
    partition by r.dataset_id, r.algorithm_id
    order by r.aries_score desc, r.test_macro_f1 desc,
             r.generalization_gap asc, r.created_at asc
  ) as position
from (
  select distinct on (student_id, dataset_id, algorithm_id) *
  from runs
  where status = 'complete' and aries_score is not null
  order by student_id, dataset_id, algorithm_id,
           aries_score desc, test_macro_f1 desc, created_at asc
) r
join students   s on s.id = r.student_id
join algorithms a on a.id = r.algorithm_id;

-- Per dataset, across all algorithms: the board that answers
-- "which algo + hyperparameter combination is best for THIS dataset?"
create or replace view leaderboard_by_dataset as
select
  r.dataset_id,
  s.id   as student_id,
  s.handle,
  s.initials,
  r.id   as run_id,
  r.algorithm_id,
  r.aries_score,
  r.test_macro_f1,
  r.generalization_gap,
  r.verdict,
  r.hyperparams,
  r.created_at,
  rank() over (
    partition by r.dataset_id
    order by r.aries_score desc, r.test_macro_f1 desc,
             r.generalization_gap asc, r.created_at asc
  ) as position
from (
  select distinct on (student_id, dataset_id) *
  from runs
  where status = 'complete' and aries_score is not null
  order by student_id, dataset_id, aries_score desc, test_macro_f1 desc, created_at asc
) r
join students s on s.id = r.student_id;

-- Global board: each student's single best run anywhere.
create or replace view leaderboard_global as
select
  s.id as student_id,
  s.handle,
  s.initials,
  max(r.aries_score)              as best_score,
  count(*)                        as total_runs,
  count(distinct r.algorithm_id)  as algorithms_tried,
  count(distinct r.dataset_id)    as datasets_tried,
  rank() over (order by max(r.aries_score) desc, min(r.created_at) asc) as position
from runs r
join students s on s.id = r.student_id
where r.status = 'complete' and r.aries_score is not null
group by s.id, s.handle, s.initials;


-- =====================================================================
-- Row Level Security (Supabase)
--
-- Reads are public — a leaderboard nobody can see is not a leaderboard.
-- Writes go through the backend using the service-role key, which bypasses
-- RLS; no anon insert policy is granted, so a student cannot POST a
-- fabricated 1000-point run straight into the table from the browser.
-- =====================================================================

alter table students        enable row level security;
alter table runs            enable row level security;
alter table run_snapshots   enable row level security;
alter table run_hyperparams enable row level security;
alter table datasets        enable row level security;
alter table algorithms      enable row level security;
alter table roast_templates enable row level security;

do $$
declare t text;
begin
  foreach t in array array['students','runs','run_snapshots','run_hyperparams',
                           'datasets','algorithms','roast_templates']
  loop
    execute format('drop policy if exists %I on %I', t || '_public_read', t);
    execute format('create policy %I on %I for select using (true)', t || '_public_read', t);
  end loop;
end $$;


-- =====================================================================
-- Seed data — catalogue rows matching the current backend
-- =====================================================================

insert into datasets (id, name, description, source, feature_names, class_names,
                      n_samples, n_features, n_classes)
values
  ('iris_2f', 'Iris (2 features)',
   'Classic flower measurements. Petal size alone separates the species almost perfectly.',
   'sklearn.datasets.load_iris',
   array['petal length (cm)','petal width (cm)'],
   array['setosa','versicolor','virginica'], 150, 2, 3),
  ('wine_2f', 'Wine (2 features)',
   'Chemical analysis of wines from three cultivars. Messier boundaries than iris.',
   'sklearn.datasets.load_wine',
   array['alcohol','flavanoids'],
   array['class_0','class_1','class_2'], 178, 2, 3),
  ('cancer_2f', 'Breast cancer (2 features)',
   'Tumor measurements, benign vs malignant. Imbalanced — this is why we rank on macro-F1.',
   'sklearn.datasets.load_breast_cancer',
   array['mean radius','mean texture'],
   array['malignant','benign'], 569, 2, 2)
on conflict (id) do update
  set name = excluded.name, description = excluded.description;

insert into algorithms (id, name, family, library, tier, is_active, blurb, hyperparameter_spec)
values
  ('decision_tree', 'Decision Tree', 'tree', 'scikit-learn', 1, true,
   'Splits the data one question at a time until each region is pure.',
   '{"max_depth":{"type":"int","min":1,"max":20,"default":3,"label":"Max depth","overfit_lever":"up"},
     "min_samples_split":{"type":"int","min":2,"max":50,"default":2,"label":"Min samples to split","overfit_lever":"down"},
     "min_samples_leaf":{"type":"int","min":1,"max":30,"default":1,"label":"Min samples per leaf","overfit_lever":"down"},
     "criterion":{"type":"enum","options":["gini","entropy","log_loss"],"default":"gini","label":"Split criterion"}}'::jsonb),

  ('random_forest', 'Random Forest', 'bagging', 'scikit-learn', 2, true,
   'Grows many different trees and lets them vote. Averaging cancels out each tree''s noise.',
   '{"n_estimators":{"type":"int","min":1,"max":200,"default":50,"label":"Number of trees"},
     "max_depth":{"type":"int","min":1,"max":20,"default":6,"label":"Max depth","overfit_lever":"up"},
     "max_features":{"type":"enum","options":["sqrt","log2","none"],"default":"sqrt","label":"Features per split"},
     "min_samples_leaf":{"type":"int","min":1,"max":30,"default":1,"label":"Min samples per leaf","overfit_lever":"down"}}'::jsonb),

  ('adaboost', 'AdaBoost', 'boosting', 'scikit-learn', 3, true,
   'Trains weak stumps in sequence, each one focusing on what the last one got wrong.',
   '{"n_estimators":{"type":"int","min":1,"max":200,"default":50,"label":"Boosting rounds","overfit_lever":"up"},
     "learning_rate":{"type":"float","min":0.01,"max":2.0,"default":1.0,"step":0.01,"label":"Learning rate","overfit_lever":"up"},
     "max_depth":{"type":"int","min":1,"max":6,"default":1,"label":"Stump depth","overfit_lever":"up"}}'::jsonb),

  ('gradient_boosting', 'Gradient Boosting', 'boosting', 'scikit-learn', 4, true,
   'Each new tree is fitted to the errors left over by all the trees before it.',
   '{"n_estimators":{"type":"int","min":1,"max":300,"default":100,"label":"Boosting rounds","overfit_lever":"up"},
     "learning_rate":{"type":"float","min":0.01,"max":1.0,"default":0.1,"step":0.01,"label":"Learning rate","overfit_lever":"up"},
     "max_depth":{"type":"int","min":1,"max":10,"default":3,"label":"Max depth","overfit_lever":"up"},
     "subsample":{"type":"float","min":0.3,"max":1.0,"default":1.0,"step":0.05,"label":"Row subsample","overfit_lever":"up"}}'::jsonb),

  ('lightgbm', 'LightGBM', 'boosting', 'lightgbm', 5, false,
   'Leaf-wise gradient boosting. Very fast, very easy to overfit if you let it off the leash.',
   '{"n_estimators":{"type":"int","min":1,"max":500,"default":100,"label":"Boosting rounds","overfit_lever":"up"},
     "learning_rate":{"type":"float","min":0.01,"max":0.5,"default":0.1,"step":0.01,"label":"Learning rate","overfit_lever":"up"},
     "num_leaves":{"type":"int","min":2,"max":128,"default":31,"label":"Number of leaves","overfit_lever":"up"},
     "min_child_samples":{"type":"int","min":1,"max":100,"default":20,"label":"Min samples per leaf","overfit_lever":"down"},
     "reg_lambda":{"type":"float","min":0.0,"max":10.0,"default":0.0,"step":0.1,"label":"L2 regularization","overfit_lever":"down"}}'::jsonb)
on conflict (id) do update
  set name = excluded.name, blurb = excluded.blurb,
      hyperparameter_spec = excluded.hyperparameter_spec;

insert into roast_templates (verdict, text) values
  ('underfit',       'Your model gave up before it started. Loosen the leash — more depth, more rounds.'),
  ('underfit',       'That is a very confident line through data it never looked at. Try letting it grow.'),
  ('clean',          'Textbook. Train and test agree — this thing actually learned something.'),
  ('clean',          'Clean generalization. This is the run you screenshot.'),
  ('slight_overfit', 'Nice pick, pal. It is starting to memorize, but you got away with it.'),
  ('slight_overfit', 'Solid run. A little leakage of memory into the score — trim it and you climb.'),
  ('overfit',        'Look at you overfitting. It knows the training set by heart and panics on anything new.'),
  ('overfit',        'Great on the exam it already saw. Cap the depth and try again.'),
  ('hard_overfit',   'That is not learning, that is a lookup table with extra steps.'),
  ('hard_overfit',   'Perfect on train, lost on test. The model built a shrine to your training data.')
on conflict do nothing;
