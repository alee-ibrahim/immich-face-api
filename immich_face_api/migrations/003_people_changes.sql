-- People change log (see README "People changes"). Immich records only deletions in its audit tables (no merge target,
-- no old name, no face moves), so the API keeps its own lightweight snapshot of the owner's named people and of which
-- named person each face belongs to, and turns the differences into events. Ids and names only: no images, no
-- embeddings. The snapshot is rebuilt from Immich at any time; the events are the only history.

-- the owner's named, not hidden people as last seen ("tracked" people = what GET /v1/people returns)
CREATE TABLE people_snapshot (
    person_id   uuid PRIMARY KEY,
    name        text NOT NULL,
    faces       integer NOT NULL DEFAULT 0,          -- faces in face_snapshot (same filter as the API)
    first_seen  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- which tracked person each counted face belonged to, as last seen
CREATE TABLE face_snapshot (
    face_id    uuid PRIMARY KEY,
    person_id  uuid NOT NULL
);
CREATE INDEX face_snapshot_person ON face_snapshot (person_id);

CREATE TABLE people_changes (
    id            bigserial PRIMARY KEY,                 -- the cursor
    at            timestamptz NOT NULL DEFAULT now(),    -- when the tracker saw the change
    window_start  timestamptz,                           -- the change happened after this (the previous check)
    type          text NOT NULL CHECK (type IN ('created', 'renamed', 'faces_moved', 'merged', 'deleted')),
    person_id     uuid NOT NULL,                         -- the person (created/renamed/deleted) or the source (faces_moved/merged)
    to_person_id  uuid,                                  -- faces_moved / merged: the target (NULL = faces unassigned)
    name          text,                                  -- the person's (new / last known) name
    old_name      text,                                  -- renamed: the name before
    to_name       text,                                  -- the target's name at that time (NULL = not a named person)
    faces         integer,                               -- faces_moved / merged: how many tracked faces moved
    reason        text                                   -- deleted: deleted | hidden | unnamed
);
CREATE INDEX people_changes_at ON people_changes (at);
CREATE INDEX people_changes_person ON people_changes (person_id);
CREATE INDEX people_changes_to_person ON people_changes (to_person_id) WHERE to_person_id IS NOT NULL;

-- one row: the tracker's state
CREATE TABLE people_tracker (
    id                 smallint PRIMARY KEY CHECK (id = 1),
    owner_id           uuid NOT NULL,
    started_at         timestamptz NOT NULL,             -- changes before this are not known
    checked_at         timestamptz NOT NULL,             -- Immich's clock at the last successful check
    last_full_at       timestamptz NOT NULL,             -- last full reconcile
    purged_through     bigint NOT NULL DEFAULT 0,        -- events with id <= this were deleted (retention / reset)
    last_cycle_ms      integer,
    last_changed_faces integer,
    last_events        integer,
    last_error         text,
    last_error_at      timestamptz
);
