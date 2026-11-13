-- Database roles for the people labeler. Run as the owner of Immich's tables (or a superuser). Set passwords
-- separately (\password in psql, or your operator's managed-role feature), never in this file.
-- Restrict logins in pg_hba.conf, TLS only, e.g.:
--   hostssl immich          immich_labeler <client network> scram-sha-256
--   hostssl immich_labeler  immich_labeler <client network> scram-sha-256

CREATE ROLE immich_labeler LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION CONNECTION LIMIT 4;

-- 1. Read access to Immich (the labeler also opens every Immich session read-only itself)
GRANT CONNECT ON DATABASE immich TO immich_labeler;
GRANT USAGE ON SCHEMA public TO immich_labeler;
GRANT SELECT ON TABLE person, asset_face, face_search, version_history TO immich_labeler;
GRANT SELECT (id, "ownerId", "deletedAt", "originalPath") ON TABLE asset TO immich_labeler;  -- folder rules only

-- 2. Its own state database (runs, proposals, audit log, caches): owned by the labeler, separate from Immich
CREATE DATABASE immich_labeler OWNER immich_labeler;

-- 3. Optional, for `share-names` only: a SEPARATE role (keep the one above read-only). Set
--    LABELER_DB_USER/PASSWORD to it for that command and LABELER_STATE_DB_USER/PASSWORD to immich_labeler.
CREATE ROLE immich_name_copy LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION CONNECTION LIMIT 2;
GRANT CONNECT ON DATABASE immich TO immich_name_copy;
GRANT USAGE ON SCHEMA public TO immich_name_copy;
GRANT SELECT ("ownerId", "personGroupId", name), UPDATE (name) ON TABLE person TO immich_name_copy;
GRANT SELECT ("personGroupId", "sharedById", "sharedWithId") ON TABLE person_user TO immich_name_copy;
GRANT SELECT (version, "createdAt") ON TABLE version_history TO immich_name_copy;
