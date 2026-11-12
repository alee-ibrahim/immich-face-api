-- Read-only database role for immich-face-api. Run as the owner of Immich's tables (or a superuser), once,
-- and again after an Immich upgrade that recreates any of these tables. Set the password separately
-- (e.g. \password immich_face_api in psql, or your operator's managed-role feature), never in this file.
--
-- Also restrict where it may log in from (pg_hba.conf), TLS only, e.g.:
--   hostssl immich immich_face_api <client network> scram-sha-256

CREATE ROLE immich_face_api LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION CONNECTION LIMIT 10;
ALTER ROLE immich_face_api SET default_transaction_read_only = on;
ALTER ROLE immich_face_api SET statement_timeout = '15s';

GRANT CONNECT ON DATABASE immich TO immich_face_api;
GRANT USAGE ON SCHEMA public TO immich_face_api;
GRANT SELECT ON TABLE person, asset_face, face_search, version_history TO immich_face_api;

-- Only if the owner is configured by e-mail (FACE_API_OWNER_EMAIL) instead of FACE_API_OWNER_ID:
-- GRANT SELECT (id, email) ON TABLE "user" TO immich_face_api;
