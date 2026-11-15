-- Main-subject labels from curators ("Label subjects" in the admin UI; see README "Main subject"). One row per photo
-- shown for labelling: the face boxes, the signals the API measured and its prediction, and - once a curator has
-- labelled it - one label per face. No images, no embeddings: an Immich photo is referenced by its asset id only and
-- fetched from Immich again while it is shown; an uploaded photo is never sent back to the server after detection.
CREATE TABLE subject_sample (
    id             bigserial PRIMARY KEY,
    source         text NOT NULL CHECK (source IN ('immich', 'upload')),
    asset_id       uuid,                                  -- source 'immich'
    created_by     text NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    image_w        integer NOT NULL,
    image_h        integer NOT NULL,
    model_version  text NOT NULL,                         -- the subject model that made the prediction
    photo_type     text NOT NULL,
    faces          jsonb NOT NULL,                        -- [{box: {x, y, w, h}, prominence, role, signals: {...}}]
    labels         jsonb,                                 -- ["main" | "co_subject" | "background", ...] per face
    skipped        boolean NOT NULL DEFAULT false,        -- "not sure / bad photo": kept out of calibration
    labeled_by     text,
    labeled_at     timestamptz
);
CREATE INDEX subject_sample_asset ON subject_sample (asset_id);
CREATE INDEX subject_sample_labeled ON subject_sample (labeled_at) WHERE labeled_at IS NOT NULL;
