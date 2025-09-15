BEGIN;

-- add password_hash to users (nullable for existing users)
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS password_hash TEXT;

-- ensure one clue per (image, contributor)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint c
    JOIN pg_class t ON c.conrelid = t.oid
    WHERE c.conname = 'uq_clues_image_contributor'
  ) THEN
    ALTER TABLE clues
      ADD CONSTRAINT uq_clues_image_contributor UNIQUE (image_id, contributor);
  END IF;
EXCEPTION WHEN duplicate_object THEN
  -- ignore
END;
$$;

COMMIT;
