"""People labeler: proposes names for the UNNAMED people (face clusters) of an Immich library owner.

Name evidence ("references"), each turned into a face embedding with Immich's own model and settings:

* **folders**: a folder named after a person (``.../people/Jane Doe/...``). The person who appears in most of the
  folder's photos (the "dominant identity") becomes the reference. The embeddings are the ones Immich already stored
  for those photos, so nothing is recomputed.
* **web pages**: a public listing page with one card per person (name + portrait). The portrait goes through Immich's
  machine-learning server (``POST /predict``), exactly like the face API does.

References that show the same face are joined into one identity. Every unnamed cluster is compared with every
identity (cosine distance between centroids, plus how many of the cluster's faces are close to the identity), and
each proposal gets a band: ``high`` (safe to apply automatically, once the operator allows it), ``medium`` and
``low`` (for human review). Clusters that look like the same person are listed as merge candidates; nothing is ever
merged automatically.

Safety: dry run by default. Names are only written through the Immich API with the owner's API key, only to people
whose name is still empty for every user who has a copy of the person, and every write is recorded in an audit
table so that ``undo`` can revert it (only while nobody has changed the name since).
"""
