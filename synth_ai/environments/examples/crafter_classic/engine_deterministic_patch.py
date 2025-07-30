"""
Apply once (import this module anywhere before CrafterEngine is used).
It replaces Env._balance_object so that every per-chunk object list is
sorted by (x, y, class-name) before any random choice is made – removing
the hash-based set-iteration nondeterminism that caused the drift.
"""

import collections
import crafter

print("[PATCH] Attempting to apply Crafter deterministic patch...")

# -----------------------------------------------------------------------------
# 1.  Make per–chunk object order stable
# -----------------------------------------------------------------------------
if not hasattr(crafter.Env, "_orig_balance_object"):
    print("[PATCH] Patching crafter.Env._balance_object...")
    crafter.Env._orig_balance_object = crafter.Env._balance_object

    def _balance_object_det(self, chunk, objs, *args, **kwargs):
        # cls, material, span_dist, despan_dist, spawn_prob, despawn_prob, ctor, target_fn
        # were part of the original signature, but *args, **kwargs is more robust.
        objs = sorted(objs, key=lambda o: (o.pos[0], o.pos[1], o.__class__.__name__))
        return crafter.Env._orig_balance_object(self, chunk, objs, *args, **kwargs)

    crafter.Env._balance_object = _balance_object_det
    print("[PATCH] crafter.Env._balance_object patched.")
else:
    print("[PATCH] crafter.Env._balance_object already patched or _orig_balance_object exists.")

# -----------------------------------------------------------------------------
# 2.  Make *chunk* iteration order stable
# -----------------------------------------------------------------------------
if not hasattr(crafter.engine.World, "_orig_chunks_prop"):
    crafter.engine.World._orig_chunks_prop = crafter.engine.World.chunks

    def _chunks_sorted(self):
        # OrderedDict keeps the sorted key order during iteration
        return collections.OrderedDict(sorted(self._chunks.items()))

    crafter.engine.World.chunks = property(_chunks_sorted)

# -----------------------------------------------------------------------------
# 3. NEW: keep per-frame object update order deterministic
# -----------------------------------------------------------------------------
if not hasattr(crafter.engine.World, "_orig_objects_prop"):
    crafter.engine.World._orig_objects_prop = crafter.engine.World.objects  # save original

    @property
    def _objects_sorted(self):
        objs = [o for o in self._objects if o]  # Filter out None (removed) objects
        # stable order: x, y, class-name, creation-index
        return sorted(
            objs,
            key=lambda o: (
                o.pos[0],
                o.pos[1],
                o.__class__.__name__,
                getattr(o, "_id", 0),
            ),
        )

    crafter.engine.World.objects = _objects_sorted
