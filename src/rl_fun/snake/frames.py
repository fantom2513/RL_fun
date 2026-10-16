"""Construct visual frames lazily, preserving ordinary callback compatibility."""


def emit_frame(callback, env, **metadata):
    if callback is None or not getattr(callback, "enabled", lambda: True)():
        return
    due = getattr(callback, "should_emit", None)
    if due is None or due(ended=metadata.get("ended", False)):
        callback({"frame": env.snapshot(), **metadata})
