"""One physical World with stable, public object identities.

Object IDs label perception tracks, never independent episodes or physics truth.
Legacy single-object packets remain valid without an objects mapping.
"""


def objects(observed):
    if "objects" not in observed:
        return {"object": observed["object"]}
    values = observed["objects"]
    if not isinstance(values, dict) or not values or any(
            not isinstance(key, str) or not key.strip() for key in values):
        raise ValueError("Scene objects require a nonempty mapping of stable object IDs")
    return values


def object_ids(observed):
    return sorted(objects(observed))


def check_objects(observed, expected):
    if object_ids(observed) != list(expected):
        raise ValueError("Scene object identities changed; do not substitute independent Worlds")


def visual_observation(observed):
    return {key: observed[key] for key in
        ("time_s", "coordinate_frame", "object", "objects", "tcp") if key in observed}


def supported(observed, tolerance=.003):
    return all(abs(obj["support_clearance_m"]) <= tolerance for obj in objects(observed).values())
