import functools
import os
import warnings
from collections import Counter

import requests


@functools.lru_cache(maxsize=1)
def _resolve_latest_version() -> str:
    """Resolve the ``latest`` release alias to the actual release tag.

    GitHub redirects ``releases/latest`` to ``releases/tag/<tag>``; the
    tag is read from the redirect target. Pinning the resolved tag makes
    new releases land in a fresh root directory instead of silently
    reusing a stale ``latest`` cache. If the tag cannot be resolved
    (e.g. no network), ``"latest"`` is returned unchanged; the caller
    falls back to a locally cached release when one exists. The result
    is cached for the lifetime of the process, so constructing several
    datasets costs one request.
    """
    try:
        response = requests.head(
            "https://github.com/aidos-lab/MANTRA/releases/latest",
            allow_redirects=False,
            timeout=10,
        )
        location = response.headers["Location"]
        return location.rstrip("/").rsplit("/", 1)[-1]
    except Exception as e:
        warnings.warn(
            f"Could not resolve the latest MANTRA release tag ({e})."
        )
        return "latest"


def _find_cached_version(root, dimension) -> str | None:
    """Find the newest release tag already cached under ``root``.

    Used as an offline fallback when the ``latest`` alias cannot be
    resolved: a previously downloaded release keeps working without
    network access.
    """
    base = os.path.join(root, "mantra")
    if not os.path.isdir(base):
        return None

    versions = []
    for name in os.listdir(base):
        if not os.path.isdir(os.path.join(base, name, f"{dimension}D")):
            continue
        try:
            versions.append(
                (tuple(int(x) for x in name.lstrip("v").split(".")), name)
            )
        except ValueError:
            continue

    return max(versions)[1] if versions else None


def _get_mantra_dataset_url(
    version: str, dimension: int, balanced: bool = False, validate: bool = True
) -> str:
    """Get URL to download dataset from.

    With ``validate=True``, a pinned version is checked against the
    published GitHub releases so that typos fail with a clear message.
    Validation is skipped for tags that were just resolved from the
    ``latest`` alias (they came from GitHub itself) and degrades to a
    warning when the release listing cannot be fetched (e.g. API rate
    limit), so a warm on-disk cache never becomes unusable.
    """
    suffix = "_balanced" if balanced else ""
    filename = f"{dimension}_manifolds{suffix}.json.gz"

    if version == "latest":
        return f"https://github.com/aidos-lab/MANTRA/releases/latest/download/{filename}"  # noqa

    if validate:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        try:
            payload = requests.get(
                "https://api.github.com/repos/aidos-lab/mantra/releases",
                headers=headers,
                params={"per_page": 100},
                timeout=10,
            ).json()
        except Exception as e:
            payload = {"message": str(e)}

        if not isinstance(payload, list):
            warnings.warn(
                "Could not validate the MANTRA version against the GitHub "
                f"releases ({payload.get('message', payload)}); continuing "
                "without validation."
            )
        else:
            all_available_versions = [
                item.get("tag_name") or item.get("name") for item in payload
            ]
            if version not in all_available_versions:
                raise ValueError(
                    f"Version {version} not available, please choose one of the following versions: {all_available_versions}."  # noqa
                )

    return f"https://github.com/aidos-lab/MANTRA/releases/download/{version}/{filename}"  # noqa


def filter_by_class_count(entries, label_source, min_count):
    """Drop entries whose ``label_source`` value occurs <= ``min_count`` times."""
    if min_count <= 0:
        return entries, Counter()
    counts = Counter(e[label_source] for e in entries)
    kept_labels = {lbl for lbl, c in counts.items() if c > min_count}
    filtered = [e for e in entries if e[label_source] in kept_labels]
    return filtered, counts
