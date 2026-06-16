import requests


def _get_mantra_dataset_url(
    version: str, dimension: int, balanced: bool = False
) -> str:
    """Get URL to download dataset from."""
    suffix = "_balanced" if balanced else ""
    filename = f"{dimension}_manifolds{suffix}.json.gz"

    if version == "latest":
        return f"https://github.com/aidos-lab/MANTRA/releases/latest/download/{filename}"  # noqa

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(
        "https://api.github.com/repos/aidos-lab/mantra/releases",
        headers=headers,
    )

    all_available_versions = [item["name"] for item in response.json()]

    if version not in all_available_versions:
        raise ValueError(
            f"Version {version} not available, please choose one of the following versions: {all_available_versions}."  # noqa
        )

    # Note that the URL order is different and thus inconsistent for a
    # specific release.
    return f"https://github.com/aidos-lab/MANTRA/releases/download/{version}/{filename}"  # noqa
