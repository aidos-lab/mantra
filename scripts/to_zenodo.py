import requests
from typing import Optional
import json
import argparse

ZENODO_URL = "https://zenodo.org"


def get_endpoints(deposition_id: str, access_token: str):
    params = {"access_token": access_token}
    url = f"{ZENODO_URL}/api/deposit/depositions/{deposition_id}"
    r = requests.get(url, params=params)

    success = r.status_code == 200
    if not success:
        print("warning in get endpoints")
        print(r.json())
    return r


def new_version(raw_endpoints: requests.Response, access_token):
    url = f"{raw_endpoints.json()['links']['latest_draft']}/actions/newversion"

    r = requests.post(url, params={"access_token": access_token})
    success = r.status_code == 201

    if not success:
        print("warning in new version")
        print(r.json())
    return r


def discard(raw_endpoints: requests.Response, access_token):
    discard_api = raw_endpoints.json()["links"]["discard"]
    r = requests.post(discard_api, params={"access_token": access_token})
    success = r.status_code == 201
    if success:
        print("Discarded edit draft")
    else:
        print("no draft to discard.")


def update_metadata(data, raw_endpoints, access_token):
    edit_api = raw_endpoints.json()["links"]["latest_draft"]

    headers = {"Content-Type": "application/json"}
    r = requests.put(
        edit_api,
        params={"access_token": access_token},
        data=json.dumps({"metadata": data}),
        headers=headers,
    )
    success = r.status_code == 200
    if not success:
        print("warning in update metadata")
        print(r.json())


def delete(endpoints, access):
    files_api = endpoints.json()["links"]["files"]
    r = requests.get(files_api, params={"access_token": access})
    json_str = r.content.decode("utf-8")
    data = json.loads(json_str)
    for file in data:
        id_ = file["id"]
        r = requests.delete(
            f"{files_api}/{id_}", params={"access_token": access}
        )


class ZenodoAPI:

    def __init__(
        self,
        access_token: str,
        deposition_id: Optional[str] = None,
    ):
        assert deposition_id is not None
        self.deposition_id = deposition_id
        self.raw_endpoints = get_endpoints(deposition_id, access_token)
        self.access_token = access_token

    def upload_file(self, filename, path):
        bucket_url = self.raw_endpoints.json()["links"]["bucket"]
        params = {"access_token": self.access_token}

        with open(path, "rb") as fp:
            r = requests.put(
                "%s/%s" % (bucket_url, filename),
                data=fp,
                params=params,
            )

    def new_version(self):
        r = new_version(self.raw_endpoints, self.access_token)
        self.raw_endpoints = r
        self.deposition_id = r.json()["id"]

    def edit_metdata(
        self,
        upload_type: str = "dataset",
        title: Optional[str] = None,
        version: Optional[str] = None,
        creators=None,
    ):

        metadata = self.raw_endpoints.json()["metadata"]
        metadata["upload_type"] = upload_type

        if title is not None:
            metadata["title"] = title
        if version is not None:
            metadata["version"] = version
        if creators is not None:
            # add dummy creators
            metadata["creators"] = [
                {"name": "Doe, John", "affiliation": "Zenodo"}
            ]
        update_metadata(metadata, self.raw_endpoints, self.access_token)

    def publish(self):
        publish_endp = self.raw_endpoints.json()["links"]["publish"]
        cur = get_endpoints(self.deposition_id, self.access_token).json()
        r = requests.post(
            publish_endp, params={"access_token": self.access_token}, data=cur
        )

        success = r.status_code == 202
        if not success:
            print("warning in publish")
            print(r.json())
        return success

    def discard(self):
        discard(self.raw_endpoints, self.access_token)

    def delete_files(self):
        delete(self.raw_endpoints, self.access_token)


def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-id", "--deposition_id", help="")
    parser.add_argument("-t", "--access_token", type=str, help="")
    parser.add_argument("-v", "--version", default="0.0.0", type=str, help="")
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",  # Accepts one or more arguments
        help="List of file paths to upload",
    )

    args = parser.parse_args()

    deposition_id = args.deposition_id
    access_token = args.access_token
    version = args.version
    files = args.files

    zenodo_api = ZenodoAPI(access_token, deposition_id)

    # create new version: https://developers.zenodo.org/#new-version
    zenodo_api.new_version()
    zenodo_api.delete_files()

    # upload files
    for file in files:
        zenodo_api.upload_file(file, file)

    # update version tag
    zenodo_api.edit_metdata(version=version)

    # publish
    zenodo_api.publish()

    # discard draft
    zenodo_api.discard()


if __name__ == "__main__":
    main()
