import requests
from typing import Optional
import json
import argparse

ZENODO_BASE_URL = "https://zenodo.org/api"


def get_latest_deposition_id(conceptrecid: str, access_token: str):
    url = f"{ZENODO_BASE_URL}/records/{conceptrecid}"

    response = requests.get(url, params={"access_token": access_token})
    response.raise_for_status()

    latest_record = response.json()
    latest_deposition_id = latest_record["id"]

    return latest_deposition_id


def get_endpoints(deposition_id: str, access_token: str):
    params = {"access_token": access_token}
    url = f"{ZENODO_BASE_URL}/deposit/depositions/{deposition_id}"
    response = requests.get(url, params=params)
    response.raise_for_status()

    return response


def new_version(raw_endpoints: requests.Response, access_token):
    url = f"{raw_endpoints.json()['links']['latest_draft']}/actions/newversion"

    response = requests.post(url, params={"access_token": access_token})
    response.raise_for_status()

    return response


def discard(raw_endpoints: requests.Response, access_token):
    discard_api = raw_endpoints.json()["links"]["discard"]
    response = requests.post(
        discard_api, params={"access_token": access_token}
    )

    # No need to check response here---it is possible that we cannot
    # discard somehing because there is no draft.


def update_metadata(data, raw_endpoints, access_token):
    edit_api = raw_endpoints.json()["links"]["latest_draft"]
    headers = {"Content-Type": "application/json"}

    response = requests.put(
        edit_api,
        params={"access_token": access_token},
        data=json.dumps({"metadata": data}),
        headers=headers,
    )
    response.raise_for_status()


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
            requests.put(
                "%s/%s" % (bucket_url, filename),
                data=fp,
                params=params,
            )

    def new_version(self):
        r = new_version(self.raw_endpoints, self.access_token)
        self.raw_endpoints = r
        self.deposition_id = r.json()["id"]

    def edit_metadata(
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

    parser.add_argument("-c", "--concept-record-id", required=True)
    parser.add_argument("-t", "--access_token", type=str, required=True)
    parser.add_argument(
        "-v", "--version", default="0.0.0", type=str, required=True
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        help="List of file paths to upload",
    )

    args = parser.parse_args()

    access_token = args.access_token
    concept_record_id = args.concept_record_id
    version = args.version
    files = args.files

    deposition_id = get_latest_deposition_id(concept_record_id, access_token)
    zenodo_api = ZenodoAPI(access_token, deposition_id)

    zenodo_api.new_version()
    zenodo_api.delete_files()

    for file in files:
        zenodo_api.upload_file(file, file)

    zenodo_api.edit_metadata(version=version)
    zenodo_api.publish()
    zenodo_api.discard()


if __name__ == "__main__":
    main()
