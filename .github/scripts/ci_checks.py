import os
import sys
from typing import Any, Dict, Optional
import tomllib
import packaging.version
import requests

sys.path.append(os.getcwd())

ACCESS_TOKEN: Optional[str] = os.getenv("GH_TOKEN", None)

GITHUB_REQUEST_CONFIG = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {ACCESS_TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
}
REPO_OWNER_USERNAME: str = "Magnus167"
REPO_NAME: str = "rustframe"
REPOSITORY_WEB_LINK: str = f"github.com/{REPO_OWNER_USERNAME}/{REPO_NAME}"

CARGO_TOML_PATH: str = "Cargo.toml"


def load_cargo_toml() -> Dict[str, Any]:
    if not os.path.exists(CARGO_TOML_PATH):
        raise FileNotFoundError(f"{CARGO_TOML_PATH} does not exist.")

    with open(CARGO_TOML_PATH, "rb") as file:
        return tomllib.load(file)

def get_latest_crates_io_version() -> str:
    url = "https://crates.io/api/v1/crates/rustframe"
    try:
        response = requests.get(url, headers=GITHUB_REQUEST_CONFIG)
        response.raise_for_status()
        data = response.json()
        return data["crate"]["max_version"]
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch latest version from crates.io: {e}")


def get_current_version() -> str:
    cargo_toml = load_cargo_toml()
    version = cargo_toml.get("package", {}).get("version", None)
    if not version:
        raise ValueError("Version not found in Cargo.toml")
    return version


def check_version() -> None:
    latest_version = get_latest_crates_io_version()
    latest_version_tuple = packaging.version.parse(latest_version)
    current_version = get_current_version()
    current_version_tuple = packaging.version.parse(current_version)

    # if the current version is >= latest, exit 1
    if latest_version_tuple >= current_version_tuple:
        sys.exit(1)

    print(f"Current version: {current_version_tuple}")


if __name__ == "__main__":
    check_version()
