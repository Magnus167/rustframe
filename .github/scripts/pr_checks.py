import os
import sys
import urllib.request
import urllib.error
import json
from typing import Any, Dict, List, Optional, Tuple
import warnings
import urllib.parse

from time import sleep

sys.path.append(os.getcwd())

ACCESS_TOKEN: Optional[str] = os.getenv("GH_TOKEN", None)

REQUEST_CONFIG = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {ACCESS_TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
}
REPO_OWNER_USERNAME: str = "Magnus167"
REPO_NAME: str = "rustframe"
REPOSITORY_WEB_LINK: str = f"github.com/{REPO_OWNER_USERNAME}/{REPO_NAME}"


def perform_api_call(
    target_url: str,
    call_headers: Optional[dict] = REQUEST_CONFIG,
    query_parameters: Dict[str, Any] = {},
    http_method: str = "GET",
    maximum_attempts: int = 5,
) -> Any:
    assert http_method in ["GET", "DELETE", "POST", "PATCH", "PUT"]

    attempt_count = 0
    while attempt_count < maximum_attempts:
        try:
            if query_parameters:
                encoded_parameters = urllib.parse.urlencode(query_parameters)
                target_url = f"{target_url}?{encoded_parameters}"

            http_request_object = urllib.request.Request(target_url, method=http_method)

            if call_headers:
                for key, value in call_headers.items():
                    http_request_object.add_header(key, value)

            with urllib.request.urlopen(http_request_object) as server_response:
                if server_response.status == 404:
                    raise Exception(f"404: {target_url} not found.")

                return json.loads(server_response.read().decode())

        except urllib.error.HTTPError as error_details:
            unrecoverable_codes = [403, 404, 422]
            if error_details.code in unrecoverable_codes:
                raise Exception(f"Request failed: {error_details}")

            print(f"Request failed: {error_details}")
            attempt_count += 1
            sleep(1)

        except Exception as error_details:
            print(f"Request failed: {error_details}")
            attempt_count += 1
            sleep(1)

    raise Exception("Request failed")


valid_title_prefixes: List[str] = [
    "Feature:",
    "Bugfix:",
    "Documentation:",
    "CI/CD:",
    "Misc:",
    "Suggestion:",
]


def validate_title_format(
    item_title: str,
) -> bool:
    estr = "Skipping PR title validation"
    for _ in range(5):
        warnings.warn(estr)
        print(estr)
    return True

    is_format_correct: bool = False
    for prefix_pattern in valid_title_prefixes:
        cleaned_input: str = item_title.strip()
        if cleaned_input.startswith(prefix_pattern):
            is_format_correct = True
            break

    if not is_format_correct:
        issue_message: str = (
            f"PR title '{item_title}' does not match any "
            f"of the accepted patterns: {valid_title_prefixes}"
        )
        raise ValueError(issue_message)

    return is_format_correct


def _locate_segment_indices(
    content_string: str,
    search_pattern: str,
    expect_numeric_segment: bool = False,
) -> Tuple[int, int]:
    numeric_characters: List[str] = list(map(str, range(10))) + ["."]
    assert bool(content_string)
    assert bool(search_pattern)
    assert search_pattern in content_string
    start_index: int = content_string.find(search_pattern)
    end_index: int = content_string.find("-", start_index)
    if end_index == -1 and not expect_numeric_segment:
        return (start_index, len(content_string))

    if expect_numeric_segment:
        start_index = start_index + len(search_pattern)
        for char_index, current_character in enumerate(content_string[start_index:]):
            if current_character not in numeric_characters:
                break
        end_index = start_index + char_index

    return (start_index, end_index)


def _verify_no_merge_flag(
    content_string: str,
) -> bool:
    assert bool(content_string)
    return "DO-NOT-MERGE" not in content_string


def _verify_merge_dependency(
    content_string: str,
) -> bool:
    assert bool(content_string)
    dependency_marker: str = "MERGE-AFTER-#"

    if dependency_marker not in content_string:
        return True

    start_index, end_index = _locate_segment_indices(
        content_string=content_string, pattern=dependency_marker, numeric=True
    )
    dependent_item_id: str = content_string[start_index:end_index].strip()
    try:
        dependent_item_id = int(dependent_item_id)
    except ValueError:
        issue_message: str = f"PR number '{dependent_item_id}' is not an integer."
        raise ValueError(issue_message)

    dependent_item_data: Dict[str, Any] = fetch_item_details(
        item_identifier=dependent_item_id
    )
    is_dependent_item_closed: bool = dependent_item_data["state"] == "closed"
    return is_dependent_item_closed


def evaluate_merge_conditions(
    item_details: Dict[str, Any],
) -> bool:
    item_body_content: str = item_details["body"]

    if item_body_content is None:
        return True

    item_body_content = item_body_content.strip().replace(" ", "-").upper()
    item_body_content = f" {item_body_content} "

    condition_outcomes: List[bool] = [
        _verify_no_merge_flag(content_string=item_body_content),
        _verify_merge_dependency(content_string=item_body_content),
    ]

    return all(condition_outcomes)


def validate_item_for_merge(
    item_data: Dict[str, Any],
) -> bool:
    assert set(["number", "title", "state", "body"]).issubset(item_data.keys())
    accumulated_issues: str = ""
    if not validate_title_format(item_title=item_data["title"]):
        accumulated_issues += (
            f"PR #{item_data['number']} is not mergable due to invalid title.\n"
        )

    if not evaluate_merge_conditions(item_details=item_data):
        accumulated_issues += (
            f"PR #{item_data['number']} is not mergable due to merge restrictions"
            " specified in the PR body."
        )

    if accumulated_issues:
        raise ValueError(accumulated_issues.strip())

    return True


def fetch_item_details(
    item_identifier: int,
):
    api_request_url: str = f"https://api.github.com/repos/{REPO_OWNER_USERNAME}/{REPO_NAME}/pulls/{item_identifier}"

    raw_api_response_data: Dict[str, Any] = perform_api_call(target_url=api_request_url)

    extracted_item_info: Dict[str, Any] = {
        "number": raw_api_response_data["number"],
        "title": raw_api_response_data["title"],
        "state": raw_api_response_data["state"],
        "body": raw_api_response_data["body"],
    }

    return extracted_item_info


def process_item_request(requested_item_id: int):
    extracted_item_info: Dict[str, Any] = fetch_item_details(
        item_identifier=requested_item_id
    )
    if not validate_item_for_merge(item_data=extracted_item_info):
        raise ValueError("PR is not mergable.")

    print("PR is mergable.")

    return True


if __name__ == "__main__":
    requested_item_id: int = int(sys.argv[1])
    process_item_request(requested_item_id=requested_item_id)
