# create_benchmark_table.py

import argparse
import json
import re
import sys
from pathlib import Path
from pprint import pprint
from collections import defaultdict
from typing import Dict, Any, Optional

import pandas as pd
import html # Import the html module for escaping


# Regular expression to parse "test_name (size)" format
DIR_PATTERN = re.compile(r"^(.*?) \((.*?)\)$")

# Standard location for criterion estimates relative to the benchmark dir
ESTIMATES_PATH_NEW = Path("new") / "estimates.json"
# Fallback location (older versions or baseline comparisons)
ESTIMATES_PATH_BASE = Path("base") / "estimates.json"

# Standard location for the HTML report relative to the benchmark's specific directory
REPORT_HTML_RELATIVE_PATH = Path("report") / "index.html"


def load_criterion_reports(criterion_root_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Loads Criterion benchmark results from a specified directory and finds HTML paths.

    Args:
        criterion_root_dir: The Path object pointing to the main
                           'target/criterion' directory.

    Returns:
        A nested dictionary structured as:
        { test_name: { size: {'json': json_content, 'html_path': relative_html_path}, ... }, ... }
        Returns an empty dict if the root directory is not found or empty.
    """
    results: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    if not criterion_root_dir.is_dir():
        print(
            f"Error: Criterion root directory not found or is not a directory: {criterion_root_dir}",
            file=sys.stderr,
        )
        return {}

    print(f"Scanning for benchmark reports in: {criterion_root_dir}")

    for item in criterion_root_dir.iterdir():
        # We are only interested in directories matching the pattern
        if not item.is_dir():
            continue

        match = DIR_PATTERN.match(item.name)
        if not match:
            # print(f"Skipping directory (name doesn't match pattern): {item.name}")
            continue

        test_name = match.group(1).strip()
        size = match.group(2).strip()
        benchmark_dir_name = item.name # Store the original directory name
        benchmark_dir_path = item     # The Path object to the benchmark dir

        json_path: Optional[Path] = None

        # Look for the estimates JSON file (prefer 'new', fallback to 'base')
        if (benchmark_dir_path / ESTIMATES_PATH_NEW).is_file():
            json_path = benchmark_dir_path / ESTIMATES_PATH_NEW
        elif (benchmark_dir_path / ESTIMATES_PATH_BASE).is_file():
            json_path = benchmark_dir_path / ESTIMATES_PATH_BASE

        # The HTML report is at a fixed location relative to the benchmark directory
        html_path = benchmark_dir_path / REPORT_HTML_RELATIVE_PATH


        if json_path is None or not json_path.is_file():
            print(
                f"Warning: Could not find estimates JSON in {benchmark_dir_path}. Skipping benchmark size '{test_name} ({size})'.",
                file=sys.stderr,
            )
            continue # Skip if no JSON data

        if not html_path.is_file():
             print(
                f"Warning: Could not find HTML report at expected location {html_path}. Skipping benchmark size '{test_name} ({size})'.",
                file=sys.stderr,
            )
             continue # Skip if no HTML report

        # Try loading the JSON data
        try:
            with json_path.open("r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Store both the JSON data and the relative path to the HTML report
            results[test_name][size] = {
                'json': json_data,
                # The path from the criterion root to the specific benchmark's report/index.html
                'html_path_relative_to_criterion_root': str(Path(benchmark_dir_name) / REPORT_HTML_RELATIVE_PATH).replace('\\', '/') # Ensure forward slashes
            }
            # print(f"  Loaded: {test_name} ({size}) from {json_path}, html: {html_path}")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {json_path}", file=sys.stderr)
        except IOError as e:
            print(f"Error: Failed to read file {json_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(
                f"Error: An unexpected error occurred loading {json_path}: {e}",
                file=sys.stderr,
            )

    # Convert defaultdict back to regular dict for cleaner output (optional)
    return dict(results)


def format_nanoseconds(ns: float) -> str:
    """Formats nanoseconds into a human-readable string with units."""
    if pd.isna(ns):
        return "-"
    if ns < 1_000:
        return f"{ns:.2f} ns"
    elif ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    else:
        return f"{ns / 1_000_000_000:.2f} s"


def generate_html_table_with_links(results: Dict[str, Dict[str, Dict[str, Any]]], html_base_path: str) -> str:
    """
    Generates an HTML table from benchmark results, with cells linking to reports.

    Args:
        results: The nested dictionary loaded by load_criterion_reports,
                 including 'json' data and 'html_path_relative_to_criterion_root'.
        html_base_path: The base URL path where the 'target/criterion' directory
                        is hosted on the static site, relative to the output HTML file.
                        e.g., '../target/criterion/'

    Returns:
        A string containing the full HTML table.
    """
    if not results:
        return "<p>No benchmark results found or loaded.</p>"

    # Get all unique sizes (columns) and test names (rows)
    # Using ordered dictionaries to maintain insertion order from loading, then sorting keys
    # Or simply sort the keys after extraction:
    all_sizes = sorted(list(set(size for test_data in results.values() for size in test_data.keys())))
    all_test_names = sorted(list(results.keys()))

    html_string = """
    <meta charset="utf-8">
    <h1 id="criterion-benchmark-results">Criterion Benchmark Results</h1>
    <p>Each cell links to the detailed Criterion report for that specific benchmark size.</p>
    <p>Note: Values shown are the midpoint of the mean confidence interval, formatted for readability.</p>
    <table class="table table-striped" border="1" justify="center">
        <thead>
            <tr>
                <th>Benchmark Name</th>
    """

    # Add size headers
    for size in all_sizes:
        html_string += f"<th>{html.escape(size)}</th>\n"

    html_string += """
            </tr>
        </thead>
        <tbody>
    """

    # Add data rows
    for test_name in all_test_names:
        html_string += f"<tr>\n"
        html_string += f"    <td>{html.escape(test_name)}</td>\n"

        # Iterate through all possible sizes to ensure columns align
        for size in all_sizes:
            cell_data = results.get(test_name, {}).get(size)
            mean_value = pd.NA # Default value
            full_report_url = "#" # Default link to self or dummy

            if cell_data and 'json' in cell_data and 'html_path_relative_to_criterion_root' in cell_data:
                try:
                    # Extract mean from JSON
                    mean_data = cell_data['json'].get("mean")
                    if mean_data and "confidence_interval" in mean_data:
                        ci = mean_data["confidence_interval"]
                        if "lower_bound" in ci and "upper_bound" in ci:
                             lower, upper = ci["lower_bound"], ci["upper_bound"]
                             if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                                 mean_value = (lower + upper) / 2.0
                             else:
                                 print(f"Warning: Non-numeric bounds for {test_name} ({size}).", file=sys.stderr)
                        else:
                             print(f"Warning: Missing confidence_interval bounds for {test_name} ({size}).", file=sys.stderr)
                    else:
                         print(f"Warning: Missing 'mean' data for {test_name} ({size}).", file=sys.stderr)

                    # Construct the full relative URL
                    relative_report_path = cell_data['html_path_relative_to_criterion_root']
                    full_report_url = f"{html_base_path}{relative_report_path}"
                    # Ensure forward slashes and resolve potential double slashes if html_base_path ends in /
                    full_report_url = str(Path(full_report_url)).replace('\\', '/')


                except Exception as e:
                    print(f"Error processing cell data for {test_name} ({size}): {e}", file=sys.stderr)
                    # Keep mean_value as NA and URL as '#'

            # Format the mean value for display
            formatted_mean = format_nanoseconds(mean_value)

            # Create the link cell
            # Only make it a link if a valid report path was found
            if full_report_url and full_report_url != "#":
                 html_string += f'    <td><a href="{html.escape(full_report_url)}">{html.escape(formatted_mean)}</a></td>\n'
            else:
                 # Display value without a link if no report path
                 html_string += f'    <td>{html.escape(formatted_mean)}</td>\n'


        html_string += f"</tr>\n"

    html_string += """
        </tbody>
    </table>
    """

    return html_string


if __name__ == "__main__":
    DEFAULT_CRITERION_PATH = "target/criterion"
    # Default relative path from benchmark_results.html to the criterion root on the hosted site
    # Assumes benchmark_results.html is in .../doc/<branch-slug>/benchmarks/
    # And target/criterion is copied to .../doc/<branch-slug>/target/criterion/
    # So the path from benchmarks/ to target/criterion/ is ../target/criterion/
    DEFAULT_HTML_BASE_PATH = "../target/criterion/"

    parser = argparse.ArgumentParser(
        description="Load Criterion benchmark results from JSON files and generate an HTML table with links to reports."
    )
    parser.add_argument(
        "--criterion-dir",
        type=str,
        default=DEFAULT_CRITERION_PATH,
        help=f"Path to the main 'target/criterion' directory (default: {DEFAULT_CRITERION_PATH}) on the runner.",
    )
    parser.add_argument(
        "--html-base-path",
        type=str,
        default=DEFAULT_HTML_BASE_PATH,
        help=f"Relative URL path from the output HTML file to the hosted 'target/criterion' directory (default: {DEFAULT_HTML_BASE_PATH}).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmark_results.html",
        help="Name of the output HTML file (default: benchmark_results.html)."
    )


    args = parser.parse_args()

    criterion_path = Path(args.criterion_dir)
    all_results = load_criterion_reports(criterion_path)

    if not all_results:
        print("\nNo benchmark results found or loaded.")
        # Still create an empty file or a file with an error message
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write("<h1>Criterion Benchmark Results</h1><p>No benchmark results found or loaded.</p>")
            print(f"Created empty/error HTML file: {args.output_file}")
        except IOError as e:
             print(f"Error creating empty/error HTML file {args.output_file}: {e}", file=sys.stderr)
        sys.exit(1) # Indicate failure if no data was loaded successfully

    print("\nSuccessfully loaded benchmark results.")
    # pprint(all_results) # Uncomment for debugging

    print(f"Generating HTML table with links using base path: {args.html_base_path}")
    html_output = generate_html_table_with_links(all_results, args.html_base_path)

    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(html_output)
        print(f"\nSuccessfully wrote HTML table to {args.output_file}")
        sys.exit(0) # Exit successfully
    except IOError as e:
        print(f"Error writing HTML output to {args.output_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred while writing HTML: {e}", file=sys.stderr)
         sys.exit(1)