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
import html  # Import the html module for escaping


# Regular expression to parse "test_name (size)" format
DIR_PATTERN = re.compile(r"^(.*?) \((.*?)\)$")

# Standard location for criterion estimates relative to the benchmark dir
ESTIMATES_PATH_NEW = Path("new") / "estimates.json"
# Fallback location (older versions or baseline comparisons)
ESTIMATES_PATH_BASE = Path("base") / "estimates.json"

# Standard location for the HTML report relative to the benchmark's specific directory
REPORT_HTML_RELATIVE_PATH = Path("report") / "index.html"


def get_default_criterion_report_path() -> Path:
    """
    Returns the default path for the Criterion benchmark report.
    This is typically 'target/criterion'.
    """
    return Path("target") / "criterion" / "report" / "index.html"


def load_criterion_reports(
    criterion_root_dir: Path,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
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
        if not item.is_dir():
            continue

        match = DIR_PATTERN.match(item.name)
        if not match:
            continue

        test_name = match.group(1).strip()
        size = match.group(2).strip()
        benchmark_dir_name = item.name
        benchmark_dir_path = item

        json_path: Optional[Path] = None

        if (benchmark_dir_path / ESTIMATES_PATH_NEW).is_file():
            json_path = benchmark_dir_path / ESTIMATES_PATH_NEW
        elif (benchmark_dir_path / ESTIMATES_PATH_BASE).is_file():
            json_path = benchmark_dir_path / ESTIMATES_PATH_BASE

        html_path = benchmark_dir_path / REPORT_HTML_RELATIVE_PATH

        if json_path is None or not json_path.is_file():
            print(
                f"Warning: Could not find estimates JSON in {benchmark_dir_path}. Skipping benchmark size '{test_name} ({size})'.",
                file=sys.stderr,
            )
            continue

        if not html_path.is_file():
            print(
                f"Warning: Could not find HTML report at expected location {html_path}. Skipping benchmark size '{test_name} ({size})'.",
                file=sys.stderr,
            )
            continue

        try:
            with json_path.open("r", encoding="utf-8") as f:
                json_data = json.load(f)

            results[test_name][size] = {
                "json": json_data,
                "html_path_relative_to_criterion_root": str(
                    Path(benchmark_dir_name) / REPORT_HTML_RELATIVE_PATH
                ).replace("\\", "/"),
            }
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {json_path}", file=sys.stderr)
        except IOError as e:
            print(f"Error: Failed to read file {json_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(
                f"Error: An unexpected error occurred loading {json_path}: {e}",
                file=sys.stderr,
            )

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


def generate_html_table_with_links(
    results: Dict[str, Dict[str, Dict[str, Any]]], html_base_path: str
) -> str:
    """
    Generates a full HTML page with a styled table from benchmark results.
    """
    css_styles = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f4f7f6;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        p.subtitle {
            text-align: center;
            margin-bottom: 8px;
            color: #555;
            font-size: 0.95em;
        }
        p.note {
            text-align: center;
            margin-bottom: 25px;
            color: #777;
            font-size: 0.85em;
        }
        .benchmark-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .benchmark-table th, .benchmark-table td {
            border: 1px solid #dfe6e9; /* Lighter border */
            padding: 12px 15px;
        }
        .benchmark-table th {
            background-color: #3498db; /* Primary blue */
            color: #ffffff;
            font-weight: 600; /* Slightly bolder */
            text-transform: uppercase;
            letter-spacing: 0.05em;
            text-align: center; /* Center align headers */
        }
        .benchmark-table td {
            text-align: right; /* Default for data cells (times) */
        }
        .benchmark-table td:first-child { /* Benchmark Name column */
            font-weight: 500;
            color: #2d3436;
            text-align: left; /* Left align benchmark names */
        }
        .benchmark-table tbody tr:nth-child(even) {
            background-color: #f8f9fa; /* Very light grey for even rows */
        }
        .benchmark-table tbody tr:hover {
            background-color: #e9ecef; /* Slightly darker on hover */
        }
        .benchmark-table a {
            color: #2980b9; /* Link blue */
            text-decoration: none;
            font-weight: 500;
        }
        .benchmark-table a:hover {
            text-decoration: underline;
            color: #1c5a81; /* Darker blue on hover */
        }
        .no-results {
            text-align: center;
            font-size: 1.2em;
            color: #7f8c8d;
            margin-top: 30px;
        }
    </style>
    """

    html_doc_start = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Criterion Benchmark Results</title>
    {css_styles}
</head>
<body>
    <div class="container">
        <h1 id="criterion-benchmark-results">Criterion Benchmark Results</h1>
"""

    html_doc_end = """
    </div>
</body>
</html>"""

    if not results:
        return f"""{html_doc_start}
        <p class="no-results">No benchmark results found or loaded.</p>
{html_doc_end}"""

    all_sizes = sorted(
        list(set(size for test_data in results.values() for size in test_data.keys())),
        key=(lambda x: int(x.split("x")[0])),
    )
    all_test_names = sorted(list(results.keys()))

    table_content = """
        <p class="subtitle">Each cell links to the detailed Criterion.rs report for that specific benchmark size.</p>
        <p class="note">Note: Values shown are the midpoint of the mean confidence interval, formatted for readability.</p>
        <p class="note"><a href="report/index.html">[Switch to the standard Criterion.rs report]</a></p>
        <table class="benchmark-table">
            <thead>
                <tr>
                    <th>Benchmark Name</th>
    """

    for size in all_sizes:
        table_content += f"<th>{html.escape(size)}</th>\n"

    table_content += """
                </tr>
            </thead>
            <tbody>
    """

    for test_name in all_test_names:
        table_content += f"<tr>\n"
        table_content += f"    <td>{html.escape(test_name)}</td>\n"

        for size in all_sizes:
            cell_data = results.get(test_name, {}).get(size)
            mean_value = pd.NA
            full_report_url = "#"

            if (
                cell_data
                and "json" in cell_data
                and "html_path_relative_to_criterion_root" in cell_data
            ):
                try:
                    mean_data = cell_data["json"].get("mean")
                    if mean_data and "confidence_interval" in mean_data:
                        ci = mean_data["confidence_interval"]
                        if "lower_bound" in ci and "upper_bound" in ci:
                            lower, upper = ci["lower_bound"], ci["upper_bound"]
                            if isinstance(lower, (int, float)) and isinstance(
                                upper, (int, float)
                            ):
                                mean_value = (lower + upper) / 2.0
                            else:
                                print(
                                    f"Warning: Non-numeric bounds for {test_name} ({size}).",
                                    file=sys.stderr,
                                )
                        else:
                            print(
                                f"Warning: Missing confidence_interval bounds for {test_name} ({size}).",
                                file=sys.stderr,
                            )
                    else:
                        print(
                            f"Warning: Missing 'mean' data for {test_name} ({size}).",
                            file=sys.stderr,
                        )

                    relative_report_path = cell_data[
                        "html_path_relative_to_criterion_root"
                    ]
                    joined_path = Path(html_base_path) / relative_report_path
                    full_report_url = str(joined_path).replace("\\", "/")

                except Exception as e:
                    print(
                        f"Error processing cell data for {test_name} ({size}): {e}",
                        file=sys.stderr,
                    )

            formatted_mean = format_nanoseconds(mean_value)

            if full_report_url and full_report_url != "#":
                table_content += f'    <td><a href="{html.escape(full_report_url)}">{html.escape(formatted_mean)}</a></td>\n'
            else:
                table_content += f"    <td>{html.escape(formatted_mean)}</td>\n"
        table_content += "</tr>\n"

    table_content += """
            </tbody>
        </table>
    """
    return f"{html_doc_start}{table_content}{html_doc_end}"


if __name__ == "__main__":
    DEFAULT_CRITERION_PATH = "target/criterion"
    DEFAULT_OUTPUT_FILE = "./target/criterion/index.html"
    DEFAULT_HTML_BASE_PATH = ""

    parser = argparse.ArgumentParser(
        description="Load Criterion benchmark results from JSON files and generate an HTML table with links to reports."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without writing the HTML file.",
    )
    parser.add_argument(
        "--criterion-dir",
        type=str,
        default=DEFAULT_CRITERION_PATH,
        help=f"Path to the main 'target/criterion' directory (default: {DEFAULT_CRITERION_PATH}) containing benchmark data.",
    )
    parser.add_argument(
        "--html-base-path",
        type=str,
        default=DEFAULT_HTML_BASE_PATH,
        help=(
            f"Prefix for HTML links to individual benchmark reports. "
            f"This is prepended to each report's relative path (e.g., 'benchmark_name/report/index.html'). "
            f"If the main output HTML (default: '{DEFAULT_OUTPUT_FILE}') is in the 'target/criterion/' directory, "
            f"this should typically be empty (default: '{DEFAULT_HTML_BASE_PATH}'). "
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to save the generated HTML summary report (default: {DEFAULT_OUTPUT_FILE}).",
    )

    args = parser.parse_args()

    if args.dry_run:
        print(
            "Dry run mode: No files will be written. Use --dry-run to skip writing the HTML file."
        )
        sys.exit(0)

    criterion_path = Path(args.criterion_dir)
    output_file_path = Path(args.output_file)

    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(
            f"Error: Could not create output directory {output_file_path.parent}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    all_results = load_criterion_reports(criterion_path)

    # Generate HTML output regardless of whether results were found (handles "no results" page)
    html_output = generate_html_table_with_links(all_results, args.html_base_path)

    if not all_results:
        print("\nNo benchmark results found or loaded.")
        # Fallthrough to write the "no results" page generated by generate_html_table_with_links
    else:
        print("\nSuccessfully loaded benchmark results.")
        # pprint(all_results) # Uncomment for debugging

    print(
        f"Generating HTML report with links using HTML base path: '{args.html_base_path}'"
    )

    try:
        with output_file_path.open("w", encoding="utf-8") as f:
            f.write(html_output)
        print(f"\nSuccessfully wrote HTML report to {output_file_path}")
        if not all_results:
            sys.exit(1)  # Exit with error code if no results, though file is created
        sys.exit(0)
    except IOError as e:
        print(f"Error writing HTML output to {output_file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while writing HTML: {e}", file=sys.stderr)
        sys.exit(1)
