# In main.py

import csv
import io
import os
import html
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

# Constants for configuration and magic strings
DOCS_DIR = "docs"
DATA_DIR_BASE = "data"
CONTENT_KEY = "_content"
COMMENT_TYPE = "comment"

# Section header prefixes mapped to their nesting levels
SECTION_HEADERS = {"# ####### ": 5, "# ###### ": 4, "# ##### ": 3, "# #### ": 2, "# ### ": 1, "# ## ": 0}

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def set_nested_item(data_dict: Dict, path: List[str], value: Any) -> None:
    """
    Sets item in nested dictionary under the '_content' key.

    Args:
        data_dict: The dictionary to modify
        path: List of keys representing the hierarchy path
        value: The value to append to the content list
    """
    current_dict = data_dict
    # Navigate to the second-to-last level
    for key in path[:-1]:
        current_dict = current_dict.setdefault(key, OrderedDict())
        current_dict.setdefault(CONTENT_KEY, [])

    # Handle the final key
    final_key = path[-1]
    current_dict = current_dict.setdefault(final_key, OrderedDict())
    current_dict.setdefault(CONTENT_KEY, []).append(value)


def parse_header(lines: List[str]) -> Tuple[Optional[List[str]], int]:
    """
    Parse the header line from the file content.

    Args:
        lines: List of file lines

    Returns:
        Tuple of (header fields list, line number) or (None, 0) if not found
    """
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            header = next(csv.reader([line]))
            logger.info(f"Found header: {header}")
            return header, line_num
        except Exception as e:
            logger.error(f"Error parsing header line {line_num}: {e}")
            return None, 0

    logger.error("Reached EOF before finding header")
    return None, 0


def parse_section_header(line: str) -> Tuple[int, Optional[str]]:
    """
    Parse a section header line to determine level and title.

    Args:
        line: Input line to parse

    Returns:
        Tuple of (level, title) or (-1, None) if not a section header
    """
    for prefix, level in SECTION_HEADERS.items():
        if line.startswith(prefix):
            title = line[len(prefix) :].strip()
            return level, title

    return -1, None


def parse_custom_structured_file(filepath_relative_to_docs: str) -> Tuple[Optional[List[str]], Optional[Dict]]:
    """
    Parses the custom file format, capturing section headers, data rows,
    and regular comments into the '_content' list.

    Args:
        filepath_relative_to_docs: Path to the data file, relative to docs directory

    Returns:
        Tuple of (header, parsed_structure) or (None, None) on error
    """
    parsed_structure = OrderedDict()
    current_path = []
    full_filepath = Path(DOCS_DIR) / filepath_relative_to_docs

    logger.info(f"Opening file: {full_filepath}")

    try:
        # Read the file content
        with io.open(full_filepath, "r", newline="", encoding="utf-8-sig") as f:
            lines = f.readlines()

        # Find header
        header, line_num = parse_header(lines)
        if not header:
            return None, None

        # Process remaining lines
        for line_idx, line in enumerate(lines[line_num:]):
            current_line_num = line_num + line_idx + 1
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # Check if it's a section header
                level, title = parse_section_header(line)

                if level != -1 and title is not None:
                    # Update the current path based on section nesting level
                    current_path = current_path[:level] + [title]
                else:
                    # Handle regular comments
                    if current_path:
                        comment_text = line.lstrip("#")
                        if comment_text.startswith(" "):
                            comment_text = comment_text[1:]
                        comment_text = comment_text.rstrip("\n\r")

                        # Store comment tuple in _content list
                        set_nested_item(parsed_structure, current_path, (COMMENT_TYPE, comment_text))
            else:
                # Process data line
                if not current_path:
                    continue

                try:
                    data_values = next(csv.reader([line]))
                    if len(data_values) == len(header):
                        data_dict_for_row = OrderedDict(zip(header, data_values))
                        set_nested_item(parsed_structure, current_path, data_dict_for_row)
                    else:
                        logger.warning(
                            f"Line {current_line_num} column mismatch: expected {len(header)}, got {len(data_values)}"
                        )
                except Exception as e:
                    logger.error(f"Error parsing data line {current_line_num}: {e}")

    except FileNotFoundError:
        logger.error(f"File not found: '{full_filepath}'")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error parsing '{full_filepath}': {e}")
        return None, None

    return header, parsed_structure


def render_structure_to_markdown(header, structure, level=2):
    """
    Recursively renders the parsed structure (including comments) into Markdown.

    Parameters:
        header (list): Column headers for data tables
        structure (OrderedDict): Nested dictionary structure representing sections and content
        level (int): Current heading level (default: 2)

    Returns:
        str: Generated Markdown content
    """
    if not header:
        return "<p style='color:red;'>Error: Cannot render without header.</p>"

    result = []  # Use a list for better performance with many string concatenations

    def render_table(rows):
        """Helper function to render a Markdown table"""
        if not rows:
            return []

        table_lines = []
        # Header row
        header_cells = [escape_md_table_cell(h) for h in header]
        table_lines.append("| " + " | ".join(header_cells) + " |")
        # Separator row
        table_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        # Data rows
        for row_dict in rows:
            try:
                row_values = [escape_md_table_cell(str(row_dict.get(h, ""))) for h in header]
                table_lines.append("| " + " | ".join(row_values) + " |")
            except Exception as e:
                logger.warning(f"Error rendering table row {row_dict}: {e}")
                # Add a placeholder for the problematic row
                table_lines.append(f"| {'ERROR' + ' | ERROR' * (len(header) - 1)} |")

        table_lines.append("")  # Extra newline after table
        return table_lines

    def escape_md_table_cell(text):
        """Escape pipe characters in Markdown table cells"""
        return text.replace("|", "\\|").replace("\n", "<br>")

    for key, value in structure.items():
        if key == CONTENT_KEY:
            continue  # Skip internal content key

        # Add section heading
        result.append(f"{'#' * level} {key}\n")

        # Process the content list for this section
        content_list = value.get(CONTENT_KEY, [])
        data_rows = []

        for item in content_list:
            if isinstance(item, tuple) and item[0] == COMMENT_TYPE:
                # Output table before comment if data rows were accumulated
                if data_rows:
                    result.extend(render_table(data_rows))
                    data_rows = []  # Reset after rendering

                # Render comment text
                result.append(f"{item[1]}\n")
            elif isinstance(item, dict):
                # Accumulate data row
                data_rows.append(item)

        # Render any remaining table data
        if data_rows:
            result.extend(render_table(data_rows))

        # Process child sections recursively
        children = OrderedDict((k, v) for k, v in value.items() if k != CONTENT_KEY)
        if children:
            child_content = render_structure_to_markdown(header, children, level + 1)
            result.append(child_content)

    return "\n".join(result)


def render_custom_format(page):
    """
    Main entry point to render custom CSV data files into Markdown.

    Args:
        page: The page object provided by MkDocs

    Returns:
        str: Rendered Markdown content or error message
    """
    try:
        # Derive data file path based on markdown file
        md_src_path = page.file.src_path
        md_parent_dir = os.path.dirname(md_src_path)
        md_filename_stem = os.path.splitext(os.path.basename(md_src_path))[0]
        data_filename = f"{md_filename_stem}.csv"
        data_path = Path(DATA_DIR_BASE) / md_parent_dir / data_filename
        data_path_relative_to_docs = str(data_path)

        logger.info(f"Processing data file: {data_path_relative_to_docs}")

        # Parse and render the data
        header, structure = parse_custom_structured_file(data_path_relative_to_docs)

        if header and structure:
            return render_structure_to_markdown(header, structure, level=2)
        elif header and not structure:
            return f"<p><em>File '{html.escape(data_path_relative_to_docs)}' parsed (header found), but no valid sections or data were identified.</em></p>"
        else:
            return f"<p style='color:red;'><strong>Error processing data file '{html.escape(data_path_relative_to_docs)}' referenced implicitly by page '{html.escape(md_src_path)}'. Check build logs.</strong></p>"

    except AttributeError:
        return "<p style='color:red;'><strong>Error: 'page' object not passed correctly or missing attributes. Ensure macro call is `{{ render_custom_format(page=page) }}`.</strong></p>"
    except Exception as e:
        logger.error(
            f"Error in render_custom_format for page {getattr(page, 'file', {}).get('src_path', 'UNKNOWN')}: {e}"
        )
        return f"<p style='color:red;'><strong>An unexpected error occurred processing data for this page. Check build logs.</strong></p>"


def define_env(env):
    """Hook function for MkDocs plugin integration"""
    env.macro(render_custom_format)
