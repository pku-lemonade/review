import csv
import io
import os
from collections import OrderedDict  # Use OrderedDict to maintain insertion order


# Helper function to navigate/create nested dictionary path
def set_nested_item(data_dict, path, value):
    """Sets item in nested dictionary, creating keys if necessary."""
    for key in path[:-1]:
        # Use setdefault which returns the value if key exists,
        # or inserts key with a default value and returns the default value.
        # We store data rows in a list under the key '_data'.
        # Child nodes are stored under their own keys.
        data_dict = data_dict.setdefault(key, OrderedDict())
        # Ensure _data list exists even for intermediate nodes
        data_dict.setdefault("_data", [])
    # Set the value at the final key
    final_key = path[-1]
    data_dict = data_dict.setdefault(final_key, OrderedDict())
    data_dict.setdefault("_data", []).append(value)


def parse_custom_structured_file(filepath):
    """
    Parses the custom file format with #-based headers and one data header.

    Args:
        filepath (str): Path to the custom structured file.

    Returns:
        tuple: (header_list, nested_data_structure) or (None, None) on error.
                nested_data_structure is an OrderedDict.
    """
    header = None
    # Use OrderedDict to preserve the order of sections as found in the file
    parsed_structure = OrderedDict()
    current_path = []  # Stores the current section path, e.g., ['Science', 'Physics']
    try:
        with io.open(os.path.join("docs", filepath), "r", encoding="utf-8-sig") as f:
            # Cannot use csv.reader directly due to comments and changing context
            lines = f.readlines()

        line_iterator = iter(lines)
        line_num = 0

        while True:  # Find the single header line
            line_num += 1
            try:
                line = next(line_iterator).strip()
                if not line or line.startswith("#"):
                    continue  # Skip comments and empty lines before header
                # Assume first non-comment line is the header
                # Use csv.reader just for this line to handle potential quoting
                header = next(csv.reader([line]))
                print(f"DEBUG: Found Header: {header}")
                break  # Header found
            except StopIteration:
                print("Error: Reached end of file without finding header row.")
                return None, None  # No header found
            except Exception as e:
                print(f"Error parsing presumed header line {line_num}: {e}")
                return None, None

        if not header:
            print("Error: Header not identified.")
            return None, None

        # Process remaining lines for sections and data
        for line in line_iterator:
            line_num += 1
            line = line.strip()

            if not line:  # Skip empty lines
                continue

            if line.startswith("#"):
                # Check for section markers (simple string checks, per user note)
                if line.startswith("# ##### "):
                    level = 3
                    title = line[len("# ##### ") :].strip()
                elif line.startswith("# #### "):
                    level = 2
                    title = line[len("# #### ") :].strip()
                elif line.startswith("# ### "):
                    level = 1
                    title = line[len("# ### ") :].strip()
                elif line.startswith("# ## "):
                    level = 0
                    title = line[len("# ## ") :].strip()
                else:
                    # It's just a regular comment, ignore
                    continue

                # Update current path based on level
                current_path = current_path[
                    :level
                ]  # Trim path back to the parent level
                current_path.append(title)
                print(f"DEBUG: Set Path: {current_path}")

            else:  # Should be a data line
                if not current_path:
                    print(
                        f"Warning: Data found before any section marker at line {line_num}. Skipping: {line}"
                    )
                    continue
                try:
                    # Parse the single data line using csv.reader
                    data_values = next(csv.reader([line]))
                    if len(data_values) == len(header):
                        data_dict_for_row = OrderedDict(zip(header, data_values))
                        # Associate this data with the current path
                        # Helper function needed here to navigate/create nested dict path
                        set_nested_item(
                            parsed_structure, current_path, data_dict_for_row
                        )
                        # print(f"DEBUG: Added data to {'/'.join(current_path)}")
                    else:
                        print(
                            f"Warning: Data line {line_num} has wrong column count. Expected {len(header)}, got {len(data_values)}. Line: '{line}'"
                        )
                except StopIteration:  # Handle empty line case if needed by csv.reader
                    continue
                except Exception as e:
                    print(f"Error parsing data line {line_num}: {e}. Line: '{line}'")
                    # Decide how to proceed - maybe skip line?
                    continue

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

    return header, parsed_structure


def render_structure_to_markdown(header, structure, level=2):
    """
    Recursively renders the parsed structure into Markdown.

    Args:
        header (list): The header list for data tables.
        structure (OrderedDict): The nested data structure.
        level (int): The starting Markdown header level (e.g., 2 for ##).

    Returns:
        str: The generated Markdown string.
    """
    markdown_output = ""
    if not header:
        return ""  # Need header to render tables

    for key, value in structure.items():
        if key == "_data":  # Skip internal data key
            continue

        # Add heading for the current key
        markdown_output += f"{'#' * level} {key}\n\n"

        # Check if this node has data directly associated with it
        node_data = value.get("_data", [])
        if node_data:
            # Generate table for data rows at this level
            markdown_output += "| " + " | ".join(header) + " |\n"
            markdown_output += "| " + " | ".join(["---"] * len(header)) + " |\n"
            for data_row_dict in node_data:
                # Ensure values are printed in the correct header order
                row_values = [str(data_row_dict.get(h, "")) for h in header]
                markdown_output += "| " + " | ".join(row_values) + " |\n"
            markdown_output += "\n"

        # Recursively render children, increasing the heading level
        # Pass only the children dict, excluding the '_data' key
        children_structure = OrderedDict(
            (k, v) for k, v in value.items() if k != "_data"
        )
        if children_structure:
            markdown_output += render_structure_to_markdown(
                header, children_structure, level + 1
            )

    return markdown_output


# Main macro function exposed to MkDocs
def render_custom_format(filepath):
    """
    Parses the custom structured file and renders it as Markdown.
    """
    print(f"Attempting to parse and render: {filepath}")
    header, structure = parse_custom_structured_file(filepath)
    if header and structure:
        return render_structure_to_markdown(
            header, structure, level=2
        )  # Start sections at H2
    elif header and not structure:
        return "<p><em>File parsed, header found, but no valid sections or data were identified.</em></p>"
    else:
        # Error messages printed during parsing
        return "<p style='color:red;'><strong>Error processing file. Check build logs.</strong></p>"


# Hook function for mkdocs-macros-plugin
def define_env(env):
    """Hook function"""
    # Expose the main rendering function to the Jinja2 environment
    env.macro(render_custom_format)
