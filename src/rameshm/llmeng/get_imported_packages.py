import os
import re
from typing import Set, List

def is_directory_or_file(path: str) -> str:
    if os.path.isdir(path):
        return 'directory'
    elif os.path.isfile(path):
        return 'file'
    else:
        return 'neither'

def get_imported_packages(path: str, package_set: Set, excluded_files: List =None):
    if excluded_files is None:
        excluded_files = []
    print(f"Process Path/File: {path}")
    # Regex pattern to find import statements
    #import_pattern = re.compile(r'^\s*(?:import|from)\s+([\w\.]+)')
    import_pattern = re.compile(r"(?:from|import)\s+(\w+(\.\w+)*)")


    # Determine if the path is a directory or file
    path_type = is_directory_or_file(path)

    for excluded_file in excluded_files:
        if excluded_file in path:
            return

    if path_type == 'file':
        # Process the file if it's not excluded
        if os.path.basename(path) not in excluded_files:
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    match = import_pattern.match(line)
                    if match:
                        package_set.add(match.group(1))
    elif path_type == 'directory':
        # Walk through the directory
        for root, _, files in os.walk(path):
            for file_name in files:
                    file_path = os.path.join(root, file_name)
                    get_imported_packages(file_path, package_set, excluded_files)
        else:
            print(f"The provided path: {path} is neither a file nor a directory.")

    #return package_set

# Example usage:
# path = 'your_directory_or_file_path_here'
# excluded_files = ['excluded_file1.py', 'excluded_file2.py']
# packages = get_imported_packages(path, excluded_files)
# print("Unique imported packages:")
# for package in sorted(packages):
#     print(package)


def main():
    paths_in = input("Please enter the directory or file list separated by '|': ")
    excluded_files_in = input(
        "Please enter the directory or file list that we don't to consider. Separate entries by '|': ")
    if paths_in:
        paths = paths_in.split("|")
    else:
        paths = [r"C:\git_repo\raindrive100\rameshm-llmeng\src\rameshm\llmeng"]
    if excluded_files_in:
        excluded_files = excluded_files_in.split("|")
    else:
        # Ignore all files and directories that have any of the below in their name
        excluded_files = ["scratch", "__pycache__", ".gitignore"]
    my_packages = set()
    for path in paths:
        print(path)
        get_imported_packages(path, my_packages, excluded_files)
    print("\n".join(sorted(my_packages)))


if __name__ == "__main__":
    main()