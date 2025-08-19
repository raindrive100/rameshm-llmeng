import os
import re
from typing import Set, List

processed_files = []
skipped_files = []

def is_directory_or_file(path: str) -> str:
    if os.path.isdir(path):
        return 'directory'
    elif os.path.isfile(path):
        return 'file'
    else:
        return 'neither'

def get_imported_packages(file_path: str, package_set: Set, excluded_files: List =None):
    if excluded_files is None:
        excluded_files = []
    # Regex pattern to find import statements
    #import_pattern = re.compile(r'^\s*(?:import|from)\s+([\w\.]+)')
    import_pattern = re.compile(r"(?:from|import)\s+(\w+(\.\w+)*)")


    # Determine if the path is a directory or file
    path_type = is_directory_or_file(file_path)

    for excluded_file in excluded_files:
        if excluded_file in file_path:
            skipped_files.append(file_path)
            return

    processed_files.append(file_path)

    if path_type == 'file':
        # Process the file if it's not excluded
        if os.path.basename(file_path) not in excluded_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    match = import_pattern.match(line)
                    if match:
                        package_set.add(match.group(1))
    elif path_type == 'directory':
        # Walk through the directory
        for root, _, files in os.walk(file_path):
            for file_name in files:
                    sub_file_path = os.path.join(root, file_name)
                    get_imported_packages(sub_file_path, package_set, excluded_files)
    else:
        print(f"The provided path: {file_path} is neither a file nor a directory.")

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
        paths = [r"C:\git_repo\raindrive100\rameshm-llmeng\src\rameshm\llmeng\rag"]
    if excluded_files_in:
        excluded_files = excluded_files_in.split("|")
    else:
        # Ignore all files and directories that have any of the below in their name
        excluded_files = ["scratch", "__pycache__", ".gitignore", ".yaml", "__init__.py"]
    my_packages = set()
    for path in paths:
        print(path)
        get_imported_packages(path, my_packages, excluded_files)
    processed_files_str = "\n".join(processed_files)
    skipped_files_str = "\n".join(skipped_files)
    my_packages_str = "\n".join(my_packages)
    print(f"\nProcessed Files: {processed_files_str}")
    print(f"\nSkipped Files: {skipped_files_str}")
    print(f"Package List: {my_packages_str}")


if __name__ == "__main__":
    main()