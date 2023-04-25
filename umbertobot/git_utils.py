import os
import fnmatch
from pathlib import Path as P


def _read_ignore_patterns(gitignore_filename):
    with open(gitignore_filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        line_nonempty = line.strip() == ""
        line_not_comment = line.startswith("#")
        line_not_negative = line_nonempty or line[0] != "!"
        if line_not_comment and not line_nonempty and line_not_negative:
            yield line


def _read_include_patterns(gitignore_filename):
    with open(gitignore_filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("!"):
            yield line[1:]


def filter_gitignore_files(file_paths, gitignore_filename=".gitignore"):
    result = []
    ignore_patterns = list(_read_ignore_patterns(gitignore_filename))
    include_patterns = list(_read_include_patterns(gitignore_filename))
    for path in file_paths:
        # Check if the file matches any ignore pattern
        matches = [fnmatch.fnmatch(path.name, pattern) for pattern in ignore_patterns]
        print(path, any(matches))
        if not any(matches):
            result.append(path)
        else:
            if any(
                [fnmatch.fnmatch(path.name, pattern) for pattern in include_patterns]
            ):
                result.append(path)
    return result


def get_unignored_files(directory, gitignore_filename, glob_pattern="*"):
    """
    return files from directory matching glob pattern that are not in .gitignore
    """
    file_paths = list(P(directory).expanduser().rglob("*"))
    return _filter_gitignore_files(file_paths, gitignore_filename)
