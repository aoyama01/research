import os


def get_relative_path(target_path, base_path=os.getcwd()):
    """
    Get the relative path from the base_path to the target_path.

    :param target_path: The target directory or file path.
    :param base_path: The base directory path. Defaults to the current working directory.
    :return: The relative path from base_path to target_path.
    """
    return os.path.relpath(target_path, base_path)


def read_file(file_path):
    """
    Read the contents of a file.

    :param file_path: The path to the file.
    :return: The contents of the file as a string.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_file(file_path, content):
    """
    Write content to a file.

    :param file_path: The path to the file.
    :param content: The content to write to the file.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def list_files(directory_path):
    """
    List all files in a directory.

    :param directory_path: The path to the directory.
    :return: A list of file names in the directory.
    """
    return [
        f
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]


def list_directories(directory_path):
    """
    List all directories in a directory.

    :param directory_path: The path to the directory.
    :return: A list of directory names in the directory.
    """
    return [
        d
        for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d))
    ]
