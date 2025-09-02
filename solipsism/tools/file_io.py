import os
import shutil
import logging
from asyncio import Queue
from ..core import tool
from ..core import lpml

logger = logging.getLogger(__name__)


# --- Tag Definitions for LLM ---

DEFINE_LIST_FILES = """
<define_tag name="list_files">
Lists all files and directories at a given path.
Attributes:
    - path (required): The directory path to list.
</define_tag>
""".strip()

DEFINE_READ_FILE = """
<define_tag name="read_file">
Reads the content of a specified file.
Attributes:
    - path (required): The path to the file.
    - line_numbers (optional): If "true", prepends line numbers to the output.
</define_tag>
""".strip()

DEFINE_WRITE_FILE = """
<define_tag name="write_file">
Writes or modifies a file. The content is placed inside the tag.
Attributes:
    - path (required): The path to the file.
    - mode (optional): "overwrite", "append", "replace_lines", "insert_at_line". Default: "overwrite".
    - start_line, end_line (for replace_lines): The 1-indexed line range to replace.
    - line (for insert_at_line): The 1-indexed line number to insert at.
</define_tag>
""".strip()

DEFINE_CREATE_DIRECTORY = """
<define_tag name="create_directory">
Creates a new directory.
Attributes:
    - path (required): The path of the directory to create.
    - parents (optional): If "true", creates parent directories as needed. Default: "false".
</define_tag>
""".strip()

DEFINE_MOVE_ITEM = """
<define_tag name="move_item">
Moves or renames a file or directory.
Attributes:
    - source (required): The path of the item to move.
    - destination (required): The new path for the item.
</define_tag>
""".strip()

DEFINE_DELETE_ITEM = """
<define_tag name="delete_item">
Deletes a file or an entire directory recursively.
Attributes:
    - path (required): The path of the item to delete.
</define_tag>
""".strip()


# --- Base Class for Filesystem Tools ---

class FileSystemTool(tool.BaseTool):
    """
    An abstract base class for tools that interact with the filesystem,
    providing shared security and execution logic.
    """
    definition = ""  # Should be overridden by subclasses

    def __init__(self, root_path="."):
        super().__init__()  # BaseToolの__init__を呼び出す
        self.root_path = os.path.abspath(root_path)
        logger.info(
            f"{self.__class__.__name__} initialized with root: {self.root_path}"
        )

    def _get_safe_path(self, path: str) -> str:
        """
        Validates and returns an absolute path within the root directory.
        Raises PermissionError if the path is outside the root directory.
        """
        if not path:
            raise ValueError("Path attribute cannot be empty.")

        safe_path = os.path.abspath(os.path.join(self.root_path, path))
        if not safe_path.startswith(self.root_path):
            raise PermissionError(
                "Access denied: Path is outside the allowed root directory."
            )
        return safe_path

    async def run(self, element: lpml.Element):
        """
        Generic async run handler that wraps synchronous file operations.
        Results are put into the system's result queue.
        """
        attributes = element.get("attributes", {})
        try:
            # For production, blocking I/O should run in a separate thread
            # via asyncio.to_thread to avoid blocking the event loop.
            result_content = self._sync_logic(element)
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}", exc_info=True)
            result_content = f"Error: An unexpected error occurred. {e}"

        output_element = lpml.generate_element(
            "output", "\n" + result_content + "\n",
            tool=self.name, **attributes
        )
        # self.system 経由で結果キューにアクセスする
        await self.system.result_queue.put(output_element)

    def _sync_logic(self, element: lpml.Element) -> str:
        """
        Placeholder for the synchronous logic of the tool.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement the _sync_logic method."
        )


# --- Tool Implementations ---

class ListFilesTool(FileSystemTool):
    name = "list_files"
    definition = DEFINE_LIST_FILES

    def _sync_logic(self, element: lpml.Element) -> str:
        path = element.get("attributes", {}).get("path")
        if path is None:
            return "Error: 'path' attribute is missing."

        target_path = self._get_safe_path(path)
        if not os.path.isdir(target_path):
            return f"Error: Path is not a directory - '{path}'"

        files = os.listdir(target_path)
        return '\n'.join(files) if files else f"Directory '{path}' is empty."


class ReadFileTool(FileSystemTool):
    name = "read_file"
    definition = DEFINE_READ_FILE

    def _sync_logic(self, element: lpml.Element) -> str:
        attributes = element.get("attributes", {})
        path = attributes.get("path")
        show_lines = attributes.get("line_numbers", "false").lower() == "true"

        if path is None:
            return "Error: 'path' attribute is missing."

        target_path = self._get_safe_path(path)
        if not os.path.isfile(target_path):
            return f"Error: Path is not a file - '{path}'"

        with open(target_path, 'r', encoding='utf-8') as f:
            if show_lines:
                lines = f.readlines()
                return "".join(f"{i+1}: {line}" for i, line in enumerate(lines))
            return f.read()


class WriteFileTool(FileSystemTool):
    name = "write_file"
    definition = DEFINE_WRITE_FILE

    def _sync_logic(self, element: lpml.Element) -> str:
        # (The complex logic from the previous version is moved here)
        # ... (implementation is long, so it's placed at the end for clarity)
        return self._write_logic(element)

    def _write_logic(self, element: lpml.Element) -> str:
        # ... (Implementation from previous response)
        attributes = element.get("attributes", {})
        path = attributes.get("path")
        mode = attributes.get("mode", "overwrite")
        content = element.get("content", "")
        content = lpml.deparse(content).strip("\n")

        if path is None:
            return "Error: 'path' attribute is missing."

        target_path = self._get_safe_path(path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if mode == "overwrite":
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully overwrote file '{path}'."

        if mode == "append":
            with open(target_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully appended to file '{path}'."

        try:
            lines = []
            if os.path.exists(target_path):
                with open(target_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

            if mode == "insert_at_line":
                line_num = int(attributes.get("line", 1)) - 1
                lines.insert(line_num, content + '\n')
            elif mode == "replace_lines":
                start = int(attributes.get("start_line", 1)) - 1
                end = int(attributes.get("end_line", start + 1)) - 1
                lines = lines[:start] + [content + '\n'] + lines[end+1:]
            else:
                return f"Error: Unknown write mode '{mode}'."

            with open(target_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return f"Successfully modified '{path}' with mode '{mode}'."

        except (ValueError, KeyError, IndexError) as e:
            return f"Error: Invalid attributes for mode '{mode}'. Details: {e}"


class CreateDirectoryTool(FileSystemTool):
    name = "create_directory"
    definition = DEFINE_CREATE_DIRECTORY

    def _sync_logic(self, element: lpml.Element) -> str:
        attributes = element.get("attributes", {})
        path = attributes.get("path")
        create_parents = attributes.get("parents", "false").lower() == "true"

        if path is None:
            return "Error: 'path' attribute is missing."

        target_path = self._get_safe_path(path)
        if os.path.exists(target_path):
            return f"Error: Path already exists - '{path}'"

        if create_parents:
            os.makedirs(target_path, exist_ok=True)
        else:
            if not os.path.exists(os.path.dirname(target_path)):
                return "Error: Parent directory does not exist. Use parents=\"true\"."
            os.mkdir(target_path)

        return f"Successfully created directory '{path}'."


class MoveItemTool(FileSystemTool):
    name = "move_item"
    definition = DEFINE_MOVE_ITEM

    def _sync_logic(self, element: lpml.Element) -> str:
        attributes = element.get("attributes", {})
        source = attributes.get("source")
        destination = attributes.get("destination")

        if not source or not destination:
            return "Error: 'source' and 'destination' attributes are required."

        source_path = self._get_safe_path(source)
        dest_path = self._get_safe_path(destination)

        if not os.path.exists(source_path):
            return f"Error: Source path does not exist - '{source}'"
        if os.path.exists(dest_path):
            return f"Error: Destination path already exists - '{destination}'"

        shutil.move(source_path, dest_path)
        return f"Successfully moved '{source}' to '{destination}'."


class DeleteItemTool(FileSystemTool):
    name = "delete_item"
    definition = DEFINE_DELETE_ITEM

    def _sync_logic(self, element: lpml.Element) -> str:
        path = element.get("attributes", {}).get("path")
        if path is None:
            return "Error: 'path' attribute is missing."

        target_path = self._get_safe_path(path)
        if not os.path.exists(target_path):
            return f"Error: Path does not exist - '{path}'"

        if os.path.isfile(target_path):
            os.remove(target_path)
            return f"Successfully deleted file '{path}'."
        elif os.path.isdir(target_path):
            shutil.rmtree(target_path)
            return f"Successfully deleted directory '{path}' and its contents."
        else:
            return f"Error: Path is not a file or directory - '{path}'"
