import os
import logging
from asyncio import Queue
from ..core import tool
from ..core import lpml


logger = logging.getLogger(__name__)


DEFINE_LIST_FILES = """
<define_tag>
Lists all files and directories at the given path.
Attributes:
    - path (optional) : The directory path. Default is root.
</define_tag>
""".strip()

DEFINE_READ_FILE = """
<define_tag name="read_file">
Reads the content of a specified file.
Attributes:
    - path (required): The path to the file.
    - line_numbers (optional): If "true", prepends line numbers to the output. Defaults to "false".
</define_tag>
""".strip()

DEFINE_WRITE_FILE = """
<define_tag name="write_file">
Writes or modifies a file. The content to be written is placed inside the tag.
Attributes:
    - path (required): The path to the file.
    - mode (optional): The writing mode. Can be "overwrite", "append", "replace_lines", or "insert_at_line". Defaults to "overwrite".
    - start_line (for replace_lines): The starting line number to replace (inclusive, 1-indexed).
    - end_line (for replace_lines): The ending line number to replace (inclusive, 1-indexed).
    - line (for insert_at_line): The line number at which to insert the content (1-indexed).
Example:
    <write_file path="./memo.txt" mode="append">This is a new line.</write_file>
    <write_file path="./config.py" mode="replace_lines" start_line="5" end_line="5">API_KEY = "new_key"</write_file>
</define_tag>
""".strip()


class ListFilesTool(tool.BaseTool):

    name = "list_files"
    definition = DEFINE_LIST_FILES

    def __init__(self, root_path="."):
        # セキュリティのため、実際のルートパスからの相対パスとして動作させる
        self.root_path = os.path.abspath(root_path)
        logger.info(
            f"ListFilesTool initialized with root path: {
                self.root_path}")

    def list_files_sync(self, path_attr: str) -> str:
        """同期的にファイルリストを取得する内部メソッド"""
        target_path = os.path.abspath(os.path.join(self.root_path, path_attr))
        if not target_path.startswith(self.root_path):
            raise PermissionError(
                "Access denied: Path is outside the allowed root directory.")

        if not os.path.exists(target_path):
            return f"Error: Path does not exist - '{path_attr}'"

        if not os.path.isdir(target_path):
            return f"Error: Path is not a directory - '{path_attr}'"

        try:
            files = os.listdir(target_path)
            if not files:
                return f"Directory '{path_attr}' is empty."
            return '\n'.join(files)
        except Exception as e:
            return f"Error listing files in '{path_attr}': {e}"

    async def run(self, element: lpml.Element, result_queue: Queue):
        attributes = element.get("attributes", {})
        path = attributes.get("path")

        if path is None:
            result_content = "Error: 'path' attribute is missing."
        else:
            try:
                # I/O処理はブロッキングなので、別スレッドで実行するのが望ましい
                # ここでは簡略化のため直接呼び出します
                result_content = '\n' + self.list_files_sync(path) + '\n'
            except Exception as e:
                logger.error(f"Error in ListFilesTool: {e}", exc_info=True)
                result_content = f"Error: {e}"

        # 結果をoutputタグを持つElementとしてキューに入れる
        output_element = lpml.generate_element(
            tag="output",
            content=result_content,
            tool=self.name,
            **attributes  # 元の属性や追加ラベルなど全部返してやる
        )
        await result_queue.put(output_element)


class ReadFileTool(tool.BaseTool):
    """Reads content from a specified file."""

    name = "read_file"
    definition = DEFINE_READ_FILE

    def __init__(self, root_path="."):
        self.root_path = os.path.abspath(root_path)
        logger.info(
            f"{self.__class__.__name__} initialized with root: {self.root_path}")

    def _get_safe_path(self, path: str) -> str:
        safe_path = os.path.abspath(os.path.join(self.root_path, path))
        if not safe_path.startswith(self.root_path):
            raise PermissionError(
                "Access denied: Path is outside the allowed root directory.")
        return safe_path

    async def run(self, element: lpml.Element, result_queue: Queue):
        attributes = element.get("attributes", {})
        path = attributes.get("path")
        show_line_numbers = attributes.get(
            "line_numbers", "false").lower() == "true"

        if path is None:
            result_content = "Error: 'path' attribute is missing."
        else:
            try:
                target_path = self._get_safe_path(path)
                if not os.path.exists(target_path):
                    result_content = f"Error: File not found - '{path}'"
                elif not os.path.isfile(target_path):
                    result_content = f"Error: Path is not a file - '{path}'"
                else:
                    with open(target_path, 'r', encoding='utf-8') as f:
                        if show_line_numbers:
                            lines = f.readlines()
                            result_content = "".join(
                                f"{i + 1}: {line}" for i, line in enumerate(lines))
                        else:
                            result_content = f.read()
            except Exception as e:
                logger.error(f"Error in {self.name}: {e}", exc_info=True)
                result_content = f"Error: {e}"

        output_element = lpml.generate_element(
            "output", result_content, tool=self.name, **attributes)
        await result_queue.put(output_element)


class WriteFileTool(tool.BaseTool):
    """Writes content to a specified file with various modes."""

    name = "write_file"
    definition = DEFINE_WRITE_FILE

    def __init__(self, root_path="."):
        self.root_path = os.path.abspath(root_path)
        logger.info(
            f"{self.__class__.__name__} initialized with root: {self.root_path}")

    def _get_safe_path(self, path: str) -> str:
        safe_path = os.path.abspath(os.path.join(self.root_path, path))
        if not safe_path.startswith(self.root_path):
            raise PermissionError(
                "Access denied: Path is outside the allowed root directory.")
        return safe_path

    def _write_sync(self, element: lpml.Element) -> str:
        """Synchronous internal method to handle all write logic."""
        attributes = element.get("attributes", {})
        path = attributes.get("path")
        mode = attributes.get("mode", "overwrite")
        content = element.get("content", "")

        if path is None:
            return "Error: 'path' attribute is missing."

        target_path = self._get_safe_path(path)

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if mode == "overwrite":
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully overwrote {len(content)} characters to '{path}'."

        elif mode == "append":
            with open(target_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully appended {len(content)} characters to '{path}'."

        elif mode in ["replace_lines", "insert_at_line"]:
            try:
                if not os.path.exists(target_path):
                    if mode == 'insert_at_line' and int(
                            attributes.get('line', 1)) == 1:
                        lines = []  # Treat as inserting into an empty file
                    else:
                        return f"Error: File not found for mode '{mode}' - '{path}'"
                else:
                    with open(target_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                if mode == "insert_at_line":
                    line_num = int(attributes.get("line", 1)) - \
                        1  # 1-indexed to 0-indexed
                    if not (0 <= line_num <= len(lines)):
                        return f"Error: Line number {line_num + 1} is out of bounds for insertion."
                    lines.insert(line_num, content + '\n')

                elif mode == "replace_lines":
                    start = int(attributes.get("start_line", 1)) - 1
                    end = int(attributes.get("end_line", start + 1)) - 1
                    if not (0 <= start <= end < len(lines)):
                        return f"Error: Line range {start + 1}-{end + 1} is out of bounds."
                    lines = lines[:start] + [content + '\n'] + lines[end + 1:]

                with open(target_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                return f"Successfully modified '{path}' with mode '{mode}'."

            except (ValueError, KeyError) as e:
                return f"Error: Missing or invalid line number attribute for mode '{mode}'. Details: {e}"
        else:
            return f"Error: Unknown write mode '{mode}'."

    async def run(self, element: lpml.Element, result_queue: Queue):
        attributes = element.get("attributes", {})
        try:
            # Note: For a production system, this blocking I/O should be
            # run in a separate thread via asyncio.to_thread
            result_content = self._write_sync(element)
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}", exc_info=True)
            result_content = f"Error: An unexpected error occurred. {e}"

        output_element = lpml.generate_element(
            "output", result_content, tool=self.name, **attributes)
        await result_queue.put(output_element)
