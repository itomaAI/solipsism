import os
import ast
import logging
from solipsism.core import tool
from solipsism.core import lpml

logger = logging.getLogger(__name__)

DEFINE_GET_TOOL_DETAILS = """
<define_tag name="get_tool_details">
Scans all tool directories, parses the Python files, and extracts the name and LPML definition of every available tool. This tool takes no attributes.
</define_tag>
""".strip()


class GetToolDetailsTool(tool.BaseTool):
    """
    A tool to inspect and list the details of all available tools.
    """
    name = "get_tool_details"
    definition = DEFINE_GET_TOOL_DETAILS

    def _parse_tool_file(self, file_path):
        """Safely parses a Python file to find tool name and its LPML definition."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
            
            tool_name = None
            tool_def_var_name = None
            tool_def_content = None

            # First pass: Find the class to get the tool name and definition variable name
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    is_tool_class = any(
                        (isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == 'tool' and base.attr == 'BaseTool')
                        for base in node.bases
                    )
                    if not is_tool_class:
                        continue

                    for class_item in node.body:
                        if isinstance(class_item, ast.Assign) and isinstance(class_item.targets[0], ast.Name):
                            target_name = class_item.targets[0].id
                            if target_name == 'name' and isinstance(class_item.value, ast.Str):
                                tool_name = class_item.value.s
                            elif target_name == 'definition' and isinstance(class_item.value, ast.Name):
                                tool_def_var_name = class_item.value.id
                    
                    if tool_name and tool_def_var_name:
                        break
            
            # Second pass: Find the definition string itself at the module level
            if tool_def_var_name:
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                        if node.targets[0].id == tool_def_var_name and isinstance(node.value, ast.Str):
                            tool_def_content = node.value.s.strip()
                            break
            
            if tool_name and tool_def_content:
                return tool_name, tool_def_content # Return the full LPML definition string
        except Exception as e:
            logger.warning(f"Could not parse tool file {file_path}: {e}")
            return None, None
        return None, None

    async def run(self, element: lpml.Element):
        attributes = element.get("attributes", {})
        tool_dirs = ["./solipsism/tools", "./workspace/tools"]
        all_tool_details = []

        for tool_dir in tool_dirs:
            if not os.path.isdir(tool_dir):
                continue
            for filename in os.listdir(tool_dir):
                if filename.endswith(".py") and filename != "__init__.py":
                    file_path = os.path.join(tool_dir, filename)
                    name, definition_content = self._parse_tool_file(file_path)
                    if name and definition_content:
                        all_tool_details.append(f"Tool Name: {name}\nDefinition:\n{definition_content}")

        if not all_tool_details:
            result_content = "No tools found or parsed."
        else:
            result_content = "Available Tools and Definitions:\n\n" + "\n\n".join(sorted(all_tool_details))
        
        try:
            output_element = lpml.generate_element(
                "output", "\n" + result_content + "\n",
                tool=self.name
            )
            await self.system.result_queue.put(output_element)
        except Exception as e:
            error_message = f"Critical error sending get_tool_details output: {e}\nAttempting to log to file.\nContent:\n{result_content}"
            logger.error(error_message)
            
            my_context_id = "12b6e855" # My known context ID
            error_log_dir = f"./workspace/context/{my_context_id}/"
            os.makedirs(error_log_dir, exist_ok=True)
            error_log_path = os.path.join(error_log_dir, "get_tool_details_error.log")
            
            try:
                with open(error_log_path, "w", encoding="utf-8") as f:
                    f.write(error_message)
                logger.info(f"Successfully wrote error log to {error_log_path}")
            except Exception as file_e:
                logger.critical(f"Failed to write error log to file: {file_e}")