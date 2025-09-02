import logging
import inspect
from ..core import tool
from ..core import lpml
from ..core.tool import ToolManager

logger = logging.getLogger(__name__)


DEFINE_REGISTER_TOOL = """
<define_tag name="register_tool">
Dynamically discovers and registers a new tool for the current context to use.
The tool must be defined in a .py file within the project's tool directories.
On success, an <output> tag is returned. On failure, the <output> tag contains an error.
Attributes:
    - name (required): The name of the tool to register (e.g., "read_file").
</define_tag>
""".strip()

DEFINE_LIST_AVAILABLE_TOOLS = """
<define_tag name="list_available_tools">
Lists all tools available for registration in the tool catalog.
This command automatically re-scans the tool directories to find the latest tools before listing them.
</define_tag>
""".strip()


class RegisterTool(tool.BaseTool):
    # このクラスは変更なし
    name = "register_tool"
    definition = DEFINE_REGISTER_TOOL

    def __init__(self, tool_manager: ToolManager, **kwargs):
        super().__init__()
        self.tool_manager = tool_manager
        self.tool_init_args = kwargs

    async def run(self, element: lpml.Element):
        # ... (実装は変更なし)
        attributes = element.get("attributes", {})
        tool_to_register = attributes.get("name")

        if not tool_to_register:
            error_content = "Error: 'name' attribute is missing."
            output = lpml.generate_element(
                "output", f"\n{error_content}\n", tool=self.name, **attributes
            )
            await self.system.result_queue.put(output)
            return

        tool_class = self.tool_manager.get_tool_class(tool_to_register)

        if not tool_class:
            error_content = f"Error: Tool '{tool_to_register}' not found in the catalog. Did you run 'list_available_tools' to refresh the catalog after creating the tool file?"
            output = lpml.generate_element(
                "output", f"\n{error_content}\n", tool=self.name, **attributes
            )
            await self.system.result_queue.put(output)
            return

        if tool_to_register in self.system.tools:
            error_content = f"Error: Tool '{tool_to_register}' is already registered."
            output = lpml.generate_element(
                "output", f"\n{error_content}\n", tool=self.name, **attributes
            )
            await self.system.result_queue.put(output)
            return

        try:
            sig = inspect.signature(tool_class.__init__)
            params = {}
            full_init_args = {"tool_manager": self.tool_manager, **self.tool_init_args}

            for param in sig.parameters.values():
                if param.name == 'self':
                    continue
                if param.name in full_init_args:
                    params[param.name] = full_init_args[param.name]
            
            new_tool_instance = tool_class(**params)
            self.system.add_tool(new_tool_instance)
            
            success_content = f"Successfully registered tool '{tool_to_register}'."
            output = lpml.generate_element(
                "output", f"\n{success_content}\n", tool=self.name, **attributes
            )
        except Exception as e:
            logger.error(
                f"Failed to instantiate/register tool '{tool_to_register}': {e}", exc_info=True
            )
            error_content = f"Error: Could not instantiate tool '{tool_to_register}'. {e}"
            output = lpml.generate_element(
                "output", f"\n{error_content}\n", tool=self.name, **attributes
            )
            
        await self.system.result_queue.put(output)


class ListAvailableToolsTool(tool.BaseTool):
    name = "list_available_tools"
    definition = DEFINE_LIST_AVAILABLE_TOOLS

    def __init__(self, tool_manager: ToolManager):
        super().__init__()
        self.tool_manager = tool_manager

    async def run(self, element: lpml.Element):
        # ▼▼▼ 修正箇所 ▼▼▼
        # ツールをリストアップする前に、必ずツールカタログをリフレッシュする
        logger.info("Refreshing tool catalog as part of list_available_tools...")
        self.tool_manager.discover_tools()
        # ▲▲▲ 修正箇所 ▲▲▲

        tool_names = self.tool_manager.get_all_tool_classes().keys()
        content = "\n".join(sorted(tool_names))
        output = lpml.generate_element(
            "output", f"\n{content}\n", tool=self.name
        )
        await self.system.result_queue.put(output)
