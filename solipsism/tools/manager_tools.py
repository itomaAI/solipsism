import logging
from typing import TYPE_CHECKING

from ..core import tool
from ..core import lpml
from ..core.lpml import findall, deparse

if TYPE_CHECKING:
    from ..core.manager import Manager

logger = logging.getLogger(__name__)

DEFINE_SEND = """
<define_tag name="send">
Sends a message to a parent or a direct child context.
Communication with other contexts (e.g., siblings) is not allowed.
The user context 'user' is an exception and can be communicated with freely.
On success, no feedback is returned. On failure, an <output> tag with an error is returned.
Attributes:
    - to (required): The ID of the destination context, or the keyword "parent".
</define_tag>
""".strip()


class SendTool(tool.BaseTool):
    name = "send"
    definition = DEFINE_SEND

    def __init__(self, manager: 'Manager'):
        super().__init__()
        self.manager = manager

    async def run(self, element: lpml.Element):
        attributes = element.get("attributes", {})
        to_id_raw = attributes.get("to")
        from_id = self.system.context_id
        from_context = self.manager.get_context(from_id)

        error_content = None
        if not to_id_raw:
            error_content = "Error: The 'to' attribute is missing."
        elif not from_context:
            error_content = f"Error: Sender context '{from_id}' not found in Manager."

        if error_content:
            await self.system.result_queue.put(lpml.generate_element(
                "output", f"\n{error_content}\n", tool=self.name, **attributes
            ))
            return

        actual_to_id = None
        if to_id_raw == "parent":
            actual_to_id = from_context.parent_id
            if not actual_to_id:
                error_content = "Error: This context has no parent."
        else:
            actual_to_id = to_id_raw
        
        if error_content:
            await self.system.result_queue.put(lpml.generate_element(
                "output", f"\n{error_content}\n", tool=self.name, **attributes
            ))
            return

        message_to_send = lpml.generate_element(
            "send", element.get("content", ""), 
            **{'from': from_id, 'to': actual_to_id}
        )

        success = await self.manager.route_message(
            from_id=from_id,
            to_id=actual_to_id, 
            element=message_to_send
        )

        if not success:
            await self.system.result_queue.put(lpml.generate_element(
                "output", f"\nError: Failed to send message to '{actual_to_id}'. Not found or permission denied.\n", 
                tool=self.name, **attributes
            ))

DEFINE_CREATE_CONTEXT = """
<define_tag name="create_context">
Creates a new, independent LLM context as a child of the current context.
This tool returns the new child context's ID immediately and starts it in the background.
On failure, an <output> tag with an error is returned.
Attributes:
    - id (optional): A specific ID to assign to the new context. If not provided, one is generated.
Sub-tags:
    - <llm> (optional): Configures the LLM for the new context. Attributes: `model`, `temperature`.
    - <prompt> (optional): Specifies a custom base prompt file. Attribute: `path`.
    - <tools> (required): A list of tools to grant to the new context.
        - <tool name="..."/>: The name of the tool to grant (e.g., "read_file").
    - <task> (required): The initial task description for the new context.
</define_tag>
""".strip()


class CreateContextTool(tool.BaseTool):
    name = "create_context"
    definition = DEFINE_CREATE_CONTEXT

    def __init__(self, manager: 'Manager'):
        super().__init__()
        self.manager = manager

    async def run(self, element: lpml.Element):
        attributes = element.get("attributes", {})
        
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ここがバグのあった箇所です。二重パースをやめます。
        # contentは既にパース済みのLPMLツリー（リスト）なので、そのまま使います。
        content_tree = element.get("content")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        
        # content_treeがNoneや空でないことを確認
        if not content_tree or not isinstance(content_tree, list):
            await self.system.result_queue.put(lpml.generate_element(
                "output", "\nError: <create_context> tag must contain <tools> and <task> sub-tags.\n", tool=self.name
            ))
            return
            
        custom_id = attributes.get("id")
        
        task_element = findall(content_tree, "task")
        if not task_element:
            await self.system.result_queue.put(lpml.generate_element(
                "output", "\nError: <task> tag is required.\n", tool=self.name
            ))
            return
        task = deparse(task_element[0].get("content", "")).strip()

        tool_elements = findall(content_tree, "tool")
        tool_names = [t.get("attributes", {}).get("name") for t in tool_elements if t.get("attributes", {}).get("name")]
        
        llm_config = findall(content_tree, "llm")
        llm_config = llm_config[0].get("attributes") if llm_config else {}

        prompt_config = findall(content_tree, "prompt")
        prompt_path = prompt_config[0].get("attributes", {}).get("path") if prompt_config else None

        try:
            parent_id = self.system.context_id
            new_context = await self.manager.create_new_context(
                parent_id=parent_id,
                custom_id=custom_id,
                task=task,
                tool_names=tool_names,
                llm_config=llm_config,
                prompt_path=prompt_path,
            )
            if new_context:
                result_content = f"Successfully created new context. ID: {new_context.id}"
                output_element = lpml.generate_element(
                    "output", f"\n{result_content}\n", tool=self.name, status="success", id=new_context.id
                )
            else:
                raise Exception("Manager failed to create context for an unknown reason.")

        except Exception as e:
            logger.error(f"Failed to create context: {e}", exc_info=True)
            output_element = lpml.generate_element(
                "output", f"\nError: Failed to create context. {e}\n", tool=self.name, status="error"
            )
            
        await self.system.result_queue.put(output_element)
