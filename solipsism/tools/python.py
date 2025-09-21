import io
import sys
import logging
import traceback
import asyncio
from contextlib import redirect_stdout, redirect_stderr

from solipsism.core import tool
from solipsism.core import lpml

logger = logging.getLogger(__name__)

DEFINE_PYTHON_REPL = """
<define_tag name="python">
Executes Python code in a stateful REPL session.
The content of the tag is the code to be executed.
Variables and imports are preserved across calls within the same context.
</define_tag>
""".strip()


class PythonReplTool(tool.BaseTool):
    """
    A tool that provides a stateful Python REPL (Read-Eval-Print Loop).
    It maintains a persistent namespace for each context instance.
    """
    name = "python"
    definition = DEFINE_PYTHON_REPL

    def __init__(self):
        super().__init__()
        # Each tool instance gets its own namespace to maintain state.
        self.namespace = {"__name__": "__console__"}
        logger.info("PythonReplTool initialized with a new namespace.")

    async def run(self, element: lpml.Element):
        """
        Asynchronously executes the Python code and puts the result in the queue.
        """
        attributes = element.get("attributes", {})
        try:
            # Run the blocking code in a separate thread to avoid blocking the event loop.
            result_content = await asyncio.to_thread(self._sync_logic, element)
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}", exc_info=True)
            result_content = f"Error: An unexpected error occurred. {e}"

        output_element = lpml.generate_element(
            "output", "\n" + result_content + "\n",
            tool=self.name, **attributes
        )
        await self.system.result_queue.put(output_element)

    def _sync_logic(self, element: lpml.Element) -> str:
        """
        Synchronously executes the Python code and captures its output.
        """
        code = element.get("content", "")
        if not code:
            return "Error: No code provided to execute."
        
        # Deparse the content to get the raw string
        code = lpml.deparse(code).strip()
        logger.info(f"Executing python code: {code[:200]}")

        # Redirect stdout and stderr to capture the output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Use exec to run the code in the persistent namespace
                exec(code, self.namespace)
        except Exception:
            # If an exception occurs, capture it
            # traceback.format_exc() provides a detailed error message
            stderr_capture.write(traceback.format_exc())

        stdout_val = stdout_capture.getvalue()
        stderr_val = stderr_capture.getvalue()

        output = ""
        if stdout_val:
            output += f"--- stdout ---\n{stdout_val}"
        if stderr_val:
            output += f"--- stderr ---\n{stderr_val}"
        
        return output if output else "[No output]"