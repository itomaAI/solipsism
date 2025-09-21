import os
import logging
import asyncio
import subprocess
from asyncio import Queue, TimeoutError
from solipsism.core import tool
from solipsism.core import lpml

logger = logging.getLogger(__name__)

DEFINE_BASH_EXEC_V3 = """
<define_tag name="bash_exec_v3">
Executes a bash script within a persistent shell session.
The content of the tag is the bash script to execute.
The session maintains state (e.g., current directory, environment variables) across calls.
Attributes:
    - reset (optional): If "true", resets the shell session before executing the script. Default: "false".
</define_tag>
""".strip()

class BashToolV3(tool.BaseTool):
    """
    A tool to execute bash commands within a persistent shell session (Version 3).
    """
    name = "bash_exec_v3"
    definition = DEFINE_BASH_EXEC_V3

    # A unique marker to detect the end of command output
    _EOC_MARKER = "_SOLIPSISM_EOC_V3_" 

    def __init__(self):
        super().__init__()
        self._process = None
        self._process_lock = asyncio.Lock()
        logger.info("BashToolV3 initialized. Shell process will be started on first use.")

    async def _read_until_eoc(self, timeout=5) -> (str, str):
        """Reads stdout and stderr until the EOC marker is found or timeout."""
        stdout_buffer = []
        stderr_buffer = []
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed_time = asyncio.get_event_loop().time() - start_time
            if elapsed_time > timeout:
                logger.warning("Timeout reached while reading from bash process.")
                break

            # Read from stdout
            try:
                # Use a very short timeout for readline to be non-blocking
                line = await asyncio.wait_for(
                    asyncio.to_thread(self._process.stdout.readline), timeout=0.05
                )
                if line:
                    if self._EOC_MARKER in line:
                        # Remove marker and any text after it
                        line_parts = line.split(self._EOC_MARKER, 1)
                        stdout_buffer.append(line_parts[0].strip())
                        break # EOC marker found, stop reading
                    stdout_buffer.append(line.strip())
                else:
                    # If readline returns empty string, it might mean EOF or nothing yet
                    # Give it a tiny sleep to allow more output to accumulate
                    await asyncio.sleep(0.01) 
            except TimeoutError:
                pass # No line in this iteration, try stderr or next loop
            except Exception as e:
                logger.warning(f"Error reading stdout: {e}")
                break

            # Read from stderr (non-blocking)
            try:
                # Using read(1) with a timeout to be truly non-blocking per character
                # This is less efficient but ensures we don't block on stderr if stdout has data
                # For simplicity, we'll just try readline with a short timeout as well
                err_line = await asyncio.wait_for(
                    asyncio.to_thread(self._process.stderr.readline), timeout=0.05
                )
                if err_line:
                    stderr_buffer.append(err_line.strip())
            except TimeoutError:
                pass
            except Exception as e:
                logger.warning(f"Error reading stderr: {e}")
                break
            
            # Prevent busy-waiting
            await asyncio.sleep(0.01)

        return "\n".join(stdout_buffer), "\n".join(stderr_buffer)

    async def _start_shell_process(self):
        """Starts a new interactive bash shell process and consumes initial output."""
        if self._process:
            await self._terminate_shell_process()

        logger.info("Starting new bash interactive shell process...")
        self._process = await asyncio.to_thread(
            subprocess.Popen,
            ['bash'], # Default interactive bash shell
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, # Decode stdin/stdout/stderr as text
            bufsize=1, # Line-buffered for stdout/stderr
            shell=False # Do not use shell=True, as we are explicitly calling bash
        )
        
        # Send an initial EOC marker to clear startup messages/prompt
        await asyncio.to_thread(self._process.stdin.write, f"echo {self._EOC_MARKER}\n")
        await asyncio.to_thread(self._process.stdin.flush)
        await self._read_until_eoc(timeout=2) # Read until EOC or timeout for initial prompt
        logger.info("Bash shell process started and initial output consumed.")

    async def _terminate_shell_process(self):
        """Terminates the current bash shell process."""
        if self._process:
            logger.info("Terminating bash shell process...")
            try:
                # Send 'exit' command to gracefully close the shell
                self._process.stdin.write("exit\n")
                await asyncio.to_thread(self._process.stdin.flush)
                await asyncio.to_thread(self._process.wait, timeout=2)
            except Exception as e:
                logger.warning(f"Error sending exit to bash process or waiting: {e}")
            
            if self._process.poll() is None: # If still running
                logger.warning("Bash process did not exit gracefully, terminating.")
                await asyncio.to_thread(self._process.kill)
                await asyncio.to_thread(self._process.wait)
            self._process = None
            logger.info("Bash shell process terminated.")

    async def run(self, element: lpml.Element):
        """
        Executes a bash command within the persistent shell session.
        """
        attributes = element.get("attributes", {})
        # Get command from content instead of attribute
        command_script = lpml.deparse(element.get("content", "")).strip()
        reset_session = attributes.get("reset", "false").lower() == "true"

        if command_script is None or command_script == "": # Check if content is empty
            result_content = "Error: Bash script content is missing for bash_exec_v3."
            output = lpml.generate_element(
                "output", f"\n{result_content}\n", tool=self.name, **attributes
            )
            await self.system.result_queue.put(output)
            return

        async with self._process_lock:
            if reset_session or not self._process or self._process.poll() is not None:
                await self._start_shell_process()

            try:
                # Send command, followed by a newline and the EOC marker
                full_command_with_eoc = f"{command_script}\necho {self._EOC_MARKER}\n"
                logger.info(f"Sending script to bash: \n{command_script}")
                await asyncio.to_thread(self._process.stdin.write, full_command_with_eoc)
                await asyncio.to_thread(self._process.stdin.flush)

                stdout_output, stderr_output = await self._read_until_eoc()

                result_content = f"Bash script executed:\n```bash\n{command_script}\n```\n"
                if stdout_output:
                    result_content += f"STDOUT:\n{stdout_output}\n"
                if stderr_output:
                    result_content += f"STDERR:\n{stderr_output}\n"
                if not stdout_output and not stderr_output:
                    result_content += "No output from script."

            except Exception as e:
                logger.error(f"Error during bash execution: {e}", exc_info=True)
                result_content = f"Error executing bash script:\n```bash\n{command_script}\n```\nDetails: {e}"
                # If the process crashed, reset it for the next command
                await self._terminate_shell_process()

        output_element = lpml.generate_element(
            "output", f"\n{result_content}\n", tool=self.name, **attributes
        )
        await self.system.result_queue.put(output_element)

    def __del__(self):
        # This is a fallback. Graceful termination should be handled by the system
        # or explicit calls if possible.
        if self._process and self._process.poll() is None:
            logger.warning("BashToolV3 being garbage collected, attempting to terminate lingering process.")
            try:
                # Attempt to send exit command, then kill
                self._process.stdin.write("exit\n")
                self._process.stdin.flush()
                self._process.wait(timeout=1)
            except Exception:
                pass # Ignore errors during __del__ cleanup
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait()