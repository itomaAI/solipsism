import asyncio
import inspect
import json
import logging
import sys
from pathlib import Path
import copy
import xml.etree.ElementTree as ET

from solipsism.core.context import Context
from solipsism.core.llm import GeminiLLM
from solipsism.core.manager import Manager
from solipsism.core.system import System
from solipsism.core.tool import ToolManager
from solipsism.interface.user_context import UserContext

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    WEB_DEPENDENCIES_INSTALLED = True
except ImportError:
    WEB_DEPENDENCIES_INSTALLED = False

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
ws_connection_manager = ConnectionManager()

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logger = logging.getLogger(__name__)
    logging.getLogger("solipsism.core.context").setLevel(logging.WARNING)

    if not WEB_DEPENDENCIES_INSTALLED:
        logger.critical("Web UI dependencies not installed. Please run: pip install fastapi uvicorn python-multipart websockets 'fastapi[cors]'")
        return

    tool_manager = ToolManager(tool_directories=["./solipsism/tools", "./workspace/tools"])
    TOOL_CATALOG = tool_manager.get_all_tool_classes()
    manager = Manager(tool_catalog=TOOL_CATALOG)
    tool_init_args = {"manager": manager, "root_path": "./", "tool_manager": tool_manager}
    
    user_context = UserContext(context_id="user", parent_id=None)
    # --- FIX: Add conversation_history to UserContext to persist user messages ---
    user_context.conversation_history = []
    
    SendTool = TOOL_CATALOG.get("send")
    user_context.add_tool(SendTool(manager=tool_init_args["manager"]))
    manager.add_context(user_context)

    llm = GeminiLLM(model="gemini-2.5-pro", thinking_budget=-1)
    llm_system = System()
    base_tools = ["list_files", "read_file", "write_file", "create_directory", "move_item", "delete_item", "send", "create_context", "register_tool", "list_available_tools"]
    for tool_name in base_tools:
        tool_class = TOOL_CATALOG.get(tool_name)
        if tool_class:
            sig = inspect.signature(tool_class.__init__)
            params = {p.name: tool_init_args[p.name] for p in sig.parameters.values() if p.name in tool_init_args}
            instance = tool_class(**params)
            llm_system.add_tool(instance)
    
    llm_context = Context(llm=llm, system=llm_system, base_prompt_path="./solipsism/prompts/root_prompt.lpml", parent_id=user_context.id)
    user_context.child_ids.append(llm_context.id)
    manager.add_context(llm_context)

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    
    ui_path = Path("solipsism/interface/web_ui")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_connection_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                from_id, to_id, content = "user", payload.get("to"), payload.get("content")
                if to_id and content:
                    # Persist user message to their history before routing
                    user_message_turn = len(user_context.conversation_history) + 1
                    user_message = {
                        "tag": "user",
                        "attributes": {"from": from_id, "to": to_id, "turn": user_message_turn},
                        "content": content
                    }
                    user_context.conversation_history.append(user_message)
                    
                    message_element = {"tag": "send", "attributes": {"from": from_id, "to": to_id}, "content": content}
                    await manager.route_message(from_id, to_id, message_element)
        except WebSocketDisconnect:
            ws_connection_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error in WebSocket endpoint: {e}", exc_info=True)
            ws_connection_manager.disconnect(websocket)

    app.mount("/static", StaticFiles(directory=ui_path), name="static")
    @app.get("/")
    async def read_index():
        return FileResponse(ui_path / "index.html")

    async def push_history_updates():
        while True:
            await asyncio.sleep(1)
            if ws_connection_manager.active_connections:
                try:
                    # --- FIX: Pre-process history to add report flags ---
                    history_snapshot = {}
                    for cid, ctx in manager.contexts.items():
                        if hasattr(ctx, 'conversation_history'):
                            processed_history = []
                            for turn in ctx.conversation_history:
                                new_turn = copy.deepcopy(turn)
                                if new_turn.get('tag') == 'assistant':
                                    try:
                                        # ラッパーを追加してパースエラーを防ぐ
                                        xml_content = f"<root>{new_turn.get('content', '')}</root>"
                                        root = ET.fromstring(xml_content)
                                        reports = [elem.text.strip() for elem in root.findall(".//send[@to='user']")]
                                        if reports:
                                            new_turn['reports'] = reports
                                    except ET.ParseError:
                                        new_turn['reports'] = [] # パース失敗時は空リスト
                                processed_history.append(new_turn)
                            history_snapshot[cid] = processed_history

                    if history_snapshot:
                        await ws_connection_manager.broadcast(json.dumps(history_snapshot, default=str))
                except Exception as e:
                    logger.error(f"Error in push_history_updates loop: {e}", exc_info=True)

    initial_task = f"You are an AI assistant. Your context ID is '{llm_context.id}'. Your parent is '{user_context.id}'. Introduce yourself."
    llm_task = llm_context.start(initial_task=initial_task, max_turns=1000, turn_sleep=1)
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    web_server_task = asyncio.create_task(server.serve())
    history_update_task = asyncio.create_task(push_history_updates())
    logger.info("Web UI is available at http://127.0.0.1:8000")
    await asyncio.gather(llm_task, web_server_task, history_update_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\nApplication interrupted by user. Exiting...")