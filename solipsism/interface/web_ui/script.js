document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM fully loaded. Initializing script v5...");
    const statusLight = document.getElementById("status-light");
    const contextList = document.getElementById("context-list");
    const logContainer = document.getElementById("log-content");
    const currentContextIdElem = document.getElementById("current-context-id");
    const messageInput = document.getElementById("message-input");
    const sendButton = document.getElementById("send-button");

    let allHistory = {};
    let selectedContextId = null;
    let ws;

    function connectWebSocket() {
        ws = new WebSocket(`ws://${window.location.host}/ws`);
        ws.onopen = () => { statusLight.className = "connected"; statusLight.title = "Connected"; };
        ws.onmessage = (event) => {
            try {
                const newHistoryData = JSON.parse(event.data);
                // Update the entire log based on server's source of truth
                allHistory = newHistoryData;
                renderContextList();
                renderFullLog(); // Re-render fully to ensure consistency
            } catch (e) {
                console.error("Failed to parse JSON from server:", e, event.data);
            }
        };
        ws.onerror = (error) => { console.error("WebSocket error:", error); };
        ws.onclose = () => {
            statusLight.className = "disconnected";
            statusLight.title = "Disconnected. Retrying...";
            setTimeout(connectWebSocket, 3000);
        };
    }

    function appendMessage(element) {
        const wasScrolledToBottom = logContainer.scrollHeight - logContainer.clientHeight <= logContainer.scrollTop + 1;

        const tag = element.tag;
        const content = element.content || "";
        
        const messageElem = document.createElement("div");
        messageElem.className = `message ${tag}`;

        // --- FIX: Use the reliable server-side flag ---
        if (element.is_report) {
            messageElem.classList.add('report-to-user');
        }

        const header = document.createElement("strong");
        const turn = element.attributes.turn || "?";
        header.textContent = `Turn ${turn} - <${tag}>`;
        messageElem.appendChild(header);

        const contentPre = document.createElement("pre");
        contentPre.textContent = content.trim();
        messageElem.appendChild(contentPre);

        logContainer.appendChild(messageElem);

        if (wasScrolledToBottom) {
            logContainer.scrollTop = logContainer.scrollHeight;
        }
    }

    function renderFullLog() {
        if (!selectedContextId || !allHistory[selectedContextId]) {
            logContainer.innerHTML = "<div>Select a context to view its log.</div>";
            return;
        }
        // Preserve scroll position if not at bottom
        const wasScrolledToBottom = logContainer.scrollHeight - logContainer.clientHeight <= logContainer.scrollTop + 1;
        const oldScrollTop = logContainer.scrollTop;

        logContainer.innerHTML = "";
        allHistory[selectedContextId].forEach(element => appendMessage(element));
        
        if (wasScrolledToBottom) {
            logContainer.scrollTop = logContainer.scrollHeight;
        } else {
            logContainer.scrollTop = oldScrollTop;
        }
    }

    function renderContextList() {
        const contextIds = Object.keys(allHistory);
        contextList.innerHTML = "";
        contextIds.forEach(id => {
            const li = document.createElement("li");
            li.textContent = id;
            li.dataset.contextId = id;
            if (id === selectedContextId) li.classList.add("active");
            contextList.appendChild(li);
        });

        if (!selectedContextId && contextIds.length > 0) {
            const firstAgentContext = contextIds.find(id => id !== 'user');
            if (firstAgentContext) selectContext(firstAgentContext);
        }
    }

    function selectContext(contextId) {
        selectedContextId = contextId;
        currentContextIdElem.textContent = `Log for: ${contextId}`;
        renderContextList();
        renderFullLog();
    }

    contextList.addEventListener("click", (e) => {
        if (e.target.tagName === "LI") selectContext(e.target.dataset.contextId);
    });

    function sendMessage() {
        const message = messageInput.value.trim();
        if (message && selectedContextId && ws.readyState === WebSocket.OPEN) {
            // --- FIX: Remove optimistic rendering. Server is now the source of truth. ---
            ws.send(JSON.stringify({ to: selectedContextId, content: message }));
            messageInput.value = "";
        }
    }

    sendButton.addEventListener("click", sendMessage);
    messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    connectWebSocket();
});