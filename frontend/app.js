document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const messageFeed = document.getElementById("message-feed");
    const newChatBtn = document.getElementById("new-chat-btn");
    const sessionIdDisplay = document.getElementById("session-id-display");
    const backendStatus = document.getElementById("backend-status");
    const statusDot = document.querySelector(".status-dot");
    const sendBtn = document.getElementById("send-btn");
    const btnText = sendBtn.querySelector(".btn-text");
    const btnLoader = sendBtn.querySelector(".loader");
    const btnIcon = sendBtn.querySelector("i");
    
    // Metrics panel elements
    const metricsPanel = document.getElementById("metrics-panel");
    const metricChunks = document.getElementById("metric-chunks");
    const metricTokens = document.getElementById("metric-tokens");
    const metricLatency = document.getElementById("metric-latency");

    // Session Setup
    let sessionId = localStorage.getItem("rag_session_id");
    if (!sessionId) {
        sessionId = generateSessionId();
        localStorage.setItem("rag_session_id", sessionId);
    }
    sessionIdDisplay.textContent = sessionId;

    // Initialize Lucide icons
    lucide.createIcons();

    // Check backend health
    checkHealth();

    // Handle Form Submit
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const message = chatInput.value.trim();
        if (!message) return;

        // Clear input field
        chatInput.value = "";
        
        // Hide welcome card if present
        const welcomeCard = document.querySelector(".welcome-card");
        if (welcomeCard) {
            welcomeCard.style.display = "none";
        }

        // Add user message to feed
        appendMessage("user", message);
        scrollToBottom();

        // Trigger loading state
        setLoading(true);

        const startTime = performance.now();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    sessionId: sessionId,
                    message: message
                })
            });

            const data = await response.json();
            const latencyMs = performance.now() - startTime;
            const latencySec = (latencyMs / 1000).toFixed(2);

            if (response.ok) {
                // Display assistant reply
                appendMessage("assistant", data.reply);
                
                // Show and update metrics dashboard
                metricsPanel.classList.remove("hidden");
                metricChunks.textContent = data.retrievedChunks;
                metricTokens.textContent = data.tokensUsed;
                metricLatency.textContent = `${latencySec}s`;
            } else {
                // Server returned an error payload
                const errorMsg = data.error || "Failed to get a response from assistant.";
                appendMessage("error", errorMsg);
            }
        } catch (err) {
            console.error("Network error:", err);
            appendMessage("error", "Network connection failure. Make sure the FastAPI server is running.");
        } finally {
            setLoading(false);
            scrollToBottom();
        }
    });

    // Handle Suggested Chips
    document.querySelectorAll(".suggested-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            chatInput.value = chip.textContent;
            chatInput.focus();
        });
    });

    // Handle New Chat Session
    newChatBtn.addEventListener("click", () => {
        sessionId = generateSessionId();
        localStorage.setItem("rag_session_id", sessionId);
        sessionIdDisplay.textContent = sessionId;

        // Clear feed and rebuild welcome card
        messageFeed.innerHTML = `
            <div class="welcome-card glass-card">
                <div class="welcome-icon"><i data-lucide="sparkles"></i></div>
                <h2>Welcome to the TrueAI Portal Assistant!</h2>
                <p>This assistant has access to the official corporate manuals. You can ask about:</p>
                <div class="suggested-chips">
                    <button class="suggested-chip">How do I reset my password?</button>
                    <button class="suggested-chip">What is the leave policy?</button>
                    <button class="suggested-chip">How can I request a monitor?</button>
                    <button class="suggested-chip">Can I work from home on Wednesdays?</button>
                </div>
            </div>
        `;
        
        // Re-bind click handlers for the new suggestion chips
        document.querySelectorAll(".suggested-chip").forEach(chip => {
            chip.addEventListener("click", () => {
                chatInput.value = chip.textContent;
                chatInput.focus();
            });
        });

        // Hide metrics panel
        metricsPanel.classList.add("hidden");
        
        // Recreate icons
        lucide.createIcons();
    });

    // Helper Functions
    function generateSessionId() {
        return Math.random().toString(36).substring(2, 8) + Date.now().toString(36).substring(4);
    }

    function appendMessage(role, text) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", role);
        
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        let avatarIcon = role === "user" ? "user" : "bot";
        if (role === "error") avatarIcon = "alert-triangle";

        // Create avatar
        const avatar = document.createElement("div");
        avatar.classList.add("msg-avatar");
        avatar.innerHTML = `<i data-lucide="${avatarIcon}"></i>`;
        messageDiv.appendChild(avatar);

        // Create body
        const bodyDiv = document.createElement("div");
        bodyDiv.classList.add("msg-body");

        const bubble = document.createElement("div");
        bubble.classList.add("msg-bubble");

        if (role === "assistant") {
            // Render markdown content safely
            bubble.innerHTML = marked.parse(text);
        } else {
            bubble.textContent = text;
        }
        bodyDiv.appendChild(bubble);

        const meta = document.createElement("div");
        meta.classList.add("msg-meta");
        const roleLabel = role === "user" ? "You" : (role === "error" ? "System Alert" : "Assistant");
        meta.innerHTML = `<span>${roleLabel}</span><span>•</span><span>${timestamp}</span>`;
        bodyDiv.appendChild(meta);

        messageDiv.appendChild(bodyDiv);
        messageFeed.appendChild(messageDiv);
        
        lucide.createIcons();
    }

    function scrollToBottom() {
        messageFeed.scrollTop = messageFeed.scrollHeight;
    }

    function setLoading(isLoading) {
        if (isLoading) {
            chatInput.disabled = true;
            sendBtn.disabled = true;
            btnText.classList.add("hidden");
            btnIcon.classList.add("hidden");
            btnLoader.classList.remove("hidden");
        } else {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            btnText.classList.remove("hidden");
            btnIcon.classList.remove("hidden");
            btnLoader.classList.add("hidden");
            chatInput.focus();
        }
    }

    async function checkHealth() {
        try {
            const res = await fetch("/health");
            const data = await res.json();
            if (res.ok && data.status === "healthy") {
                backendStatus.textContent = "Online";
                backendStatus.className = "status-value healthy";
                statusDot.className = "status-dot healthy";
            } else {
                setOffline();
            }
        } catch (err) {
            setOffline();
        }
    }

    function setOffline() {
        backendStatus.textContent = "Offline";
        backendStatus.className = "status-value failed";
        statusDot.className = "status-dot failed";
    }
});
