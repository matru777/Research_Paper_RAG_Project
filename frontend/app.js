// Utility: Generate UUID for Session
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0,
            v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

const SESSION_ID = generateUUID();
document.getElementById('session-id-display').textContent = SESSION_ID.substring(0, 8);

// DOM Elements
const uploadBtn = document.getElementById('upload-btn');
const docSourceInput = document.getElementById('document-source');
const uploadStatus = document.getElementById('upload-status');
const statusText = document.getElementById('status-text');
const docsUl = document.getElementById('docs-ul');

const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatHistory = document.getElementById('chat-history');

// Setup marked.js options
marked.setOptions({
    breaks: true,
    gfm: true
});

// --- Document Upload & Polling ---
uploadBtn.addEventListener('click', async () => {
    const source = docSourceInput.value.trim();
    if (!source) return;

    // UI Update
    uploadStatus.className = 'status-container'; // removes hidden, success, error
    uploadStatus.querySelector('.spinner').style.display = 'block';
    statusText.textContent = 'Submitting to worker...';
    uploadBtn.disabled = true;

    try {
        const response = await fetch('/api/v1/documents/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: source, session_id: SESSION_ID })
        });

        const data = await response.json();
        
        if (response.ok) {
            statusText.textContent = 'Processing Paper...';
            pollJobStatus(data.job_id, source);
        } else {
            throw new Error(data.detail || 'Failed to submit job');
        }
    } catch (error) {
        uploadStatus.className = 'status-container error';
        uploadStatus.querySelector('.spinner').style.display = 'none';
        statusText.textContent = 'Error: ' + error.message;
        uploadBtn.disabled = false;
    }
});

async function pollJobStatus(jobId, sourceName) {
    const pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/v1/documents/status/${jobId}`);
            const data = await res.json();
            
            if (data.status === 'COMPLETED') {
                clearInterval(pollInterval);
                uploadStatus.className = 'status-container success';
                uploadStatus.querySelector('.spinner').style.display = 'none';
                statusText.textContent = 'Ready!';
                uploadBtn.disabled = false;
                docSourceInput.value = '';
                
                // Add to sidebar list
                const li = document.createElement('li');
                li.textContent = sourceName.split('/').pop().split('\\').pop(); // rough basename
                docsUl.appendChild(li);
            } 
            else if (data.status === 'FAILED') {
                clearInterval(pollInterval);
                uploadStatus.className = 'status-container error';
                uploadStatus.querySelector('.spinner').style.display = 'none';
                statusText.textContent = 'Failed: ' + (data.error_message || 'Unknown error');
                uploadBtn.disabled = false;
            }
        } catch (e) {
            console.error("Polling error", e);
        }
    }, 2000); // poll every 2 seconds
}


// --- Chat with SSE ---
function addMessageToUI(content, isUser) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'user' : 'ai'}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = isUser ? 'U' : 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    
    if (isUser) {
        contentDiv.textContent = content; // raw text for user
    } else {
        // Markdown parsing for AI
        contentDiv.innerHTML = DOMPurify.sanitize(marked.parse(content));
    }
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);
    chatHistory.appendChild(msgDiv);
    
    // Scroll to bottom
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return contentDiv; // return so we can stream into it
}

async function sendChatMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Add User Message
    addMessageToUI(query, true);
    chatInput.value = '';
    sendBtn.disabled = true;

    // Create empty AI Message box to stream into
    const aiContentDiv = addMessageToUI('', false);
    aiContentDiv.innerHTML = '<span style="color:var(--text-secondary)">Thinking...</span>';

    try {
        const response = await fetch('/api/v1/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, session_id: SESSION_ID })
        });

        if (!response.ok) throw new Error("Failed to connect to chat stream");

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullText = "";
        let buffer = "";
        let isFirstToken = true;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.substring(6).trim();
                    if (dataStr === '[DONE]') {
                        break;
                    }
                    try {
                        const dataObj = JSON.parse(dataStr);
                        if (dataObj.token) {
                            if (isFirstToken) {
                                aiContentDiv.innerHTML = "";
                                isFirstToken = false;
                            }
                            fullText += dataObj.token;
                            // Re-render markdown on every token (fast enough for small streams)
                            aiContentDiv.innerHTML = DOMPurify.sanitize(marked.parse(fullText));
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        } else if (dataObj.error) {
                            fullText += `\n\n**Error:** ${dataObj.error}`;
                            aiContentDiv.innerHTML = DOMPurify.sanitize(marked.parse(fullText));
                        }
                    } catch (e) {
                        // might be a partial JSON chunk, standard SSE parsers handle this better, 
                        // but this simple split works for our controlled fastAPI stream
                    }
                }
            }
        }
    } catch (error) {
        aiContentDiv.innerHTML = `<span style="color:#ef4444">Connection Error: ${error.message}</span>`;
    } finally {
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

sendBtn.addEventListener('click', sendChatMessage);

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
});

// Auto-resize textarea
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if(this.value === '') this.style.height = '56px';
});
