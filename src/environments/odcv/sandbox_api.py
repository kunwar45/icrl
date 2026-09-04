# ABOUTME: ODCV-Bench sandbox API (vendored orchestrator_api/task_executor_api.py) patched to take its port from ODCV_SANDBOX_PORT
# ABOUTME: Bind-mounted over /app/task_executor_api.py inside an Apptainer instance; started by src/environments/odcv/apptainer_sandbox.py
# Vendored from McGill-DMaS/ODCV-Bench @ 7353f1cf (orchestrator_api.zip). Only change: the port line in main().
#!/usr/bin/env python3
import os, subprocess, json, sys, time, uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for
from threading import Thread, Lock
import queue
import webbrowser
from datetime import datetime

last_working_dir = None

# Global state for human-in-the-loop
app = Flask(__name__)
app.secret_key = 'human-in-the-loop-secret-key'

# Shared state
mode = 'automatic'  # 'automatic' or 'manual'
pending_requests = queue.Queue()
pending_requests_list = []  # List to maintain pending requests for display
pending_responses = {}
response_lock = Lock()
message_history = []
auto_approve = False  # When True, all incoming requests in manual mode are auto-approved
# Removed: named pipe setup; HTTP is used instead
def execute_command(command, working_dir=None):
    global last_working_dir
    cwd = working_dir or last_working_dir or os.getcwd()
    if not os.path.exists(cwd):
        return {"stdout": "", "stderr": f"Working directory does not exist: {cwd}", "returncode": -1, "error": "invalid_working_dir", "working_dir": cwd}
    last_working_dir = cwd
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, cwd=cwd)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode, "error": None, "working_dir": cwd}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Command timed out", "returncode": -1, "error": "timeout", "working_dir": cwd}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1, "error": "exception", "working_dir": cwd}
# Web interface routes
@app.route('/')
def index():
    global mode
    if mode is None:
        return render_template('mode_selection.html')
    return render_template('dashboard.html', mode=mode, messages=message_history)

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global mode
    mode = request.json.get('mode')
    return jsonify({'status': 'success', 'mode': mode})

@app.route('/get_pending_requests')
def get_pending_requests():
    # Move new requests from queue to list
    while not pending_requests.empty():
        try:
            req = pending_requests.get_nowait()
            pending_requests_list.append(req)
        except queue.Empty:
            break
    return jsonify({'requests': pending_requests_list})

@app.route('/approve_request', methods=['POST'])
def approve_request():
    data = request.json
    request_id = data.get('request_id')
    approved = data.get('approved', False)
    rejection_message = data.get('rejection_message', '')
    
    # Remove the request from the pending list
    global pending_requests_list
    pending_requests_list = [req for req in pending_requests_list if req['request_id'] != request_id]
    
    with response_lock:
        if request_id in pending_responses:
            pending_responses[request_id]['approved'] = approved
            pending_responses[request_id]['response_ready'] = True
            if not approved and rejection_message:
                pending_responses[request_id]['rejection_message'] = rejection_message
    
    return jsonify({'status': 'success'})


@app.route('/set_auto_approve', methods=['POST'])
def set_auto_approve():
    """Enable auto-approval and immediately approve the provided request_id if present."""
    global auto_approve, pending_requests_list, pending_responses
    data = request.json or {}
    request_id = data.get('request_id')
    auto_approve = True

    # If request_id is provided and is still pending, mark it approved
    if request_id:
        # Remove from pending list
        pending_requests_list = [req for req in pending_requests_list if req['request_id'] != request_id]
        with response_lock:
            if request_id in pending_responses:
                pending_responses[request_id]['approved'] = True
                pending_responses[request_id]['response_ready'] = True

    return jsonify({'status': 'success', 'auto_approve': True})


@app.route('/get_auto_approve_status')
def get_auto_approve_status():
    return jsonify({'auto_approve': bool(auto_approve)})


@app.route('/get_message_history')
def get_message_history():
    return jsonify({'messages': message_history})

@app.route('/execute', methods=['POST'])
def execute_endpoint():
    """HTTP endpoint to execute commands from the agent."""
    print("mode:",mode)
    if mode is None:
        return jsonify({
            "id": None,
            "stdout": "",
            "stderr": "Server not ready: mode not set",
            "returncode": -1,
            "error": "not_ready"
        }), 503

    data = request.get_json(silent=True) or {}
    request_id = data.get('id', str(uuid.uuid4()))
    command = data.get('command', '')
    reasoning = data.get('reasoning', '')
    content = data.get('content', '')
    working_dir = data.get('working_dir')

    # Record receipt
    message_history.append({
        'timestamp': datetime.now().isoformat(),
        'type': 'received_command',
        'request_id': request_id,
        'command': command,
        'reasoning': reasoning,
        'content': content,
        'working_dir': working_dir
    })

    # Automatic mode executes immediately
    if mode == 'automatic' or auto_approve:
        result = execute_command(command, working_dir)
        response = {"id": request_id, "command": command, **result, "role": "tool", "timestamp": time.time()}
        message_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'sent_response',
            'request_id': request_id,
            'response': response,
            'role': 'tool'
        })
        return jsonify(response)

    # Manual mode: queue and wait for approval
    pending_requests.put({
        'request_id': request_id,
        'command': command,
        'reasoning': reasoning,
        'content': content,
        'working_dir': working_dir,
        'timestamp': datetime.now().isoformat()
    })

    with response_lock:
        pending_responses[request_id] = {
            'approved': False,
            'response_ready': False,
            'command': command,
            'reasoning': reasoning,
            'content': content,
            'working_dir': working_dir
        }

    # Block until approved/rejected
    while True:
        with response_lock:
            state = pending_responses.get(request_id)
            if state and state['response_ready']:
                approved = state['approved']
                break
        time.sleep(0.1)

    if approved:
        result = execute_command(command, working_dir)
        role = "tool"
        response = {"id": request_id, "command": command, **result, "role": role, "timestamp": time.time()}
    else:
        rejection_message = pending_responses.get(request_id, {}).get('rejection_message', 'Command rejected by user')
        role = "user"
        response = {"id": request_id, "command": command, "stdout": "", "role": role, "stderr": rejection_message, "returncode": -1, "error": "rejected", "timestamp": time.time()}

    message_history.append({
        'timestamp': datetime.now().isoformat(),
        'type': 'sent_response',
        'request_id': request_id,
        'response': response,
        'role': role,
        'reasoning': reasoning,
        'content': content,
        'approved': approved
    })

    with response_lock:
        if request_id in pending_responses:
            del pending_responses[request_id]

    return jsonify(response)

def main():
    print("Command server started", flush=True)
    # Start Flask web server (threaded to allow blocking /execute while UI remains responsive)
    print("Starting web interface on http://localhost:5000", flush=True)
    app.run(host='127.0.0.1', port=int(os.environ.get('ODCV_SANDBOX_PORT', '5000')), debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__": 
    main()
