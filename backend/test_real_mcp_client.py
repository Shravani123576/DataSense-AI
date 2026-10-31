"""
DataSense AI V6 — Task 3: Real MCP Client Process Communication Test
Spawns the actual backend/mcp_server.py process over stdio JSON-RPC,
verifies:
1. Process starts and listens on stdio.
2. 'initialize' handshake succeeds with protocolVersion 2024-11-05.
3. 'tools/list' discovers all registered data science tools.
4. 'tools/call' successfully routes and executes a tool via JSON-RPC.
"""
import sys
import json
import subprocess
import os

def test_real_mcp_process():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(backend_dir, "mcp_server.py")

    # Start mcp_server.py as a real subprocess communicating over stdin/stdout
    proc = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=backend_dir
    )

    try:
        # 1. Send 'initialize'
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "AntigravityTestClient", "version": "1.0.0"}
            }
        }
        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()

        init_resp_line = proc.stdout.readline()
        assert init_resp_line, "MCP server closed stdout without responding to initialize"
        init_resp = json.loads(init_resp_line)
        assert init_resp.get("id") == 1
        assert init_resp["result"]["protocolVersion"] == "2024-11-05"
        assert init_resp["result"]["serverInfo"]["name"] in ["datasense-ai", "datasense-ai-mcp"]
        print("[MCP Client Test] 1. Initialize Handshake PASSED")

        # 2. Send 'tools/list'
        list_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        proc.stdin.write(json.dumps(list_req) + "\n")
        proc.stdin.flush()

        list_resp_line = proc.stdout.readline()
        assert list_resp_line, "MCP server closed stdout without responding to tools/list"
        list_resp = json.loads(list_resp_line)
        assert list_resp.get("id") == 2
        tools = list_resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        assert len(tools) >= 20, f"Expected at least 20 tools, found {len(tools)}"
        assert "profile_dataset" in tool_names
        assert "generate_chart" in tool_names
        assert "analyze_chart" in tool_names
        assert "run_statistical_test" in tool_names
        assert "train_and_compare_models" in tool_names
        print(f"[MCP Client Test] 2. Tool Discovery PASSED ({len(tools)} tools discovered)")

        # 3. Send 'tools/call' for 'run_statistical_test'
        # Without dataset_id, it should return graceful validation error result rather than crashing
        call_req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "run_statistical_test",
                "arguments": {
                    "dataset_id": "non_existent_dataset",
                    "test_type": "two_sample_t",
                    "variable_x": "grp",
                    "variable_y": "val"
                }
            }
        }
        proc.stdin.write(json.dumps(call_req) + "\n")
        proc.stdin.flush()

        call_resp_line = proc.stdout.readline()
        assert call_resp_line, "MCP server closed stdout without responding to tools/call"
        call_resp = json.loads(call_resp_line)
        assert call_resp.get("id") == 3
        content = call_resp["result"]["content"]
        assert len(content) > 0
        parsed_result = json.loads(content[0]["text"])
        print("[DEBUG parsed_result]:", parsed_result)
        assert parsed_result["tool_name"] == "run_statistical_test"
        assert parsed_result["status"] == "error"
        assert "not found" in parsed_result["error_details"].lower()
        print("[MCP Client Test] 3. Tool Execution via JSON-RPC PASSED")

    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait(timeout=5)

if __name__ == "__main__":
    test_real_mcp_process()
    print("ALL MCP CLIENT TESTS PASSED SUCCESSFULLY.")
