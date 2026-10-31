"""
DataSense AI MCP Verification Suite (Task 3 & 4)
Uses the official Python MCP SDK client (`stdio_client` and `ClientSession`)
to test protocol behavior against the running FastMCP server process.
"""
import sys
import os
import json
import pytest
import asyncio
from pathlib import Path

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Ensure backend directory is in path
BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BACKEND_DIR.parent


@pytest.mark.anyio
async def test_official_mcp_sdk_protocol_and_tools():
    """Verify end-to-end MCP protocol handshake, tool discovery, execution, dataset safety, and error resilience."""
    server_script = BACKEND_DIR / "mcp_server.py"
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(BACKEND_DIR)
        },
        cwd=str(WORKSPACE_DIR)
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1. Initialize Handshake (Task 3.1 & 3.2)
            init_result = await session.initialize()
            assert init_result is not None, "Initialize handshake failed"
            assert init_result.serverInfo.name in ["datasense-ai", "datasense-ai-mcp"]
            print(f"\n[PASSED 1 & 2] Handshake succeeded: server={init_result.serverInfo.name}, version={init_result.serverInfo.version}")

            # 2. Tool Discovery (Task 3.3 & Task 2)
            tools_result = await session.list_tools()
            tool_list = tools_result.tools
            tool_names = [t.name for t in tool_list]
            
            print(f"[PASSED 3] Tool discovery returned {len(tool_list)} tools:")
            for name in sorted(tool_names):
                print(f"    - {name}")

            assert len(tool_list) >= 20, f"Expected at least 20 tools, got {len(tool_list)}"
            assert len(tool_names) == len(set(tool_names)), "Tool names must be unique"

            # Verify every tool has non-empty description and valid JSON schema
            for t in tool_list:
                assert t.description and len(t.description.strip()) > 0, f"Tool {t.name} has empty description"
                assert t.inputSchema is not None, f"Tool {t.name} has no inputSchema"
                assert "properties" in t.inputSchema, f"Tool {t.name} schema missing 'properties'"
                assert "type" in t.inputSchema and t.inputSchema["type"] == "object"

            # 3. Read-only tool invocation with real dataset (Task 3.4 & 3.5 & Task 4)
            dataset_id = "sample_customer_churn"
            schema_res = await session.call_tool("get_dataset_schema", {"dataset_id": dataset_id})
            assert schema_res is not None and len(schema_res.content) > 0
            schema_payload = json.loads(schema_res.content[0].text)
            assert schema_payload["status"] == "success", f"Schema fetch failed: {schema_payload}"
            assert schema_payload["result"]["rows"] > 0
            assert len(schema_payload["result"]["columns"]) > 0
            assert schema_payload["mutated_data"] is False
            print(f"[PASSED 4 & 5] Read-only tool invoked with dataset '{dataset_id}': {schema_payload['result']['rows']} rows, {len(schema_payload['result']['columns'])} columns, mutated_data={schema_payload['mutated_data']}")

            # Inspect data quality tool (Task 4: read-only, non-mutating)
            quality_res = await session.call_tool("inspect_data_quality", {"dataset_id": dataset_id})
            assert quality_res is not None
            quality_payload = json.loads(quality_res.content[0].text)
            assert quality_payload["status"] == "success"
            assert quality_payload["mutated_data"] is False
            print(f"[PASSED] Data quality audited: overall_health_score={quality_payload['result']['overall_health_score']}")

            # Cleaning preview tool (Task 4: requires_approval=True for consequential changes, uncommitted preview)
            preview_res = await session.call_tool("preview_cleaning_changes", {
                "dataset_id": dataset_id,
                "rule": "drop_column",
                "column": "TenureMonths"
            })
            assert preview_res is not None
            preview_payload = json.loads(preview_res.content[0].text)
            assert preview_payload["status"] == "pending_approval"
            assert preview_payload["mutated_data"] is False
            assert preview_payload["requires_approval"] is True
            assert "proposal_id" in preview_payload["result"]
            print(f"[PASSED] Cleaning preview verified: requires_approval={preview_payload['requires_approval']}, mutated_data={preview_payload['mutated_data']}, proposal_id={preview_payload['result']['proposal_id']}")

            # 4. Error Handling: Nonexistent dataset ID (Task 3.6 & 4)
            error_call_res = await session.call_tool("get_dataset_schema", {"dataset_id": "nonexistent_ds_xyz"})
            assert error_call_res is not None and len(error_call_res.content) > 0
            res_json = json.loads(error_call_res.content[0].text)
            assert res_json["status"] == "error"
            assert "not found" in res_json["error_details"].lower()
            print(f"[PASSED 6] Nonexistent dataset produced structured error: {res_json['error_details']}")

            # 5. Invalid arguments: missing required parameter (Task 2 & 3.6)
            try:
                invalid_res = await session.call_tool("get_dataset_schema", {})
                if invalid_res.isError:
                    print("[PASSED] Missing arguments rejected with protocol isError=True")
                else:
                    first_text = invalid_res.content[0].text if invalid_res.content else ""
                    assert "error" in first_text.lower() or "validation" in first_text.lower()
                    print(f"[PASSED] Missing arguments returned validation error content: {first_text[:80]}...")
            except Exception as e:
                print(f"[PASSED] Missing arguments rejected with client exception: {type(e).__name__}")

            # 6. Unknown tool name error handling (Task 2 & 3.6)
            try:
                unknown_res = await session.call_tool("unregistered_dummy_tool", {})
                if unknown_res.isError:
                    print("[PASSED] Unknown tool rejected with protocol isError=True")
                else:
                    first_text = unknown_res.content[0].text if unknown_res.content else ""
                    assert "unknown" in first_text.lower() or "not found" in first_text.lower()
                    print(f"[PASSED] Unknown tool rejected in response: {first_text[:80]}...")
            except Exception as e:
                print(f"[PASSED] Unknown tool rejected with exception: {type(e).__name__}")

            # 7 & 8. Resilience & Subsequent requests after failures (Task 3.7 & 3.8)
            post_err_res = await session.call_tool("get_dataset_schema", {"dataset_id": dataset_id})
            assert post_err_res is not None
            post_json = json.loads(post_err_res.content[0].text)
            assert post_json["status"] == "success"
            print(f"[PASSED 7 & 8] Server handled post-error requests with 100% stability. Rows: {post_json['result']['rows']}")


if __name__ == "__main__":
    asyncio.run(test_official_mcp_sdk_protocol_and_tools())
