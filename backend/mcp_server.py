#!/usr/bin/env python3
"""
DataSense AI MCP Server (Phase 3 / Focused MCP Resolution)
Exposes validated DataSense AI data-science tools conforming to the Model Context Protocol standard.
Uses the official Python MCP SDK's FastMCP server implementation, with the central
ToolRegistry (default_registry) remaining the single source of truth for tool implementations.
"""
import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional

# Ensure the backend directory is in sys.path regardless of execution CWD
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Setup stderr logging so stdout is strictly reserved for JSON-RPC messages
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("datasense.mcp")

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, ArgModelBase

from tools import default_registry, ToolExecutionResult
from main import (
    DATASET_STORAGE,
    DATASET_VERSIONS_STORAGE,
    PROPOSED_TRANSFORMATIONS_STORAGE,
    AUDIT_LOG_STORAGE,
    PROFILE_STORAGE,
    METADATA_STORAGE,
    LAST_ML_STORAGE,
    build_tool_context
)


def get_mcp_tool_manifest() -> List[Dict[str, Any]]:
    """Convert registered tools into official MCP Tool representation metadata."""
    raw_tools = default_registry.list_tools()
    mcp_tools = []
    for t in raw_tools:
        mcp_tools.append({
            "name": t["name"],
            "description": f"[{t['category'].upper()} | {t['permission'].upper()}] {t['description']}",
            "inputSchema": t["input_schema"]
        })
    return mcp_tools


def create_fastmcp_server() -> FastMCP:
    """
    Construct a FastMCP server instance using the official Python MCP SDK.
    Dynamically registers all data-science tools from the central default_registry
    without duplicating analytical logic.
    """
    mcp_app = FastMCP(
        name="datasense-ai",
        instructions="DataSense AI Model Context Protocol server exposing data-science, cleaning, profiling, AutoML, and statistical tools."
    )

    for tool_info in default_registry.list_tools():
        t_name = tool_info["name"]
        t_desc = f"[{tool_info['category'].upper()} | {tool_info['permission'].upper()}] {tool_info['description']}"
        tool_obj = default_registry.get_tool(t_name)
        if not tool_obj:
            continue

        def make_tool_fn(target_tool_name: str):
            def tool_fn(**kwargs):
                dataset_id = kwargs.get("dataset_id")
                context = build_tool_context(dataset_id)
                res: ToolExecutionResult = default_registry.execute_tool(target_tool_name, kwargs, context)
                return res.model_dump() if hasattr(res, "model_dump") else res.dict()
            return tool_fn

        # Subclass the tool's input schema with ArgModelBase so FastMCP can validate and describe arguments
        tool_arg_model = type(f"{t_name}_Args", (tool_obj.input_schema, ArgModelBase), {})
        fn_meta = FuncMetadata(arg_model=tool_arg_model)

        tool = Tool(
            fn=make_tool_fn(t_name),
            name=t_name,
            title=t_name.replace("_", " ").title(),
            description=t_desc,
            parameters=tool_arg_model.model_json_schema(by_alias=True),
            fn_metadata=fn_meta,
            is_async=False
        )

        mcp_app._tool_manager._tools[t_name] = tool

    return mcp_app


def main():
    """Server entry point running official FastMCP with standard stdio transport."""
    logger.info("Initializing DataSense AI FastMCP Server...")
    mcp_app = create_fastmcp_server()
    logger.info(f"FastMCP initialized with {len(mcp_app._tool_manager._tools)} tools. Starting stdio transport...")
    mcp_app.run(transport="stdio")


if __name__ == "__main__":
    main()
