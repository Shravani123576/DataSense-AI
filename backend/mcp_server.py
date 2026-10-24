#!/usr/bin/env python3
"""
DataSense AI MCP Server (JSON-RPC over stdio)
Exposes validated DataSense AI tools conforming to the Model Context Protocol standard.
"""
import sys
import json
import logging
from typing import Dict, Any, List

# Setup simple stderr logging so stdout is reserved for JSON-RPC messages
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("datasense.mcp")

# Tool Definitions
MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "profile_dataset",
        "description": "Examine dataset schema, compute health metrics, missing values, duplicates, and 1.5x IQR outliers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Active session dataset ID"}
            },
            "required": ["dataset_id"]
        }
    },
    {
        "name": "get_cleaning_recommendations",
        "description": "Generate evidence-backed data cleaning opportunities with affected row counts and rule strategies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Active session dataset ID"}
            },
            "required": ["dataset_id"]
        }
    },
    {
        "name": "apply_cleaning_action",
        "description": "Apply approved auditable transformation on dataset without destroying raw data, returning audit provenance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Active session dataset ID"},
                "rule": {"type": "string", "enum": ["deduplicate", "impute_missing", "normalize_categories", "cap_outliers"]},
                "column": {"type": "string", "description": "Target column (optional for deduplication)"},
                "strategy": {"type": "string", "description": "Specific strategy (e.g. median, mode, strip_and_titlecase, cap_iqr)"}
            },
            "required": ["dataset_id", "rule"]
        }
    },
    {
        "name": "train_automl",
        "description": "Train dummy baseline and candidate models (Random Forest, Logistic/Ridge, Decision Trees) with leakage-free pipeline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Active session dataset ID"},
                "target_column": {"type": "string", "description": "Feature to predict"},
                "task_type": {"type": "string", "enum": ["auto", "classification", "regression"], "default": "auto"}
            },
            "required": ["dataset_id", "target_column"]
        }
    },
    {
        "name": "what_if_predict",
        "description": "Run counterfactual what-if simulation against trained best model and compare against baseline median.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Active session dataset ID"},
                "target_column": {"type": "string", "description": "Target predicted feature"},
                "feature_inputs": {"type": "object", "description": "Dictionary of modified feature values"}
            },
            "required": ["dataset_id", "target_column", "feature_inputs"]
        }
    },
    {
        "name": "generate_executive_report",
        "description": "Synthesize data profiling, auditable cleaning log, and ML evaluation into executive Markdown & HTML report.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Active session dataset ID"},
                "objective": {"type": "string", "description": "Business or analytical objective"}
            },
            "required": ["dataset_id"]
        }
    }
]


def send_response(response: Dict[str, Any]):
    """Write JSON-RPC message to stdout followed by newline."""
    serialized = json.dumps(response)
    sys.stdout.write(serialized + "\n")
    sys.stdout.flush()


def handle_request(req: Dict[str, Any]):
    method = req.get("method")
    msg_id = req.get("id")

    if method == "initialize":
        send_response({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "datasense-ai-mcp",
                    "version": "2.0.0"
                }
            }
        })
        return

    if method == "tools/list":
        send_response({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": MCP_TOOLS
            }
        })
        return

    if method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        send_response({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "status": "ready",
                            "tool": tool_name,
                            "arguments": arguments,
                            "message": f"Tool '{tool_name}' successfully resolved in DataSense AI V2 runtime."
                        })
                    }
                ]
            }
        })
        return

    # Unknown method
    send_response({
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {
            "code": -32601,
            "message": f"Method '{method}' not found."
        }
    })


def main():
    logger.info("DataSense AI MCP Server started. Listening on stdio...")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            handle_request(req)
        except Exception as e:
            logger.error(f"Error handling MCP input: {e}")
            send_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            })


if __name__ == "__main__":
    main()
