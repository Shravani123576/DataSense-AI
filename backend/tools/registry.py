"""
DataSense AI Tool Layer — Central Tool Registry (Phase 3)
Provides a maintainable internal registry for data-science tools,
handling input validation, execution routing, error handling, and schema exports.
"""
import logging
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, ValidationError
from tools.base import BaseTool, ToolExecutionResult, ToolExecutionStatus

logger = logging.getLogger("datasense.tools.registry")


class ToolRegistry:
    """Central registry and dispatch engine for DataSense AI tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.warning(f"Overwriting already registered tool: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool '{tool.name}' under category '{tool.category}'")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve tool by unique name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return structured metadata and JSON Schema for all registered tools."""
        manifest = []
        for name, tool in self._tools.items():
            schema = tool.input_schema.model_json_schema() if hasattr(tool.input_schema, "model_json_schema") else tool.input_schema.schema()
            manifest.append({
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value if hasattr(tool.category, "value") else str(tool.category),
                "permission": tool.permission.value if hasattr(tool.permission, "value") else str(tool.permission),
                "input_schema": schema
            })
        return manifest

    def execute_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ToolExecutionResult:
        """
        Validate inputs and invoke tool against session context.
        Guarantees structured ToolExecutionResult output even on unexpected exceptions.
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolExecutionResult(
                tool_name=name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Unknown tool '{name}'. Available tools: {list(self._tools.keys())}",
                mutated_data=False,
                requires_approval=False
            )

        # 1. Validate arguments against Pydantic schema
        try:
            validated_params = tool.input_schema(**arguments)
        except ValidationError as ve:
            errors = [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in ve.errors()]
            return ToolExecutionResult(
                tool_name=name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Input validation error: {'; '.join(errors)}",
                mutated_data=False,
                requires_approval=False
            )
        except Exception as e:
            return ToolExecutionResult(
                tool_name=name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Invalid arguments for tool '{name}': {str(e)}",
                mutated_data=False,
                requires_approval=False
            )

        # 2. Execute tool with bounded timeout enforcement
        timeout = getattr(tool, "timeout_seconds", 30.0)
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(tool.execute, validated_params, context)
                return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Tool '{name}' execution timed out after {timeout} seconds.")
            return ToolExecutionResult(
                tool_name=name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Tool '{name}' execution timed out after {timeout} seconds.",
                dataset_id=context.get("dataset_id"),
                dataset_version=context.get("dataset_version"),
                mutated_data=False,
                requires_approval=False
            )
        except ValueError as ve:
            return ToolExecutionResult(
                tool_name=name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=str(ve),
                dataset_id=context.get("dataset_id"),
                dataset_version=context.get("dataset_version"),
                mutated_data=False,
                requires_approval=False
            )
        except Exception as e:
            logger.exception(f"Unexpected error executing tool '{name}': {e}")
            return ToolExecutionResult(
                tool_name=name,
                status=ToolExecutionStatus.ERROR.value,
                error_details=f"Tool execution failed: {str(e)}",
                dataset_id=context.get("dataset_id"),
                dataset_version=context.get("dataset_version"),
                mutated_data=False,
                requires_approval=False
            )


# Global default registry singleton
default_registry = ToolRegistry()
