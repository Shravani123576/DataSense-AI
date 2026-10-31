"""
DataSense AI Tool Layer — Base Tool Definitions and Execution Contract (Phase 3)
Provides the core contract, status enums, and base class for all data-science tools.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field


class ToolExecutionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING_APPROVAL = "pending_approval"


class ToolPermission(str, Enum):
    READ_ONLY = "read_only"
    MUTATES_DATA = "mutates_data"
    REQUIRES_APPROVAL = "requires_approval"


class ToolCategory(str, Enum):
    DATASET = "dataset"
    CLEANING = "cleaning"
    ANALYSIS = "analysis"
    STATISTICAL = "statistical"
    ML = "ml"
    REPORTING = "reporting"


class ToolExecutionResult(BaseModel):
    """Uniform structured output returned by all DataSense AI tool executions."""
    tool_name: str
    status: str = Field(description="Execution status: 'success', 'error', or 'pending_approval'")
    result: Optional[Any] = Field(default=None, description="Structured tool-specific computation payload")
    dataset_id: Optional[str] = Field(default=None, description="Dataset session identifier")
    dataset_version: Optional[int] = Field(default=None, description="Dataset version index applied")
    warnings: List[str] = Field(default_factory=list, description="Data quality, assumption, or execution warnings")
    error_details: Optional[str] = Field(default=None, description="Safe user-facing error message on failure")
    mutated_data: bool = Field(default=False, description="Whether active dataset was mutated")
    requires_approval: bool = Field(default=False, description="Whether human approval is required to commit")
    provenance: Optional[Dict[str, Any]] = Field(default=None, description="Audit provenance and parameter metadata")


class BaseTool(ABC):
    """Abstract base class that all DataSense AI tools must implement."""
    name: str
    description: str
    category: ToolCategory
    permission: ToolPermission
    input_schema: Type[BaseModel]
    output_schema: Optional[Type[BaseModel]] = None
    timeout_seconds: float = 30.0

    @abstractmethod
    def execute(self, params: BaseModel, context: Dict[str, Any]) -> ToolExecutionResult:
        """Execute deterministic tool logic against verified session context."""
        pass
