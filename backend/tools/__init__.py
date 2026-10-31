"""
DataSense AI Tool Layer Package (Phase 3)
Initializes and registers all validated data-science tools in the central default_registry.
"""
from tools.base import (
    BaseTool,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolPermission,
    ToolCategory
)
from tools.registry import ToolRegistry, default_registry

from tools.dataset_tools import (
    ProfileDatasetTool,
    InspectDataQualityTool,
    GetDatasetSchemaTool,
    GetDatasetVersionHistoryTool
)
from tools.cleaning_tools import (
    PreviewCleaningChangesTool,
    ApplyApprovedCleaningTool,
    RejectCleaningProposalTool
)
from tools.analysis_tools import (
    RunExploratoryAnalysisTool,
    GenerateChartTool,
    AnalyzeChartTool,
    CompareGroupsTool,
    InvestigateAnomaliesTool,
    AnalyzeTimeSeriesTool,
    ComputeRateOfChangeTool,
    ComputePeriodComparisonTool
)
from tools.statistical_tools import (
    RunStatisticalTestTool
)
from tools.ml_tools import (
    PrepareModelingTaskTool,
    TrainAndCompareModelsTool,
    EvaluateFinalModelTool,
    AnalyzeModelErrorsTool,
    ExplainFeatureImportanceTool
)
from tools.reporting_tools import (
    GenerateAnalysisReportTool,
    ExportAnalysisArtifactTool
)


def register_all_tools(registry: ToolRegistry = default_registry) -> None:
    """Register all verified core data-science tools into the specified registry."""
    # Dataset tools
    registry.register(ProfileDatasetTool())
    registry.register(InspectDataQualityTool())
    registry.register(GetDatasetSchemaTool())
    registry.register(GetDatasetVersionHistoryTool())

    # Cleaning tools
    registry.register(PreviewCleaningChangesTool())
    registry.register(ApplyApprovedCleaningTool())
    registry.register(RejectCleaningProposalTool())

    # Analysis & Chart tools
    registry.register(RunExploratoryAnalysisTool())
    registry.register(GenerateChartTool())
    registry.register(AnalyzeChartTool())
    registry.register(CompareGroupsTool())
    registry.register(InvestigateAnomaliesTool())
    registry.register(AnalyzeTimeSeriesTool())
    registry.register(ComputeRateOfChangeTool())
    registry.register(ComputePeriodComparisonTool())

    # Statistical hypothesis testing tool
    registry.register(RunStatisticalTestTool())

    # Machine-learning tools
    registry.register(PrepareModelingTaskTool())
    registry.register(TrainAndCompareModelsTool())
    registry.register(EvaluateFinalModelTool())
    registry.register(AnalyzeModelErrorsTool())
    registry.register(ExplainFeatureImportanceTool())

    # Reporting tools
    registry.register(GenerateAnalysisReportTool())
    registry.register(ExportAnalysisArtifactTool())


# Automatically populate the default singleton registry upon module import
register_all_tools(default_registry)

__all__ = [
    "BaseTool",
    "ToolExecutionResult",
    "ToolExecutionStatus",
    "ToolPermission",
    "ToolCategory",
    "ToolRegistry",
    "default_registry",
    "register_all_tools"
]
