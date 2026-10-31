"""
DataSense AI — Service Layer Storage Interfaces & In-Memory Stores
Abstracts dataset sessions and multi-turn conversations behind clean interfaces
to provide clean seams for Redis or persistent database backends.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd


class DatasetStoreInterface(ABC):
    """Abstract interface for dataset persistence and versioning."""

    @abstractmethod
    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Retrieve the active dataframe for a given dataset ID."""
        pass

    @abstractmethod
    def set_dataset(self, dataset_id: str, df: pd.DataFrame, version: int = 0) -> None:
        """Store or update active dataframe and version."""
        pass

    @abstractmethod
    def get_version(self, dataset_id: str) -> int:
        """Get the current version index for a dataset."""
        pass

    @abstractmethod
    def list_datasets(self) -> List[str]:
        """List all active dataset identifiers."""
        pass


class ConversationStoreInterface(ABC):
    """Abstract interface for multi-turn conversation memory and signature tracking."""

    @abstractmethod
    def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve message history for a conversation."""
        pass

    @abstractmethod
    def append_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """Append a message to the conversation history."""
        pass

    @abstractmethod
    def get_chart_signature(self, conversation_id: str) -> Optional[str]:
        """Get the active chart signature associated with this conversation."""
        pass

    @abstractmethod
    def set_chart_signature(self, conversation_id: str, signature: str) -> None:
        """Update the active chart signature for this conversation."""
        pass

    @abstractmethod
    def clear_history(self, conversation_id: str) -> None:
        """Reset or clear conversation state."""
        pass


class InMemoryDatasetStore(DatasetStoreInterface):
    """In-memory dataset storage bridging to existing application runtime."""

    def __init__(self, storage_dict: Optional[Dict[str, pd.DataFrame]] = None, version_dict: Optional[Dict[str, List[Any]]] = None):
        self._storage = storage_dict if storage_dict is not None else {}
        self._versions = version_dict if version_dict is not None else {}

    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        return self._storage.get(dataset_id)

    def set_dataset(self, dataset_id: str, df: pd.DataFrame, version: int = 0) -> None:
        self._storage[dataset_id] = df

    def get_version(self, dataset_id: str) -> int:
        history = self._versions.get(dataset_id, [])
        if history:
            return history[-1].get("version_id", 0) if isinstance(history[-1], dict) else getattr(history[-1], "version_id", 0)
        return 0

    def list_datasets(self) -> List[str]:
        return list(self._storage.keys())


class InMemoryConversationStore(ConversationStoreInterface):
    """In-memory multi-turn conversation and chart signature store."""

    def __init__(self):
        self._conversations: Dict[str, List[Dict[str, Any]]] = {}
        self._signatures: Dict[str, str] = {}

    def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self._conversations.get(conversation_id, [])

    def append_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
        self._conversations[conversation_id].append(message)

    def get_chart_signature(self, conversation_id: str) -> Optional[str]:
        return self._signatures.get(conversation_id)

    def set_chart_signature(self, conversation_id: str, signature: str) -> None:
        self._signatures[conversation_id] = signature

    def clear_history(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)
        self._signatures.pop(conversation_id, None)


# Default in-memory conversation store singleton
default_conversation_store = InMemoryConversationStore()
