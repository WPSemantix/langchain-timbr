from typing import Any, Dict, List
from langchain_core.runnables import Runnable


class Chain(Runnable):
    """
    Compatibility base class that mimics the legacy langchain.chains.base.Chain
    interface (removed in langchain 1.x).

    Subclasses should implement:
      - ``_call(self, inputs, run_manager=None) -> dict``
      - ``input_keys`` property
      - ``output_keys`` property
    """

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        raise NotImplementedError

    def invoke(self, input: Dict[str, Any], config=None, **kwargs) -> Dict[str, Any]:
        return self._call(input)

    async def ainvoke(self, input: Dict[str, Any], config=None, **kwargs) -> Dict[str, Any]:
        return self._call(input)
