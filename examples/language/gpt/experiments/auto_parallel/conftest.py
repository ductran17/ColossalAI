# conftest.py — shared pytest configuration for auto_parallel tests.
#
# Python 3.13 work-around: torch.compile raises RuntimeError on first import
# of some ColossalAI modules. A retry loop lets the partial-import cache
# settle so the second or third attempt succeeds.
import sys

for _attempt in range(3):
    try:
        import colossalai.auto_parallel.pipeline_shard  # noqa: F401
        break
    except RuntimeError:
        pass
