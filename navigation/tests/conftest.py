"""Put navigation/ and ball-detection-algo/ on sys.path for the tests.

Both are flat script directories (hyphenated names can't be packages), so their
modules are imported top-level, the same way the scripts import each other.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "..", "ball-detection-algo"))
sys.path.insert(0, _ROOT)
