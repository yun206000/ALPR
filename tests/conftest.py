"""pytest configuration"""
import sys
from pathlib import Path

# 確保 src 在路徑中
sys.path.insert(0, str(Path(__file__).parent.parent))
