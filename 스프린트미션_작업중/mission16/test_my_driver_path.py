"""
Simple quick test for my_driver_path function
"""
import os
import sys
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, "helper_utils"))

from helper_utils_colab import my_driver_path, my_driver

# Test 1: Basic join
print("TEST 1: Basic join")
base = my_driver()
result = my_driver_path("model", "v1")
print(f"  Base: {base}")
print(f"  Result: {result}")
print(f"  Expected: {str(Path(base) / 'model' / 'v1')}")
print()

# Test 2: Create directory
print("TEST 2: Create directory")
test_path = my_driver_path("test_dir_quicktest", create=True)
print(f"  Created: {test_path}")
print(f"  Exists: {os.path.isdir(test_path)}")
os.rmdir(test_path)
print(f"  Cleaned up")
print()

# Test 3: Absolute path (base should be ignored)
print("TEST 3: Absolute path")
abs_result = my_driver_path("/tmp/abs_test")
print(f"  Input: /tmp/abs_test")
print(f"  Result: {abs_result}")
print()

# Test 4: Escape detection
print("TEST 4: Escape detection (should raise ValueError)")
try:
    result = my_driver_path("..", allow_escape=False)
    print(f"  ERROR: Should have raised ValueError but got: {result}")
except ValueError as e:
    print(f"  OK: Correctly raised ValueError")
    print(f"  Message: {e}")
print()

# Test 5: Escape allowed
print("TEST 5: Escape allowed")
result = my_driver_path("..", allow_escape=True)
print(f"  Result: {result}")
print(f"  Is parent of base: {result == str(Path(base).parent)}")
print()

# Test 6: Validate missing
print("TEST 6: Validate missing (should raise FileNotFoundError)")
try:
    result = my_driver_path("nonexistent_xyz", validate=True)
    print(f"  ERROR: Should have raised FileNotFoundError but got: {result}")
except FileNotFoundError as e:
    print(f"  OK: Correctly raised FileNotFoundError")
    print(f"  Message: {e}")
