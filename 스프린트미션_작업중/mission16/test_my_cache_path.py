#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for my_cache and my_cache_path functions."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

from helper_utils.helper_utils_colab import my_cache, my_cache_path

print("=" * 60)
print("Testing my_cache() and my_cache_path()")
print("=" * 60)

# Test 1: my_cache()
print("\n1. my_cache() - Get cache root:")
cache_root = my_cache()
print(f"   Cache root: {cache_root}")

# Test 2: my_cache_path() with subpaths
print("\n2. my_cache_path('models', 'v1') - Combine subpaths:")
cache_models = my_cache_path('models', 'v1', validate=False)
print(f"   Path: {cache_models}")

# Test 3: my_cache_path() with creation
print("\n3. my_cache_path('temp') - With default create=True:")
cache_temp = my_cache_path('temp')
print(f"   Path: {cache_temp}")
print(f"   Exists: {os.path.exists(cache_temp)}")
print(f"   Is directory: {os.path.isdir(cache_temp)}")

# Test 4: my_cache_path() with absolute path (Windows)
print("\n4. my_cache_path('C:\\\\tmp\\\\absolute') - Absolute path override:")
cache_abs = my_cache_path('C:\\tmp\\absolute', validate=False)
print(f"   Path: {cache_abs}")

# Test 5: my_cache_path() with allow_escape
print("\n5. my_cache_path('..', allow_escape=True) - Escape allowed:")
cache_escape = my_cache_path('..', allow_escape=True, validate=False)
print(f"   Path: {cache_escape}")

print("\n" + "=" * 60)
print("Tests completed successfully!")
print("=" * 60)
