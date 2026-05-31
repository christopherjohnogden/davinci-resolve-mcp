"""Local test package marker.

Some optional CV dependencies install a top-level ``tests`` package into the
virtualenv. This marker ensures imports like ``tests._error_envelope_helpers``
resolve to this repository's tests.
"""
