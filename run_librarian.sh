#!/bin/sh
# Run librarian.py with the project's virtual environment
cd "$(dirname "$0")" || exit 1
exec venv/bin/python3 librarian.py "$@"
