#!/usr/bin/env bash
# Decides which jobs of ci.yml a change needs, from the paths it changes.
#
# Reads the changed paths on stdin, one per line, and prints two lines in the
# form of $GITHUB_OUTPUT:
#
#   tests=true|false        false when every path is documentation, which no
#                           job of ci.yml reads
#   numpy_range=true|false  true when a path is one the NumPy range check is
#                           for: the package, the lock, or this workflow
#
# No paths at all (a list that could not be read) counts as a change to
# everything. The design is in docs/ci-queue-design.md.
set -euo pipefail

# Documentation: Markdown anywhere (READMEs, attempts logs, release notes,
# recorded results), the design documents and their images, the site and its
# builder, and the licence. No step of ci.yml reads any of them.
DOCS='(\.md$|^docs/|^docsite/|^LICENSE$)'
# What the numpy-range job checks against other NumPy versions. It runs on
# ARM64 macOS, where the package builds every kernel under kernels/ that has
# an arm64/ directory, or its naive.c where it has none: so the package and its
# binding, those kernels and their shared spec and reference, the delivery
# check's own cases, the lock that pins NumPy, the project's dependencies, and
# the workflow and this script themselves. Markdown under any of them is
# documentation and is filtered out first.
PACKAGE='^(hotkernels/|kernels/[^/]+/(arm64/|naive\.c$|spec\.py$)|delivery/(cases|check)\.py$|uv\.lock$|pyproject\.toml$|\.github/workflows/ci\.yml$|\.github/scripts/ci-scope\.sh$)'

paths=$(grep -v '^$' || true)
if [ -z "$paths" ]; then
  echo "tests=true"
  echo "numpy_range=true"
  exit 0
fi
if grep -qvE "$DOCS" <<< "$paths"; then echo "tests=true"; else echo "tests=false"; fi
# Filtered into a variable, not piped: with pipefail, `grep -q` leaving early
# could fail the pipe and read as no match.
code=$(grep -vE '\.md$' <<< "$paths" || true)
if grep -qE "$PACKAGE" <<< "$code"; then echo "numpy_range=true"; else echo "numpy_range=false"; fi
