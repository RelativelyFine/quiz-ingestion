#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# USAGE: ./run.sh "../test_pdfs/1-6.pdf" 6 2
# ----------------------------------------------------------------------------

# Check if we have at least three arguments:
if [ $# -lt 3 ]; then
  echo "Usage: $0 <path_to_pdf> <total_pages> <per_test_page_count>"
  exit 1
fi

# Pass all arguments ("$@") to main.py
python src/main.py "$@"