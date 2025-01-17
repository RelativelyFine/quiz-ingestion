@echo off
REM ----------------------------------------------------------------------------
REM USAGE: run.bat "path\to\your.pdf" <total_pages> <per_test_page_count>
REM Example: run.bat "..\test_pdfs\1-6.pdf" 6 2
REM ----------------------------------------------------------------------------

IF "%~3"=="" (
    echo "Usage: run.bat <path_to_pdf> <total_pages> <per_test_page_count>"
    goto :EOF
)

REM
python src\main.py %*

REM
pause