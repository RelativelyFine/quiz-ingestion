# quiz-ingestion

The following instructions will change dramatically. This is a WIP.

Take note of the paths for the following:

- Install tessaract
- Install poppler
- Add the paths for tessaract and poppler to the first lines of `src/main.py`
- Create a python 3.12.8 env
- `pip install -r ./requirements.txt`
- execute run.sh or run.bat with specified arguments (<path_to_pdf> <total_pages> <per_test_page_count>)

## TODO

- Make tessaract and poppler bundled or installed in a contained environment (like docker)
- Streamline .env file
- Add gradescope API
- More robust error checking
- More robust error handling
- More robust error reporting
- Scribble detection
- Better reset point indicator detection
