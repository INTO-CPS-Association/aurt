& venv\Scripts\Activate.ps1

# Ensure that imports will work when running the tests from inside the tests folder.
$Env:PYTHONPATH = "."
# Tell the tests that they are not to expect input from the user (like having to close a plot).
$Env:NONINTERACTIVE = "ON"

& python build.py $args

Pop-Location