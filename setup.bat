@echo off
echo Setting up NeuroPipe...

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Checking for Granite 4 model...
ollama list | findstr "granite4"
if %errorlevel% neq 0 (
    echo Granite 4 model not found. Pulling now...
    ollama pull granite4
) else (
    echo Granite 4 model found.
)

echo.
echo Setup complete!
echo.
echo To run the web server:
echo   python main.py serve
echo.
echo To process a file:
echo   python main.py process path/to/file.pdf
echo.
pause
