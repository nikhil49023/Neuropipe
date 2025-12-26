@echo off
echo Starting Data Pipeline...
echo Input: "./input_docs"
echo Output: "./datasets/formatted"

python main.py batch "./input_docs" --name sample_dataset

echo.
echo Processing complete.
pause
