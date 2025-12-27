# Industrial Classifier (Streamlit)

A simple Streamlit app that loads a YOLO classifier and prints a formatted summary line like `0: 224x224 bent 0.75, good 0.21, color 0.03, scratch 0.01, flip 0.00, 187.7ms` when you upload or capture an image.

## Run locally

```bash
# From Windows PowerShell
cd D:\mvtech
python -m pip install -r requirements.txt
streamlit run app.py
```

## Notes
- `best (1).pt` is tracked via Git LFS to avoid large file issues.
- If you see path errors on Windows, using `pathlib.Path` in `app.py` handles spaces and backslashes safely.
