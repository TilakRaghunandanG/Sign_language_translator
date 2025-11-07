# Sign Language Translator

A real-time sign language translation application using computer vision and machine learning.

## Quick Start

1. Clone the repository:
```powershell
git clone https://github.com/TilakRaghunandanG/Sign_language_translator.git
cd Sign_language_translator
```

2. Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

4. Run the application:
```powershell
streamlit run app/streamlit_app.py
```

## Features

- Real-time sign language translation using webcam
- Image upload support for static gesture recognition
- Text-to-speech output
- Translation support for multiple languages
- Clean and intuitive user interface

## Project Structure

- `app/`: Main application code
  - `streamlit_app.py`: Web interface
  - `translator.py`: Translation utilities
  - `utils.py`: Helper functions
- `data/`: Training data
- `training/`: Model training scripts
