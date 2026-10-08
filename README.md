# HeatShield X

A 36-hour hackathon project prototype for urban heat risk analysis.

## Setup
Requires Python 3.13.13.

1. Create virtual environment: `python -m venv .venv`
2. Activate: `source .venv/bin/activate` or `.venv\Scripts\activate` on Windows
3. Install dependencies: `pip install -r requirements.txt`
4. Create `.env` from `.env.example`
5. Run tests: `pytest`
6. Load data (CLI): `python scripts/load_data.py`
7. Run app: `streamlit run app/Home.py`
