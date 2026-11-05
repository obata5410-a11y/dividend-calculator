FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY notion_history.py .
COPY static/ static/
COPY nisa_codes.txt .
COPY ticker_aliases.txt .
COPY yield_overrides.txt .
COPY dividend_evidence.json .
COPY distribution_history.json .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
