FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python3 download_models.py

EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0", "--timeout", "120", "app:app"]
