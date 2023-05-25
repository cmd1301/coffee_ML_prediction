FROM python:3.8-slim-buster

WORKDIR /

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "script:app", "--host", "0.0.0.0", "--port", "80"]

