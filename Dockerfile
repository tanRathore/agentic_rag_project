
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY .env ./.env

EXPOSE 8501
CMD ["streamlit", "run", "src/main_chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]