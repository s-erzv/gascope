FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x start.sh
RUN chmod -R 777 /app
EXPOSE 7860
CMD ["./start.sh"]