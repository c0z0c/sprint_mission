FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

COPY train_model.py .

ENV RUNNING_IN_DOCKER=True

ENTRYPOINT ["/usr/local/bin/startup.sh"]
CMD ["python", "train_model.py"]