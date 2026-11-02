FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY focus_classifier ./focus_classifier
COPY research ./research
COPY deployment ./deployment
COPY scripts/check_install.py ./scripts/check_install.py
RUN python scripts/check_install.py
EXPOSE 8767
CMD ["python", "-m", "focus_classifier.serve", "--host", "0.0.0.0"]
