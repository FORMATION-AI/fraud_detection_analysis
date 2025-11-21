FROM python:3.11-slim

# Prevents Python from writing pyc files and enables unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for scientific stack and xgboost/catboost wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt pyproject.toml setup.py ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir gunicorn \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (includes artifacts and templates)
COPY . .

# Expose the Flask port
EXPOSE 5000

# Start the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
