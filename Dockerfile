FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
# Installed in its own layer before the code is copied, so docker build only
# reinstalls dependencies when requirements.txt changes, not on every code edit
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API actually needs at runtime, not the notebooks or the dashboard
COPY main/ ./main/
COPY src/ ./src/
COPY Finlora_Dataset/artifacts/Cleaned_Data.csv ./Finlora_Dataset/artifacts/Cleaned_Data.csv

EXPOSE 8000

# host 0.0.0.0 is required here, uvicorn's default of 127.0.0.1 is unreachable
# from outside the container even with the port published
CMD ["uvicorn", "main.app:app", "--host", "0.0.0.0", "--port", "8000"]