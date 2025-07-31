FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary application code and artifacts
COPY src/ src/
COPY models/ models/

# Set the default command for the container to run predict.py 
CMD ["python", "-m", "src.predict"]