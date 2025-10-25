# Use an official Python 3.10 image
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy your application code
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit default port
EXPOSE 8501

# Run Streamlit via CLI so ScriptRunContext exists
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]