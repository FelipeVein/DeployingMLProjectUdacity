FROM python:3.8
RUN pip install --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . /usr/src/app

# Set working directory
WORKDIR /usr/src/app/api

# Expose port
EXPOSE 8000

# Run server
ENTRYPOINT ["python", "app.py"]

