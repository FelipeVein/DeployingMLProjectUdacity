FROM python:3.8
RUN pip install --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . /usr/src/app

# Set working directory
WORKDIR /usr/src/app

# Expose port
EXPOSE 80

# Run server
ENTRYPOINT ["python", "main.py"]

