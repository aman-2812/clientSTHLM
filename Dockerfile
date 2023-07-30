# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install the dependencies using pip
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the FastAPI application files into the container
COPY . /app/

# Expose the port that FastAPI will be running on (change the port number if needed)
EXPOSE 8020

# Start the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8020"]
