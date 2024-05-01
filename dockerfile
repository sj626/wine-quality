FROM apache/spark:3.2.0

# Install Python and required dependencies
RUN yum update -y && \
    yum install -y python3 && \
    yum clean all && \
    python3 -m ensurepip --upgrade && \
    rm -r /root/.cache

# Set working directory
WORKDIR /app

# Copy the code files
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Expose necessary ports
EXPOSE 8080

# Command to run the Python script
CMD ["python3", "app.py"]
