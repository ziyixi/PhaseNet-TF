# Use the official Python image as the base image
FROM python:3.9-slim
LABEL org.opencontainers.image.authors="docker@ziyixi.science"
LABEL org.opencontainers.image.source=https://github.com/ziyixi/cloudmailin-dida365-app
LABEL org.opencontainers.image.description="PhaseNet-TF: Advanced Seismic Arrival Time Detection via Deep Neural Networks in the Spectrogram Domain, Leveraging Cutting-Edge Image Segmentation Approaches"
LABEL org.opencontainers.image.licenses=MIT

# Set the working directory
WORKDIR /app

# Install curl, gcc, and other necessary system packages
RUN apt-get update && \
    apt-get install -y curl gcc python3-dev axel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the 'pyproject.toml' and 'poetry.lock' files into the container
COPY pyproject.toml poetry.lock ./

# Set the PATH environment variable to include Poetry's bin directory
ENV PATH="/root/.local/bin:$PATH"

# Install Poetry and the necessary dependencies
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false && \
    poetry config installer.parallel true && \
    poetry install --no-interaction --no-ansi

# Download the model checkpoint and store it in the 'models' directory
RUN mkdir -p models && \
    axel -n 10 -o models/model.ckpt https://github.com/ziyixi/PhaseNet-TF/releases/download/v0.1.0/model.ckpt

# Copy the 'src' directory into the container
COPY src src
COPY configs configs
COPY .project-root .project-root

# Set the environment variables for host and port
ENV HOST=127.0.0.1
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Start the application
CMD ["python", "-m", "src.app", "experiment=app_serve"]
