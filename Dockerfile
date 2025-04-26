FROM apache/airflow:2.8.1

# Switch to root user for system-level operations
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    pkg-config \
    libffi-dev \
    libcairo2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set proper permissions for directories
RUN mkdir -p /opt/airflow/logs /opt/airflow/dags /opt/airflow/plugins /opt/airflow/data
RUN chown -R airflow:root /opt/airflow

# Switch back to airflow user for Python package installations
USER airflow

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    chromadb \
    gitpython \
    pytest \
    pandas \
    numpy \
    transformers \
    pillow \
    scikit-learn

# Create symbolic link to make imports work correctly
RUN ln -sf /opt/airflow/Github_Multimodal_RAG /opt/airflow/multimodal_rag_system

# Set the PYTHONPATH for module imports
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"

# Work directory
WORKDIR /opt/airflow

# The airflow user is used by default when the container starts
ENTRYPOINT ["bash", "-c"]