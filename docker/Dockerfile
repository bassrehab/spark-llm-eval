# Custom Spark image with Python 3.11 for spark-llm-eval integration tests
# Based on official Apache Spark image, uses miniforge for Python 3.11
# This ensures consistent Python version across driver and workers

FROM apache/spark:3.5.0

LABEL maintainer="spark-llm-eval"
LABEL description="Apache Spark 3.5.0 with Python 3.11 for LLM evaluation"

# Switch to root for installations
USER root

# Install miniforge with Python 3.11 (uses conda-forge, no TOS issues)
ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Detect architecture for miniforge download
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then ARCH="aarch64"; elif [ "$ARCH" = "x86_64" ]; then ARCH="x86_64"; fi && \
    curl -fsSL -o /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda install -y python=3.11 && \
    conda clean -afy

# Set environment variables for PySpark
ENV PYSPARK_PYTHON=${CONDA_DIR}/bin/python3
ENV PYSPARK_DRIVER_PYTHON=${CONDA_DIR}/bin/python3
ENV PYTHONUNBUFFERED=1

# Install Python packages for LLM evaluation
RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    delta-spark==3.2.0 \
    pytest==9.0.2 \
    numpy \
    pandas \
    pyarrow \
    openai \
    anthropic \
    google-generativeai \
    jinja2 \
    scipy \
    mlflow

# Expose Spark ports
EXPOSE 4040 7077 8080 8081

# Default command
CMD ["bash"]
