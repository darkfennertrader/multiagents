FROM quay.io/jupyter/base-notebook


# Install system dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    build-essential \
    # package necessary for pyPDF
    # wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

# Install Python wheel package
RUN pip install wheel

# Return to the notebook user
USER ${NB_UID}

# Install in the default python3 environment
RUN pip install --no-cache-dir 'flake8' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Install from the requirements.txt file
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"