FROM "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel" as development

# download and install poetry

RUN apt-get -y update
RUN apt-get -y install curl

RUN curl -sSL https://install.python-poetry.org | python3 -

COPY ./ /app/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

RUN /root/.local/bin/poetry config --local virtualenvs.in-project true

RUN /root/.local/bin/poetry install
