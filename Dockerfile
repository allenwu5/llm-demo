FROM python:3.10.13-slim-bullseye

ENV TZ=Asia/Taipei

WORKDIR /workspaces

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt