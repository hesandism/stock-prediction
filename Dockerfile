# search for the meanings and the reasons for this commands.





FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.main"]








# FROM python:3.11-slim

# WORKDIR /app

# RUN apt-get update && apt-get install -y --no-install-recommends \
#  build-essential \
#  && rm -rf /var/lib/apt/lists/*

#  COPY requirements.txt .

#  RUN pip install --no-cache-dir -r requirements.txt

#  COPY . .

#  CMD ["python", "-m", "src.main"]


#-----------------------------------------------

# in the root folder, open terminal and run this:
    # docker compose up --build app