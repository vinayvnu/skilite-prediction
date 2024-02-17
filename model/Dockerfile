FROM python:3.11-slim
LABEL authors="vinay"
WORKDIR /app
RUN mkdir -p /model
COPY app.py /app
COPY model.pkl /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
ENV MODEL_PATH /app/model.pkl
EXPOSE 8080
CMD ["top"]
CMD ["python", "app.py"]