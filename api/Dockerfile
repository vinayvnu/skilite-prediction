FROM python:3.11-slim
LABEL authors="vinay"
WORKDIR /app
COPY app.py /app
COPY model.pkl /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
ENV MODEL_PATH /app/models/model.pkl
EXPOSE 5000
#ENTRYPOINT ["top", "-b"]
CMD ["top"]
COPY loop.py /app
CMD ["python", "loop.py"]