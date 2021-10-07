# Q5: 32a5625aad35
FROM agrigorev/zoomcamp-model:3.8.12-slim
FROM python:3.8.12-slim
RUN pip install pipenv
WORKDIR /app
COPY ["model1.bin", "dv.bin", "./"]
RUN pipenv install numpy scikit-learn==0.24.2 flask gunicorn
RUN pipenv install --system --deploy
COPY ["predict.py","predict_Script.py", "./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

