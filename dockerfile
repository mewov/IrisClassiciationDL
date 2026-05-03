FROM python:3.14
WORKDIR /app
COPY . .
RUN pip install -r req.txt
CMD [ "python", "app_main.py"]