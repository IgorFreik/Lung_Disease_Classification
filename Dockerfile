FROM python:3.9

RUN pip install -r requirements.txt

RUN mkdir /app
WORKDIR /app
ADD . /app

EXPOSE 7860
ENV RE-TRAIN=True

ENTRYPOINT ["python"]
CMD ["__main__.py"]

