FROM public.ecr.aws/lambda/python:3.10 as base

COPY . ./

RUN pip install -r requirements.txt

CMD ["main.handler"]