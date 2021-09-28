FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-training:2.4.1-transformers4.6.1-gpu-py37-cu110-ubuntu18.04

COPY requirements.txt .
COPY fine_tune.py .
COPY train.sh .
COPY data/split/* data/split/

RUN pip install -r requirements.txt

CMD [ "./train.sh" ]
