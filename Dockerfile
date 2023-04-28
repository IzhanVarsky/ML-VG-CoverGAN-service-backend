FROM python:3.7

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip cmake
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y cmake

# Install PyTorch
RUN pip3 install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0

COPY ./requirements.txt ./requirements_covergan.txt
COPY ./covergan/requirements.txt ./requirements_server.txt
WORKDIR .
# Install other Python libraries
RUN pip install -r ./requirements_covergan.txt
RUN pip install -r ./requirements_server.txt

RUN git clone --recursive https://github.com/IzhanVarsky/diffvg2022
RUN cd diffvg2022 && python setup.py install && cd .. && rm -rf diffvg2022

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libmagickwand-dev

# Install fonts
COPY ./covergan/fonts /usr/share/fonts
RUN fc-cache -f -v

EXPOSE 9013

ENTRYPOINT ["./entry.sh"]
