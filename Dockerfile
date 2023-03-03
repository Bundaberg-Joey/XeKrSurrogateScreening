FROM continuumio/miniconda3

# set envs to mitigate issues within OPENBLAS
# avoid pyc files cluttering everything up
# not have tensorflow complain about being run on CPU 
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=3

RUN apt-get -y update \
    && apt-get -y install vim

# General dependency installation
COPY requirements.txt .

RUN conda install python=3.10 \
    && conda install -c conda-forge raspa2 \
    && conda clean -afy \
    && python3 -m pip install -r requirements.txt --no-cache-dir

# pip installation of mkl and ami (need to copy dir across)
ADD code_libs /code_libs

RUN python3 -m pip install -e code_libs/ami/ \ 
    && python3 -m pip install -e code_libs/surrogate

WORKDIR /app

CMD ["bash"]