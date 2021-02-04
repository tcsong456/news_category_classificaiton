FROM conda/miniconda3

COPY yaml/ci_dependencies.yml /setup/

ENV PATH /usr/local/envs/news_clf_dependencies/bin:$PATH

RUN apt-get update && apt-get install -y \
        build-essential \
        wget \
        git \
        unzip \
        && rm -rf /var/cache/apk/* \
        && git clone https://github.com/facebookresearch/fastText.git && \
        cd fastText && pip install . && cd ..
    
RUN conda update -n base -c defaults conda && \
    conda install python=3.7.5 && \
    conda env create -f /setup/ci_dependencies.yml && \
    /bin/bash -c "source activate news_clf_dependencies" && \
    /bin/bash -c "chmod -R 777 /usr/local/envs/news_clf_dependencies/lib/python3.7"