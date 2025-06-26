FROM condaforge/miniforge3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

SHELL ["/bin/bash", "-c"]

COPY . .

EXPOSE 8501

CMD ["./start.sh"]
