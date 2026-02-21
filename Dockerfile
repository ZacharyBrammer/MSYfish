FROM condaforge/miniforge3:24.11.3-0

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda clean -afy

SHELL ["/bin/bash", "-c"]

COPY . .

RUN mkdir -p /app/simulations

RUN chmod +x start.sh

EXPOSE 8501

CMD ["./start.sh"]
