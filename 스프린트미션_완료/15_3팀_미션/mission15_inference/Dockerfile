# filepath: d:\GoogleDrive\homepage\sprint_mission\스프린트미션_작업중\mission15\mission15_inference\Dockerfile
FROM quay.io/jupyter/scipy-notebook:python-3.11

WORKDIR /home/jovyan/workspace

USER root
RUN apt-get update && \
    apt-get install -y nodejs npm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
USER jovyan

COPY --chown=jovyan:users requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# JupyterLab 4.2.5 호환 Mermaid Extension
RUN npm install -g @jupyterlab/mermaid-extension@4.2.5 && \
    jupyter labextension list

COPY --chown=jovyan:users startup.sh /home/jovyan/
RUN chmod +x /home/jovyan/startup.sh

COPY --chown=jovyan:users ./notebook/inference.ipynb .

ENV RUNNING_IN_DOCKER=True

ENTRYPOINT ["/home/jovyan/startup.sh"]

EXPOSE 8888