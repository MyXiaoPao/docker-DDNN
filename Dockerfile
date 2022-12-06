FROM myruntime:base

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}
COPY alexnet_cifar10.py ${LAMBDA_TASK_ROOT}
COPY multiprocess.py ${LAMBDA_TASK_ROOT}
COPY trigger_local.py ${LAMBDA_TASK_ROOT}
COPY data /data/

RUN pip install opencv-python && \
    pip install opencv-python-headless


CMD [ "app.handler" ]
