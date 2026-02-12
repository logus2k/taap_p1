FROM logus2k/gan_game.server:latest

USER root

WORKDIR /app/bin
COPY bin/ ./

WORKDIR /app/classifier
COPY classifier/ ./

WORKDIR /app/model/cgan
COPY model/cgan ./

WORKDIR /app/web
COPY web/ ./

WORKDIR /app/dataset
COPY dataset/ ./

WORKDIR /app/bin

EXPOSE 8993

CMD ["python3", "run_game.py", "--model-dir", "/app/model/cgan", "--classifier", "/app/classifier/model/mnist_cnn_best.ckpt", "--mnist-dir", "/app/dataset"]
