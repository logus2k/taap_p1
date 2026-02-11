FROM logus2k/gan_game.server:latest

USER root

WORKDIR /src
COPY src/ ./

WORKDIR /src/app/server

EXPOSE 8993

CMD ["uvicorn", "run_game:socket_app", "--host", "0.0.0.0", "--port", "8993"]
