# **"GAN vs. Human" Game - Docker Image User Manual**

---

## **1. Prerequisites**
- **Docker Installed**: Ensure Docker is installed on your system.
  - [Download Docker Desktop](https://www.docker.com/products/docker-desktop) (Windows/macOS).
  - For Linux, follow the [official installation guide](https://docs.docker.com/engine/install/).

---

## **2. Download the Pre-Built Image**
Open a terminal or command prompt and run:
```bash
docker pull logus2k/gan_game
```
---

## **3. Run the Container**
Start the container with:
```bash
docker run -d -p 8993:8993 --name gan_game logus2k/gan_game
```
- `-d`: Runs the container in the background.
- `-p 8993:8993`: Maps port 8993 on your machine to the container.
- `--name femulator`: Names the container for easy reference.

---

## **4. Access the Application**
- Open a web browser and navigate to:
  [http://localhost:8993](http://localhost:8993)

---

## **5. Stop and Remove the Container**
When you’re done, stop the container:
```bash
docker stop gan_game
```
To remove the container:
```bash
docker rm gan_game
```

---

## **6. Verify the Image and Container**
- To list downloaded images:
  ```bash
  docker images
  ```
- To check running containers:
  ```bash
  docker ps
  ```

---

## **Troubleshooting**
- **Port Conflict**: If port 8993 is in use, change the host port (e.g., `-p 8080:8993`).
- **Image Not Found**: Ensure the image name and tag are correct.
- **Permission Issues**: Run Docker commands with administrative privileges if needed.

---

## **Notes**
- The image is pre-built and ready to use.
- No additional setup or configuration is required.
- For advanced usage, refer to the Docker documentation.

---
