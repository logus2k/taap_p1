@echo off
setlocal

echo ===========================================
echo [%TIME%] "GAN vs. Human" Game - FULL REMOVE
echo ===========================================

:: -------------------------------------------------
:: Stop containers
:: -------------------------------------------------
call stop.bat || (
	echo [%TIME%] ERROR: Failed to stop containers.
	exit /b 1
)

:: -------------------------------------------------
:: Remove femulator containers (if any)
:: -------------------------------------------------
echo [%TIME%] Removing femulator containers...
for /f "tokens=*" %%i in ('docker ps -aq --filter "name=gan_game" 2^>nul') do (
	echo [%TIME%] Removing container %%i...
	docker rm -f %%i
)

:: -------------------------------------------------
:: Remove images (all tags)
:: -------------------------------------------------
echo [%TIME%] Removing images...

for /f "tokens=*" %%i in ('docker images -q logus2k/gan_game.server 2^>nul') do (
	docker rmi -f %%i
	echo [%TIME%] Image %%i removed.
)

for /f "tokens=*" %%i in ('docker images -q logus2k/gan_game 2^>nul') do (
	docker rmi -f %%i
	echo [%TIME%] Image %%i removed.
)

:: -------------------------------------------------
:: Optional: clean dangling layers
:: -------------------------------------------------
docker image prune -f >nul

echo =========================================
echo [%TIME%] Remove complete
echo =========================================
