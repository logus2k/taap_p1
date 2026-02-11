@echo off
setlocal

echo =============================================
echo [%TIME%] "GAN vs. Human" - FULL CLEAN REBUILD
echo =============================================

:: -----------------------------------------------
:: Stop containers
:: -----------------------------------------------
call stop.bat || (
	echo [%TIME%] ERROR: Failed to stop containers.
	exit /b 1
)

:: -----------------------------------------------
:: Remove "GAN vs. Human" game containers (if any)
:: -----------------------------------------------
for /f "tokens=*" %%i in ('docker ps -aq --filter "name=gan_game"') do (
	echo [%TIME%] Removing gan_game containers...
	docker rm -f %%i
)

:: -----------------------------------------------
:: Image 1: gan_game.server:latest
:: -----------------------------------------------
set "image_name=logus2k/gan_game.server:latest"
set "dockerfile=gan_game.server.Dockerfile"

echo [%TIME%] Checking image: %image_name%
docker image inspect %image_name% >nul 2>&1
if %ERRORLEVEL% equ 0 (
	echo [%TIME%] Removing image '%image_name%'...
	docker rmi -f %image_name%
)

echo [%TIME%] Building image '%image_name%'...
docker build --no-cache -t %image_name% -f %dockerfile% .. || exit /b 1

:: -----------------------------------------------
:: Image 2: gan_game:latest
:: -----------------------------------------------
set "image_name=logus2k/gan_game:latest"
set "dockerfile=gan_game.Dockerfile"

echo [%TIME%] Checking image: %image_name%
docker image inspect %image_name% >nul 2>&1
if %ERRORLEVEL% equ 0 (
	echo [%TIME%] Removing image '%image_name%'...
	docker rmi -f %image_name%
)

echo [%TIME%] Building image '%image_name%'...
docker build --no-cache -t %image_name% -f %dockerfile% .. || exit /b 1

echo =========================================
echo [%TIME%] Rebuild complete. Run 'start.bat'
echo =========================================
