@REM Description: This script is used to install the VCClient for Zeroshot VC.
@REM reference:
@REM https://github.com/niel-blue/beatrice-trainer-webui/blob/main/setup.bat

chcp 65001 > NUL
@echo off

echo.
echo  =================================================
echo  VCClient for Zeroshot VC 
echo  =================================================
echo.

cd vcclient-for-zeroshot-server
call venv\Scripts\activate.bat

python vcclient_for_zeroshot_server/main.py start


