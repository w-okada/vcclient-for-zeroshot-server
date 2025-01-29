chcp 65001 > NUL
@echo off

echo  =================================================
echo  VCClient for Zeroshot VC 
echo  =================================================

cd vcclient-for-zeroshot-server
call venv\Scripts\activate.bat

python vcclient_for_zeroshot_server/main.py start
