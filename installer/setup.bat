@REM Description: This script is used to install the VCClient for Zeroshot VC.
@REM reference:
@REM https://github.com/niel-blue/beatrice-trainer-webui/blob/main/setup.bat

chcp 65001 > NUL
@echo off

echo.
echo  =================================================
echo  VCClient for Zeroshot VC Installer
echo  =================================================
echo.

set TOOLS=%~dp0tools
set PS=PowerShell -ExecutionPolicy Bypass -Command

set PATH=%TOOLS%\PortableGit\bin;%TOOLS%\python;%TOOLS%\python\Scripts;%PATH%
set PYTHONPATH=%TOOLS%\python;
set PY="%TOOLS%\python\python.exe"
@REM set PIP_CACHE_DIR=%TOOLS%\pip

if exist "%TOOLS%\python" (goto :DLGit)
echo.
echo [Python download and Setup]
echo.

%PS% Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip -OutFile python.zip
%PS% Expand-Archive -Path python.zip -DestinationPath %TOOLS%\python
del python.zip

echo python312.zip> %TOOLS%\python\python312._pth
echo .>> %TOOLS%\python\python312._pth
echo # Uncomment to run site.main() automatically>> %TOOLS%\python\python312._pth
echo import site>> %TOOLS%\python\python312._pth


%PS% Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile %TOOLS%\python\get-pip.py
%PY% "%TOOLS%\python\get-pip.py" --no-warn-script-location
del %TOOLS%\python\get-pip.py
%PY% -m pip install virtualenv --no-warn-script-location


:DLGit
if exist "%TOOLS%\PortableGit" (goto :Gitclone)
echo.
echo [PortableGit Download]
echo.

%PS% Invoke-WebRequest -Uri https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.2/PortableGit-2.47.1.2-64-bit.7z.exe -OutFile %TOOLS%\PortableGit-2.47.1.2-64-bit.7z.exe
%TOOLS%\PortableGit-2.47.1.2-64-bit.7z.exe -y
del %TOOLS%\PortableGit-2.47.1.2-64-bit.7z.exe

:Gitclone
echo.
echo [Clone a repository]
echo.

set datetime=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set datetime=%datetime: =0%
set backupdir=vcclient-for-zeroshot-server_bkup_%datetime%
ren vcclient-for-zeroshot-server %backupdir%


git lfs install
git clone https://github.com/w-okada/vcclient-for-zeroshot-server.git vcclient-for-zeroshot-server

echo.
echo [Packages install]
echo.

cd vcclient-for-zeroshot-server

@REM seed-vcをclone
cd third_party
git clone https://github.com/Plachtaa/seed-vc.git 
cd seed-vc
git checkout 09d0b5cf131e364e7143c8069d0d03b6889072ba
cd ../..



%PY% -m virtualenv --copies venv
call venv\Scripts\activate.bat

@REM python -m pip install --upgrade pip
python -m pip install --upgrade pip==24.0

pip install -e .
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121




