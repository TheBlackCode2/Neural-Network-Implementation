@echo off

premake5 gmake
make config=debug_win64

if %errorlevel% equ 0 (
    call bin\Debug-Win64\NeuralNetwork.exe
) else (
    echo [ERROR]: Failed to compile program!
)

pause
exit