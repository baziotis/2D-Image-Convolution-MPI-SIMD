@echo off
cl /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include\x64" /I. /I"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Include" /I"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" /c %1.c

link /machine:x64 /out:%1.exe /dynamicbase "msmpi.lib" /libpath:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" /LIBPATH:"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\lib\amd64" /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.1A\Lib\x64" %1.obj