Set WshShell = CreateObject("WScript.Shell")
strPath = WshShell.CurrentDirectory & "\launch_app.bat"
WshShell.Run chr(34) & strPath & chr(34), 0
Set WshShell = Nothing
