@echo off
REM =============================================================================
REM teardown.bat — DELETE the entire Azure resource group.
REM
REM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
REM WARNING: THIS IS DESTRUCTIVE AND PERMANENT.
REM
REM Running this command deletes:
REM   - The Azure resource group AND every resource inside it:
REM     Azure OpenAI, Azure AI Vision, Azure AI Speech, Azure AI Content Safety,
REM     Azure Container Registry, Azure Container Apps, Application Insights —
REM     ALL OF IT, GONE.
REM
REM There is NO undo. All data, logs, and deployments are permanently deleted.
REM
REM ONLY run this when you are fully done with the project and want to stop
REM ALL charges. Typically: after recording your demo video.
REM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
REM
REM FILL IN BEFORE RUNNING:
REM   Replace RESOURCE_GROUP below with your actual resource group name.
REM   The --yes flag skips the "are you sure?" prompt — remove it if you want
REM   to be asked for confirmation.
REM =============================================================================

set RESOURCE_GROUP=audible-frames-rg

echo.
echo ============================================================
echo  WARNING: About to DELETE resource group: %RESOURCE_GROUP%
echo  This will delete ALL Azure resources in this project.
echo  This is PERMANENT and cannot be undone.
echo ============================================================
echo.
pause

az group delete --name %RESOURCE_GROUP% --yes --no-wait

echo.
echo Deletion initiated. Resources are being removed in the background.
echo Check the Azure Portal to confirm deletion is complete.
echo After deletion, no further charges will accrue.
