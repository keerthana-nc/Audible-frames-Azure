@echo off
REM =============================================================================
REM pause.bat — Scale the Azure Container App to zero replicas.
REM
REM WHAT THIS DOES:
REM   Sets the minimum AND maximum replica count to 0. This means:
REM   - The app is still deployed (URL still exists, config still there)
REM   - BUT no container is running, so compute charges stop immediately
REM   - Azure Container Apps charges only for running replicas, so 0 = $0/hour
REM
REM WHEN TO USE:
REM   When you're done working for the day but want to keep the deployment alive
REM   (e.g. you want the URL to still work when you scale back up tomorrow).
REM
REM HOW TO SCALE BACK UP:
REM   az containerapp update ^
REM     --name audible-frames-app ^
REM     --resource-group audible-frames-rg ^
REM     --min-replicas 0 ^
REM     --max-replicas 1
REM
REM FILL IN BEFORE RUNNING:
REM   Replace RESOURCE_GROUP and APP_NAME below with your actual values.
REM   These are set in Phase 6 when you deploy.
REM =============================================================================

set RESOURCE_GROUP=audible-frames-rg
set APP_NAME=audible-frames-app

echo Scaling %APP_NAME% to zero replicas...
az containerapp update ^
    --name %APP_NAME% ^
    --resource-group %RESOURCE_GROUP% ^
    --min-replicas 0 ^
    --max-replicas 0

echo Done. No compute charges will accrue until you scale back up.
