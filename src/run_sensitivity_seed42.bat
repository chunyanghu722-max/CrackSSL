@echo off
setlocal EnableExtensions

REM ============================================================
REM Crack500 10%% seed42 sensitivity analysis
REM lambda_skel in {0.02, 0.05, 0.10}
REM lambda_con  in {0.005, 0.01, 0.02}
REM Central setting (0.05,0.01) reuses existing official Full run:
REM results\contrast_seed42_10pct
REM ============================================================

cd /d C:\Users\73231\Desktop\ICASSP2027_CrackSSL
call conda activate crackssl
if errorlevel 1 exit /b 1

set LABELED=splits\Crack500\seed42\labeled_10pct.txt
set UNLABELED=splits\Crack500\seed42\unlabeled_10pct.txt

echo.
echo ============================================================
echo [0/9] Ensure existing central setting is evaluated
echo lambda_skel=0.05, lambda_con=0.01
echo ============================================================
if exist results\contrast_seed42_10pct\checkpoints\best.pt (
    python src\evaluate_ema_baseline.py --checkpoint results\contrast_seed42_10pct\checkpoints\best.pt --model teacher --max-images 0
    if errorlevel 1 exit /b 1
) else (
    echo WARNING: Existing central checkpoint not found.
    echo It will be marked as missing by the collector.
)

call :RUN 1 0.02 0.005 sens_skel002_con0005_seed42
if errorlevel 1 exit /b 1
call :RUN 2 0.02 0.01  sens_skel002_con001_seed42
if errorlevel 1 exit /b 1
call :RUN 3 0.02 0.02  sens_skel002_con002_seed42
if errorlevel 1 exit /b 1

call :RUN 4 0.05 0.005 sens_skel005_con0005_seed42
if errorlevel 1 exit /b 1
REM 0.05 / 0.01 is the existing official run
call :RUN 6 0.05 0.02  sens_skel005_con002_seed42
if errorlevel 1 exit /b 1

call :RUN 7 0.10 0.005 sens_skel010_con0005_seed42
if errorlevel 1 exit /b 1
call :RUN 8 0.10 0.01  sens_skel010_con001_seed42
if errorlevel 1 exit /b 1
call :RUN 9 0.10 0.02  sens_skel010_con002_seed42
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo All sensitivity runs finished. Collecting results...
echo ============================================================
python collect_sensitivity_results.py
if errorlevel 1 exit /b 1

echo.
echo DONE.
echo See:
echo   sensitivity_results_seed42.csv
echo   sensitivity_dice_matrix_seed42.csv
echo   sensitivity_summary_seed42.txt
echo.
pause
exit /b 0

:RUN
set IDX=%1
set SKEL=%2
set CON=%3
set OUT=%4

echo.
echo ============================================================
echo [%IDX%/9] lambda_skel=%SKEL% lambda_con=%CON%
echo Output: results\%OUT%
echo ============================================================

python src\train_ema_skeleton_contrastive.py ^
  --labeled-list %LABELED% ^
  --unlabeled-list %UNLABELED% ^
  --seed 42 ^
  --epochs 100 ^
  --topo-weight %SKEL% ^
  --contrast-weight %CON% ^
  --output results\%OUT%

if errorlevel 1 exit /b 1

python src\evaluate_ema_baseline.py ^
  --checkpoint results\%OUT%\checkpoints\best.pt ^
  --model teacher ^
  --max-images 0

if errorlevel 1 exit /b 1
exit /b 0
