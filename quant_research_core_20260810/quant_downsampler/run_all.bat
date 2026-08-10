@echo off
setlocal
cd /d "%~dp0"

call run.bat run_step1.py
if errorlevel 1 exit /b %errorlevel%

call run.bat run_factor_eval.py --save
if errorlevel 1 exit /b %errorlevel%

call run.bat run_factor_robustness.py
if errorlevel 1 exit /b %errorlevel%

call run.bat run_task4_strict.py --factor-set required --out ..\output\backtest_required
if errorlevel 1 exit /b %errorlevel%

call run.bat run_strategy_analysis.py --task4-only --task4-dir backtest_required --recent-days 45
if errorlevel 1 exit /b %errorlevel%

call run.bat run_task4_strict.py --factor-set extended --out ..\output\backtest_strict
if errorlevel 1 exit /b %errorlevel%

call run.bat run_strategy_analysis.py --task4-only --task4-dir backtest_strict --recent-days 45
if errorlevel 1 exit /b %errorlevel%

call run.bat run_task4_robustness.py
if errorlevel 1 exit /b %errorlevel%

call run.bat run_factor_independence.py
if errorlevel 1 exit /b %errorlevel%

call run.bat ..\build_task4_report.py
if errorlevel 1 exit /b %errorlevel%

call run.bat run_factor_models.py --out ..\output\factor_models_aligned
if errorlevel 1 exit /b %errorlevel%

call run.bat run_factor_models.py --rolling --out ..\output\factor_models_rolling_aligned
if errorlevel 1 exit /b %errorlevel%

if /I "%~1"=="--with-lstm" (
    call run.bat run_lstm.py --stocks 5 --epochs 12 --seq-len 60 --class-weight-power 0.5 --out ..\output\lstm_next_minute
    if errorlevel 1 exit /b %errorlevel%
    call run.bat run_lstm_ensemble.py --stocks 5 --epochs 12 --seq-len 60 --device cuda --out-dir ..\output\lstm_ensemble --overwrite
    if errorlevel 1 exit /b %errorlevel%
    call run.bat validate_lstm_ensemble.py --run-dir ..\output\lstm_ensemble --data-dir ..\output\minute --device cuda
    if errorlevel 1 exit /b %errorlevel%
    call run.bat run_lstm_baselines.py
    if errorlevel 1 exit /b %errorlevel%
    call run.bat run_strategy_analysis.py --task5-only --sell-fee-bps 5
    if errorlevel 1 exit /b %errorlevel%
    call run.bat run_lstm_magnitude.py --epochs 12 --return-loss-weight 0.25 --sell-fee-bps 5 --device cpu --overwrite
    if errorlevel 1 exit /b %errorlevel%
    call run.bat run_tradable_return_research.py --sell-fee-bps 5
    if errorlevel 1 exit /b %errorlevel%
    call run.bat run_tradable_lstm.py --epochs 8 --sell-fee-bps 5 --device cpu
    if errorlevel 1 exit /b %errorlevel%
    call run.bat run_lstm_feature_independence.py --epochs 8 --sell-fee-bps 5 --device cpu
    if errorlevel 1 exit /b %errorlevel%
    call run.bat run_lstm_minimal_four.py --epochs 8 --sell-fee-bps 5 --device cpu
    if errorlevel 1 exit /b %errorlevel%
)

call run.bat validate_project_outputs.py
if errorlevel 1 exit /b %errorlevel%

echo.
echo Tasks 1-4 and the factor-learning baselines completed.
echo Use run_all.bat --with-lstm to include the Task 5 baseline and ensemble.
