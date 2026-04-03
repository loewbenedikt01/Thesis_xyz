Structure of the Program (includign output and uncoded yet)



Thesis_xyz
    analysis                        # Folder
        analyze_benchmark.py                # Run individually - calculates metrics, plots, etc.
        analyze_equal_weight.py             # Run individually - calculates metrics, plots, etc.
        analyze_hrp.py                      # Run individually - calculates metrics, plots, etc.
        analyze_lstm.py                     # Run individually - calculates metrics, plots, etc.
        analyze_market_cap.py               # Run individually - calculates metrics, plots, etc.
        analyze_markowitz_unconstrained.py  # Run individually - calculates metrics, plots, etc.
        analyze_markowitz.py                # Run individually - calculates metrics, plots, etc.
        analyze_random_forest.py            # Run individually - calculates metrics, plots, etc.
        analyze_xgboost.py                  # Run individually - calculates metrics, plots, etc.
        analyze_ml_models.py                # Run individually - calculates metrics, plots, etc.
        analyze_all_models.py               # Run individually - compares metrics, plots, etc.
    claude_updates                  # Folder
        benchmark                   # Showcase updates on what has happened in each model
        equal_weight                # Showcase updates on what has happened in each model
        hrp                         # Showcase updates on what has happened in each model
        lstm                        # Showcase updates on what has happened in each model
        market_cap                  # Showcase updates on what has happened in each model
        markowitz_unconstrained     # Showcase updates on what has happened in each model
        markowitz                   # Showcase updates on what has happened in each model
    	random_forest               # Showcase updates on what has happened in each model
        xgboost                     # Showcase updates on what has happened in each model
    models                          # Folder
        benchmark.py                # Run individually
        equal_weight.py             # Run individually
        hrp.py                      # Run individually
        lstm.py                     # Run individually
        market_cap.py               # Run individually
        markowitz_unconstrained.py  # Run individually
        markowitz.py                # Run individually
        random_forest.py            # Run individually
        xgboost.py                  # Run individually
    results                     # Folder    
        analysis                # Folder
            benchmark			        # Folder
                returns_buy_hold        # CSV, log return per month of the portfolio
            equal_weight			    # Folder
                returns_monthly         # CSV, log return per month of the portfolio
                returns_quarterly       # CSV, log return per month of the portfolio
                returns_semi-annual     # CSV, log return per month of the portfolio
                returns_annual          # CSV, log return per month of the portfolio
            hrp				            # Folder
                returns_monthly         # CSV, log return per month of the portfolio
                returns_quarterly       # CSV, log return per month of the portfolio
                returns_semi-annual     # CSV, log return per month of the portfolio
                returns_annual          # CSV, log return per month of the portfolio
            lstm				        # Folder
                run_01                  # Folder, for each run, create a new one (10) in total
                    returns_monthly         # CSV, log return per month of the portfolio
                    returns_quarterly       # CSV, log return per month of the portfolio
                    returns_semi-annual     # CSV, log return per month of the portfolio
                    returns_annual          # CSV, log return per month of the portfolio
            market_cap			        # Folder
                returns_monthly         # CSV, log return per month of the portfolio
                returns_quarterly       # CSV, log return per month of the portfolio
                returns_semi-annual     # CSV, log return per month of the portfolio
                returns_annual          # CSV, log return per month of the portfolio
            markowitz_unconstrained	    # Folder
                returns_monthly         # CSV, log return per month of the portfolio
                returns_quarterly       # CSV, log return per month of the portfolio
                returns_semi-annual     # CSV, log return per month of the portfolio
                returns_annual          # CSV, log return per month of the portfolio
            markowitz			        # Folder
                returns_monthly         # CSV, log return per month of the portfolio
                returns_quarterly       # CSV, log return per month of the portfolio
                returns_semi-annual     # CSV, log return per month of the portfolio
                returns_annual          # CSV, log return per month of the portfolio
            random_forest		        # Folder
                run_01                  # Folder, for each run, create a new one (10) in total
                    returns_monthly         # CSV, log return per month of the portfolio
                    returns_quarterly       # CSV, log return per month of the portfolio
                    returns_semi-annual     # CSV, log return per month of the portfolio
                    returns_annual          # CSV, log return per month of the portfolio
            xgboost			            # Folder
                run_01                  # Folder, for each run, create a new one (10) in total
                    returns_monthly         # CSV, log return per month of the portfolio
                    returns_quarterly       # CSV, log return per month of the portfolio
                    returns_semi-annual     # CSV, log return per month of the portfolio
                    returns_annual          # CSV, log return per month of the portfolio
        data                     # Folder
            benchmark			        # Folder
                portfolio               # CSV, columns (date[month end], ticker, price adj close, returns per month) [it is buy&hold]
            equal_weight			    # Folder
                portfolio_monthly       # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_quarterly     # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_semi-annual   # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_annual        # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
            hrp				            # Folder
                portfolio_monthly       # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_quarterly     # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_semi-annual   # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_annual        # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
            lstm				        # Folder
                statistics              # Folder
                    statistics_monthly              # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score, best nodes, best dropout, best val loss)
                    statistics_quarterly            # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score, best nodes, best dropout, best val loss)
                    statistics_semi-annual          # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score, best nodes, best dropout, best val loss)
                    statistics_annual               # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score, best nodes, best dropout, best val loss)
                run_01                  # Folder, for each run, create a new one (10) in total
                    portfolio_monthly       # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                    portfolio_quarterly     # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                    portfolio_semi-annual   # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                    portfolio_annual        # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
            market_cap			        # Folder
                portfolio_monthly       # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_quarterly     # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_semi-annual   # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_annual        # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
            markowitz_unconstrained	    # Folder
                portfolio_monthly       # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_quarterly     # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_semi-annual   # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_annual        # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
            markowitz			        # Folder
                portfolio_monthly       # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_quarterly     # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_semi-annual   # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                portfolio_annual        # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
            random_forest		        # Folder
                statistics              # Folder
                    statistics_monthly              # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score)
                    statistics_quarterly            # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score)
                    statistics_semi-annual          # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score)
                    statistics_annual               # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score)
                run_01                  # Folder, for each run, create a new one (10) in total
                    portfolio_monthly       # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                    portfolio_quarterly     # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                    portfolio_semi-annual   # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                    portfolio_annual        # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
            xgboost			            # Folder
                statistics              # Folder
                    statistics_monthly              # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score, ADF statistic, ADF p-value)
                    statistics_quarterly            # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score, ADF statistic, ADF p-value)
                    statistics_semi-annual          # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score, ADF statistic, ADF p-value)
                    statistics_annual               # CSV, all statistics per run, labeled with column name for each run to differentiate (RMSE, MSE, MAE, Rsquared, MAPE, Directional Accuracy, Geometric Score, ADF statistic, ADF p-value)
                run_01                  # Folder, for each run, create a new one (10) in total
                    portfolio_monthly       # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                    portfolio_quarterly     # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                    portfolio_semi-annual   # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
                    portfolio_annual        # CSV, columns (date[month end], ticker, initial weight of rebalancing, current weight[month end], pct change of stock within that month, weight*pct return this month, portfolio value, turnover)
        plots                    # Folder with plots created for visualization
            all_models                      # Folder      
            benchmark                       # Folder
            equal_weight                    # Folder
            hrp                             # Folder
            lstm                            # Folder
            market_cap                      # Folder
            markowitz_unconstrained         # Folder
            markowitz                       # Folder
            random_forest                   # Folder
            xgboost                         # Folder
    universe
    benchmark_price.parquet        # Benchmark with its closing prices over all years
    universe_prices.parquet         # All stocks with all closing prices over all years
    download_meta.txt
    .gitignore
    README.md                       # Documentation
    requirements.txt
