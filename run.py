import pandas as pd
from src import scenario3

columns_formatting = ["k", "return", "volatility", "turnover", "drawdown", "sharpe"]

# Page 46 of 76

result_scenario3 = scenario3.main()
print(pd.DataFrame(result_scenario3)[columns_formatting])
