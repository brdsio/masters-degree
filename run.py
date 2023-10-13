import pandas as pd
from src import scenario2
from src import scenario3

columns_formatting = ["k", "return", "volatility", "turnover", "drawdown", "sharpe"]

# Page 43 of 76
result_scenario2 = scenario2.main()
print(result_scenario2)
print(pd.DataFrame(result_scenario2)[columns_formatting])

# Page 46 of 76
# result_scenario3 = scenario3.main()
# print(pd.DataFrame(result_scenario3)[columns_formatting])
