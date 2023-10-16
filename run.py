import pandas as pd
from src import scenario1
from src import scenario2
from src import scenario3
from src import scenario4
from src import scenario5

columns_formatting = ["k", "return", "volatility", "turnover", "drawdown", "sharpe"]

# Page 29
result_scenario1 = scenario1.main()
print(result_scenario1)
print(pd.DataFrame(result_scenario1)[columns_formatting])

# Page 32
#result_scenario2 = scenario2.main()
#print(result_scenario2)
#print(pd.DataFrame(result_scenario2)[columns_formatting])

# Page 35
# result_scenario3 = scenario3.main()
# print(pd.DataFrame(result_scenario3)[columns_formatting])

# Page 37
# result_scenario4 = scenario4.main()
#print(result_scenario4)
# print(pd.DataFrame(result_scenario4)[columns_formatting])

# Page 40
# result_scenario5 = scenario5.main()
#print(result_scenario5)
# print(pd.DataFrame(result_scenario5)[columns_formatting])
