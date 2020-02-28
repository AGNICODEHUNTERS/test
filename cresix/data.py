import pandas as pd
customerData = pd.read_csv('customer_data.csv',)
salesDetails = pd.read_csv('sales_detal.csv')
salesSummary = pd.read_csv('sales_summary.csv')

customerData = customerData.dropna()
#customerData =pd.get_dummies(customerData,'gender')

salesDetails = salesDetails.dropna()
#salesDetails =pd.get_dummies(salesDetails,columns = ['year','month','catagory_code','end_date'])



res = pd.merge(salesSummary,salesDetails)
#res = pd.merge(res,salesSummary)
print(res)
res.to_csv('sale.csv')
