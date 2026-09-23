import pandas as pd
import numpy as np


def createSalary():
    # Create a sample DataFrame for London Housing data
    df = pd.DataFrame({

        'ExperienceYears':np.random.randint(1, 7, 100),
        'Salary': np.random.randint(200000, 500000, 100),

    })

    # Save the DataFrame to a CSV file
    df.to_csv('./data/salary.csv', index=False)


def companyStartup():
    df = pd.DataFrame({
    'AdminExp': np.random.randint(200000, 500000, 100),
    'RnDExp': np.random.randint(200000, 500000, 100),
    'MarketingExp':np.random.randint(200000, 500000, 100),
    'Location': np.random.choice(['Central London', 'North London', 'South London', 'East London'], 100),
    'Profit': np.random.randint(200000, 500000, 100),

    })
    df.to_csv('./data/startup.csv', index=False)
#createSalary()
companyStartup()