import pandas as pd
import numpy as np
import os




def createSalary():
    # Create a sample DataFrame for London Housing data
    target_file:str='./data/salary.csv'
    if not os.path.exists(target_file):
        df = pd.DataFrame({

            'ExperienceYears':np.random.randint(1, 7, 100),
            'Salary': np.random.randint(200000, 500000, 100),

        })


        # Save the DataFrame to a CSV file
        df.to_csv(target_file, index=False)


def companyStartup():
    target_file:str='./data/startup.csv'
    if not os.path.exists(target_file):
        df = pd.DataFrame({
        'AdminExp': np.random.randint(200000, 500000, 100),
        'RnDExp': np.random.randint(200000, 500000, 100),
        'MarketingExp':np.random.randint(200000, 500000, 100),
        'Location': np.random.choice(['Central London', 'North London', 'South London', 'East London'], 100),
        'Profit': np.random.randint(200000, 500000, 100),

        })
        df.to_csv('./data/startup.csv', index=False)


def annModeling():
    target_file:str='./data/ann.csv'
    if not os.path.exists(target_file):
        df = pd.DataFrame({
            'CreditScore': np.random.randint(100, 900, 100),
            'Location': np.random.choice(['UK', 'France', 'Spain', 'Germany'], 100),
            'Gender': np.random.choice(['M', 'F'], 100),
            'Age': np.random.randint(18, 70, 100),
            'RightsToWork': np.random.choice([0, 1], 100),
            'Balance': np.random.randint(10000, 90000, 100),
            'TotalProducts': np.random.randint(1, 5, 100),
            'HasCrCard': np.random.choice([0, 1], 100),
            'isActiveMember': np.random.choice([0, 1], 100),
            'isActive': np.random.choice([0, 1], 100),
        })
        df.to_csv('./data/ann.csv', index=False)


createSalary()
companyStartup()
annModeling()