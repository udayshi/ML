import pandas as pd
import numpy as np



# Create a sample DataFrame for London Housing data
df = pd.DataFrame({

    'ExperienceYears':np.random.randint(1, 7, 100),
    'Salary': np.random.randint(200000, 500000, 100),

})

# Save the DataFrame to a CSV file
df.to_csv('./data/salary.csv', index=False)