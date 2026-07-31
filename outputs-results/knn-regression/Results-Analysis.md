# Results Interpretation

## KNN Regression Performance

The K-Nearest Neighbors (KNN) Regression model was evaluated using 5-fold cross-validation with feature standardization applied within each fold to prevent data leakage. The model achieved an average coefficient of determination (**R²**) of approximately **0.80**, indicating that the selected features explain around **80% of the variance** in vehicle fuel consumption (MPG).

The obtained error metrics suggest that the model is capable of producing accurate fuel consumption predictions. The Mean Absolute Error (**MAE**) indicates that predictions differ from the actual MPG values by approximately **2–3 MPG on average**, while the Root Mean Squared Error (**RMSE**) remains relatively low considering the range of MPG values present in the dataset.

## Feature Analysis

### Weight

Vehicle weight exhibits the strongest inverse relationship with fuel efficiency. Heavier vehicles consistently achieve lower MPG values, indicating that more energy is required to move larger masses. This feature appears to be one of the most influential predictors in the dataset.

### Horsepower

Horsepower also demonstrates a strong negative correlation with MPG. Vehicles equipped with more powerful engines tend to consume more fuel, resulting in lower fuel efficiency.

### Engine Displacement

Engine displacement shows a similar pattern to horsepower. Larger engines generally require more fuel to operate and are associated with lower MPG values.

### Number of Cylinders

Vehicles with a greater number of cylinders typically exhibit lower fuel efficiency. Four-cylinder engines are commonly associated with higher MPG values, while six- and eight-cylinder engines are more frequently observed among low-efficiency vehicles.

### Model Year

A positive relationship is observed between model year and fuel efficiency. Newer vehicles generally achieve higher MPG values, likely due to technological advancements in engine design, fuel management systems, and stricter fuel economy regulations introduced throughout the 1970s and early 1980s.

### Acceleration

The acceleration feature requires careful interpretation. In this dataset, acceleration represents the time required for a vehicle to accelerate from 0 to 60 mph, measured in seconds. Therefore, higher acceleration values indicate slower vehicles rather than faster ones.

A moderate positive relationship is observed between acceleration time and MPG. This does not imply that accelerating harder improves fuel efficiency. Instead, it suggests that vehicles requiring more time to reach 60 mph tend to be equipped with smaller, less powerful engines that consume less fuel. Consequently, slower-accelerating vehicles often achieve better fuel economy.

## Overall Findings

The analysis indicates that fuel efficiency is primarily influenced by vehicle size and engine characteristics. Weight, displacement, horsepower, and cylinder count exhibit strong negative relationships with MPG, while model year shows a positive relationship. Acceleration contributes additional information regarding vehicle performance but is less influential than the engine and weight-related features.

The KNN regression model successfully captures these relationships, achieving strong predictive performance and demonstrating that the selected vehicle characteristics are effective predictors of fuel consumption.