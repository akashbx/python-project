import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load and clean the CSV
admission_data = pd.read_csv("bca_admissions.csv")

# Extract specialization names (excluding 'Year' and 'Impact_Reason')
specialization_names = [col for col in admission_data.columns if col not in ['Year', 'Impact_Reason']]

# Add "All" option
specialization_options = ["All (Total BCA Admissions)"] + specialization_names

# Display options
print("Available Specializations:")
for index, specialization in enumerate(specialization_options, 1):
    print(f"{index}. {specialization}")

# Get user input
selected_index = int(input("Select a specialization by number: "))
selected_specialization = specialization_options[selected_index - 1]

# Prepare input (X)
years = admission_data[['Year']]

# Prepare output (y)
if selected_specialization == "All (Total BCA Admissions)":
    admissions = pd.DataFrame(admission_data[specialization_names].sum(axis=1), columns=["TotalAdmissions"])
    y_label = "Total BCA Admissions"
    title_spec = "All Specializations"
else:
    admissions = admission_data[[selected_specialization]]
    y_label = f"{selected_specialization} Admissions"
    title_spec = selected_specialization

# Linear Regression
regression_model = LinearRegression()
regression_model.fit(years, admissions)

# Predict future
upcoming_years = [2025, 2026, 2027, 2028, 2029]
future_years_df = pd.DataFrame(upcoming_years, columns=["Year"])
predicted_admissions = regression_model.predict(future_years_df)

# Plotting
plt.figure(figsize=(13, 7))

# Historical data
plt.plot(admission_data["Year"], admissions, marker='o', label='Historical Data', color='blue')

# Annotate historical data: Admission above, Reason on opposite side
for i, year in enumerate(admission_data["Year"]):
    actual_value = int(admissions.values[i][0])
    reason = admission_data["Impact_Reason"].iloc[i]

    # Admission number above point
    plt.text(year, actual_value + 3, str(actual_value), fontsize=9, ha='center', color='blue')

    # Determine trend direction to place reason on opposite side
    if i > 0:
        prev_value = int(admissions.values[i - 1][0])
        trend_up = actual_value >= prev_value
    else:
        trend_up = True

    # Place reason below if trending up, above if down
    reason_y = actual_value - 8 if trend_up else actual_value + 8
    plt.text(year, reason_y, reason, fontsize=8, ha='center', rotation=30, color='darkgreen')

# Predicted data
plt.plot(upcoming_years, predicted_admissions, marker='x', linestyle='--', label='Predicted', color='red')

# Annotate predicted values
for i, year in enumerate(upcoming_years):
    predicted_value = int(predicted_admissions[i][0])
    plt.text(year, predicted_value + 2, f"{predicted_value}", fontsize=9, color='red', ha='center')

# Adjust axis limits tightly around data
all_years = list(admission_data["Year"]) + upcoming_years
all_values = list(admissions.values.flatten()) + list(predicted_admissions.flatten())
plt.xlim(min(all_years) - 1, max(all_years) + 1)
plt.ylim(min(all_values) - 10, max(all_values) + 30)

# Labels and Title
plt.title(f"BCA Admissions & Forecast - {title_spec}", fontsize=14)
plt.xlabel("Year")
plt.ylabel(y_label)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show it
plt.show()
