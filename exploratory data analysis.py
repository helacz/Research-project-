import matplotlib
matplotlib.use('TkAgg')  # Suggested by ChatGPT - Fixes PyCharm backend crash

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Open cleaned dataset
df = pd.read_csv('cleaned_application_train.csv')

# Color palette for graphs
gender_palette = {'F': '#6a0dad',  # dark purple
                  'M': '#add8e6'}  # light blue

# Gender distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='CODE_GENDER', hue='CODE_GENDER', palette=gender_palette, legend=False)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Income
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='CODE_GENDER', y='AMT_INCOME_TOTAL', palette=gender_palette)
plt.title("Income Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Annual Income (AMT_INCOME_TOTAL)")
plt.yscale('log')  # log scale to handle outliers, for better visibility
plt.tight_layout()
plt.show()

income_stats = df.groupby('CODE_GENDER')['AMT_INCOME_TOTAL'].agg(['mean', 'median', 'max'])
print("\nIncome statistics by gender:")
print(income_stats)

# Income - statistical significance with Mann-Whitney U test
from scipy.stats import mannwhitneyu
# Split income data by gender
income_f = df[df['CODE_GENDER'] == 'F']['AMT_INCOME_TOTAL']
income_m = df[df['CODE_GENDER'] == 'M']['AMT_INCOME_TOTAL']

stat, p = mannwhitneyu(income_f, income_m, alternative='two-sided')
print(f"Mann-Whitney U Test:")
print(f"U-statistic = {stat:.2f}")
print(f"p-value     = {p:.5f}")

# Total loan
total_loan_by_gender = df.groupby('CODE_GENDER')['AMT_CREDIT'].sum().reset_index()

# Plot
plt.figure(figsize=(6, 5))
sns.barplot(
    data=total_loan_by_gender,
    x='CODE_GENDER',
    y='AMT_CREDIT',
    palette=gender_palette
)
plt.title("Total Loan Amount by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Loan Amount")
plt.tight_layout()
plt.show()

# Credit to income ratio
df = df[df['AMT_INCOME_TOTAL'] > 0] # to avoid division by 0
df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
avg_ratio = df.groupby('CODE_GENDER')['CREDIT_INCOME_RATIO'].mean().reset_index()

# Bar plot
plt.figure(figsize=(6, 4))
sns.barplot(data=avg_ratio, x='CODE_GENDER', y='CREDIT_INCOME_RATIO', palette=gender_palette)
plt.title("Average Credit-to-Income Ratio by Gender")
plt.xlabel("Gender")
plt.ylabel("Avg Credit / Income Ratio")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
ax = sns.barplot(data=avg_ratio, x='CODE_GENDER', y='CREDIT_INCOME_RATIO', palette=gender_palette)

# Create income brackets - to control for income
bins = [0, 50000, 100000, 200000, 400000, df['AMT_INCOME_TOTAL'].max()]
labels = ['0–50k', '50–100k', '100–200k', '200–400k', '400k+']

df['INCOME_BRACKET'] = pd.cut(df['AMT_INCOME_TOTAL'], bins=bins, labels=labels, include_lowest=True)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='INCOME_BRACKET', y='CREDIT_INCOME_RATIO', hue='CODE_GENDER', palette=gender_palette,  ci=None)
plt.title("Credit-to-Income Ratio by Gender Across Income Brackets")
plt.xlabel("Income Bracket")
plt.ylabel("Credit / Income Ratio")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()

# Loan types
credit_type_counts = df.groupby(['CODE_GENDER', 'NAME_CONTRACT_TYPE']).size().reset_index(name='count')
print(credit_type_counts)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='NAME_CONTRACT_TYPE', hue='CODE_GENDER', palette=gender_palette)
plt.title("Types of Credit Requested by Gender")
plt.xlabel("Credit Type")
plt.ylabel("Count")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()

# Average loan amount by gender and loan type
avg_credit = df.groupby(['NAME_CONTRACT_TYPE', 'CODE_GENDER'])['AMT_CREDIT'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=avg_credit, x='NAME_CONTRACT_TYPE', y='AMT_CREDIT', hue='CODE_GENDER', palette=gender_palette)
plt.title("Average Loan Amount by Gender and Credit Type")
plt.ylabel("Average Credit Amount")
plt.tight_layout()
plt.show()

# Gender distribution for occupation type
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    y='OCCUPATION_TYPE',
    hue='CODE_GENDER',
    palette=gender_palette
)
plt.title("Gender Distribution by Occupation Type")
plt.xlabel("Count")
plt.ylabel("Occupation Type")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()

# Proportion of male and female for occupation
occupation_gender_counts = df.groupby(['OCCUPATION_TYPE', 'CODE_GENDER']).size().reset_index(name='Count')
# Normalize
total_per_occupation = occupation_gender_counts.groupby('OCCUPATION_TYPE')['Count'].transform('sum')
occupation_gender_counts['Percentage'] = (occupation_gender_counts['Count'] / total_per_occupation) * 100

plt.figure(figsize=(10, 6))
sns.barplot(
    data=occupation_gender_counts,
    x='Percentage',
    y='OCCUPATION_TYPE',
    hue='CODE_GENDER',
    palette=gender_palette
)
plt.title("Gender Proportion by Occupation Type (%)")
plt.xlabel("Percentage")
plt.ylabel("Occupation Type")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()

# Check for pay gap
avg_income_occupation = df.groupby(['OCCUPATION_TYPE', 'CODE_GENDER'])['AMT_INCOME_TOTAL'].mean().reset_index()
plt.figure(figsize=(12, 7))
sns.barplot(
    data=avg_income_occupation,
    x='AMT_INCOME_TOTAL',
    y='OCCUPATION_TYPE',
    hue='CODE_GENDER',
    palette=gender_palette
)
plt.title("Average Income by Gender and Occupation Type")
plt.xlabel("Average Income")
plt.ylabel("Occupation Type")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()

# Check if significant - suggested by ChatGPT
occupation_pvals = []
for occupation in df['OCCUPATION_TYPE'].dropna().unique():
    group = df[df['OCCUPATION_TYPE'] == occupation]
    female_income = group[group['CODE_GENDER'] == 'F']['AMT_INCOME_TOTAL']
    male_income = group[group['CODE_GENDER'] == 'M']['AMT_INCOME_TOTAL']
    # Only test if both groups exist
    if len(female_income) > 0 and len(male_income) > 0:
        stat, p = mannwhitneyu(female_income, male_income, alternative='two-sided')
        occupation_pvals.append({
            'Occupation': occupation,
            'Female mean income': female_income.mean(),
            'Male mean income': male_income.mean(),
            'p-value': p
        })
occupation_results = pd.DataFrame(occupation_pvals)
occupation_results = occupation_results.sort_values(by='p-value')
print(occupation_results)

# Gender distribution among education
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='NAME_EDUCATION_TYPE', hue='CODE_GENDER', palette=gender_palette)
plt.title("Gender Distribution by Education Level")
plt.xlabel("Count")
plt.ylabel("Education Level")
plt.tight_layout()
plt.show()

# Loan amount by education
avg_credit_edu = df.groupby(['NAME_EDUCATION_TYPE', 'CODE_GENDER'])['AMT_CREDIT'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=avg_credit_edu, y='NAME_EDUCATION_TYPE', x='AMT_CREDIT', hue='CODE_GENDER', palette=gender_palette)
plt.title("Average Credit Amount by Gender and Education Level")
plt.xlabel("Average Credit Amount")
plt.ylabel("Education Level")
plt.tight_layout()
plt.show()


# Income & gender distribution among income types
avg_income_by_type = df.groupby(['NAME_INCOME_TYPE', 'CODE_GENDER'])['AMT_INCOME_TOTAL'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=avg_income_by_type, y='NAME_INCOME_TYPE', x='AMT_INCOME_TOTAL', hue='CODE_GENDER', palette=gender_palette)
plt.title("Average Income by Gender and Income Type")
plt.xlabel("Average Income")
plt.ylabel("Income Type")
plt.tight_layout()
plt.show()

# Who do women and men apply for credit with?
companion_counts = df.groupby(['NAME_TYPE_SUITE', 'CODE_GENDER']).size().reset_index(name='Count')
companion_totals = companion_counts.groupby('NAME_TYPE_SUITE')['Count'].transform('sum')
companion_counts['Percentage'] = (companion_counts['Count'] / companion_totals) * 100

plt.figure(figsize=(10, 6))
sns.barplot(
    data=companion_counts,
    y='NAME_TYPE_SUITE',
    x='Percentage',
    hue='CODE_GENDER',
    palette=gender_palette
)
plt.title("Application Context (NAME_TYPE_SUITE) by Gender (Percentage)")
plt.xlabel("Percentage (%)")
plt.ylabel("Type of Companion")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()

# Family status
family_counts = df.groupby(['NAME_FAMILY_STATUS', 'CODE_GENDER']).size().reset_index(name='Count')
family_totals = family_counts.groupby('NAME_FAMILY_STATUS')['Count'].transform('sum')
family_counts['Percentage'] = (family_counts['Count'] / family_totals) * 100

plt.figure(figsize=(10, 6))
sns.barplot(
    data=family_counts,
    x='Percentage',
    y='NAME_FAMILY_STATUS',
    hue='CODE_GENDER',
    palette=gender_palette
)
plt.title("Gender Distribution by Family Status (Percentage)")
plt.xlabel("Percentage (%)")
plt.ylabel("Family Status")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()


# Credit amount by family status and gender
avg_credit_family = df.groupby(['NAME_FAMILY_STATUS', 'CODE_GENDER'])['AMT_CREDIT'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_credit_family, y='NAME_FAMILY_STATUS', x='AMT_CREDIT', hue='CODE_GENDER', palette=gender_palette)
plt.title("Average Credit Amount by Gender and Family Status")
plt.xlabel("Average Credit Amount")
plt.ylabel("Family Status")
plt.tight_layout()
plt.show()

# Children
# Grouped values so that big amount of children is in group for better interpretability
df['CHILDREN_GROUPED'] = df['CNT_CHILDREN'].apply(lambda x: x if x < 5 else '5+')
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    x='CHILDREN_GROUPED',
    hue='CODE_GENDER',
    palette=gender_palette,
    order=[0, 1, 2, 3, 4, '5+']  # Ensure correct order
)
plt.title("Number of Children by Gender (Grouped)")
plt.xlabel("Number of Children")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Credit amount by children and gender
avg_credit_grouped = df.groupby(['CHILDREN_GROUPED', 'CODE_GENDER'])['AMT_CREDIT'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=avg_credit_grouped,
    x='CHILDREN_GROUPED',
    y='AMT_CREDIT',
    hue='CODE_GENDER',
    palette=gender_palette,
    order=[0, 1, 2, 3, 4, '5+']
)
plt.title("Average Credit Amount by Gender and Number of Children (Grouped)")
plt.xlabel("Number of Children")
plt.ylabel("Average Credit Amount")
plt.tight_layout()
plt.show()

# Column for single/separated/widowed women with children
solo_statuses = ['Single / not married', 'Separated', 'Widow']
df['SOLO_WOMAN_WITH_CHILDREN'] = (
    (df['CODE_GENDER'] == 'F') &
    (df['NAME_FAMILY_STATUS'].isin(solo_statuses)) &
    (df['CNT_CHILDREN'] > 0)
)
# Count
solo_count = df['SOLO_WOMAN_WITH_CHILDREN'].sum()
print(f"Total solo women with children: {solo_count}")

# Average credit compared to others
plt.figure(figsize=(6, 5))
sns.barplot(
    data=df,
    x='SOLO_WOMAN_WITH_CHILDREN',
    y='AMT_CREDIT',
    palette=[gender_palette['M'], gender_palette['F']],
    ci=None
)
plt.title("Average Credit: Solo Women with Children vs Others")
plt.xticks([0, 1], ['Others', 'Solo Women with Children'])
plt.ylabel("Average Credit Amount")
plt.tight_layout()
plt.show()

# Solo dads
df['SOLO_MAN_WITH_CHILDREN'] = (
    (df['CODE_GENDER'] == 'M') &
    (df['NAME_FAMILY_STATUS'].isin(['Single / not married', 'Separated', 'Widow'])) &
    (df['CNT_CHILDREN'] > 0)
)
solo_parents = df[df['SOLO_WOMAN_WITH_CHILDREN'] | df['SOLO_MAN_WITH_CHILDREN']].copy()
# Create a combined column to use for plotting
solo_parents['SOLO_PARENT_GENDER'] = solo_parents.apply(
    lambda row: 'Solo Mother' if row['SOLO_WOMAN_WITH_CHILDREN'] else 'Solo Father',
    axis=1
)
plt.figure(figsize=(6, 5))
sns.barplot(
    data=solo_parents,
    x='SOLO_PARENT_GENDER',
    y='AMT_CREDIT',
    palette={'Solo Mother': gender_palette['F'], 'Solo Father': gender_palette['M']},
    ci=None
)
plt.title("Average Credit: Solo Mothers vs Solo Fathers")
plt.xlabel("")
plt.ylabel("Average Credit Amount")
plt.tight_layout()
plt.show()
# Income of solo parents
plt.figure(figsize=(6, 5))
sns.barplot(
    data=solo_parents,
    x='SOLO_PARENT_GENDER',
    y='AMT_INCOME_TOTAL',
    hue='SOLO_PARENT_GENDER',
    palette={'Solo Mother': gender_palette['F'], 'Solo Father': gender_palette['M']},
    errorbar=None,
    legend=False
)
plt.title("Average Income: Solo Mothers vs Solo Fathers")
plt.xlabel("")
plt.ylabel("Average Income")
plt.tight_layout()
plt.show()

# C/I for solo parents
solo_parents['CREDIT_TO_INCOME'] = solo_parents['AMT_CREDIT'] / solo_parents['AMT_INCOME_TOTAL']
plt.figure(figsize=(6, 5))
sns.barplot(
    data=solo_parents,
    x='SOLO_PARENT_GENDER',
    y='CREDIT_TO_INCOME',
    palette={'Solo Mother': gender_palette['F'], 'Solo Father': gender_palette['M']},
    ci=None  # No error bars
)
plt.title("Average Credit-to-Income Ratio: Solo Mothers vs Solo Fathers")
plt.xlabel("")
plt.ylabel("Credit / Income Ratio")
plt.tight_layout()
plt.show()

# Occupation type and solo parents
# Count
occupation_counts = solo_parents.groupby(['OCCUPATION_TYPE', 'SOLO_PARENT_GENDER']).size().reset_index(name='Count')
# Normalize within each occupation
totals = occupation_counts.groupby('OCCUPATION_TYPE')['Count'].transform('sum')
occupation_counts['Percentage'] = (occupation_counts['Count'] / totals) * 100

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=occupation_counts,
    x='Percentage',
    y='OCCUPATION_TYPE',
    hue='SOLO_PARENT_GENDER',
    palette={'Solo Mother': gender_palette['F'], 'Solo Father': gender_palette['M']}
)
plt.title("Proportion of Solo Mothers vs Solo Fathers by Occupation Type")
plt.xlabel("Percentage (%)")
plt.ylabel("Occupation Type")
plt.legend(title="Solo Parent Gender")
plt.tight_layout()
plt.show()

# Income
avg_income_by_occ = solo_parents_occ.groupby(['OCCUPATION_TYPE', 'SOLO_PARENT_GENDER'])['AMT_INCOME_TOTAL'].mean().reset_index()

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(
    data=avg_income_by_occ,
    x='AMT_INCOME_TOTAL',
    y='OCCUPATION_TYPE',
    hue='SOLO_PARENT_GENDER',
    palette={'Solo Mother': '#6a0dad', 'Solo Father': '#add8e6'}
)
plt.title("Average Income by Occupation Type: Solo Mothers vs Solo Fathers")
plt.xlabel("Average Income")
plt.ylabel("Occupation Type")
plt.legend(title="Solo Parent Gender")
plt.tight_layout()
plt.show()