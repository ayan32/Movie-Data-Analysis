import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, mannwhitneyu, wilcoxon, ks_2samp, bootstrap
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.decomposition import PCA

#%% D1
#Q1
df = pd.read_csv('movieDataReplicationSet.csv')
data = df.columns[:400] #Restrict to first 400 columns as the last 5 are irrelevant to the calculations


#Q2
means = df[data].mean(axis=0, skipna=True)

#Q3
medians = df[data].median(axis=0, skipna=True)

#Q4
modalMovieRatings = df[data].mode(axis=0, dropna=True)

#Q5
meanOfMeans = means.mean()

print("Means:\n", means)
print("\nMedians:\n", medians)
print("\nModal Movie Ratings:\n", modalMovieRatings)
print("\nMean of Means:", meanOfMeans)

#%% D2
#Q2
stds = df[data].std(axis=0, skipna=True)

#Q3: No module for Mean Absolute Deviation, so will manually calculate. Equation: (sum(|num-mean|))/(number of data)
mean_abs_devs = df[data].apply(lambda x: (x - means[x.name]).abs().mean(), axis=0)
#mad = (data - data.mean(axis=0, skipna=True)).abs().mean(axis=0, skipna=True)

#Q4
stds_mean = stds.mean()
stds_median = stds.median()

#Q5
mean_abs_devs_mean = mean_abs_devs.mean()
mean_abs_devs_median = mean_abs_devs.median()

#Q6
pearson_correlation = df[data].corr(method='pearson')

#Q7
correlation_vals = pearson_correlation.values.flatten() #Collapse the 2D array to a 1D for mean/median
mean_correlation = np.mean(correlation_vals)
median_correlation = np.median(correlation_vals)

print("\nStandard Deviations:\n", stds)
print("\nMean Absolute Deviations:\n", mean_abs_devs)
#print(mad)
print("\nMean of Standard Deviations:", stds_mean)
print("\nMedian of Standard Deviations:", stds_median)
print("\nMean of Mean Absolute Deviations:", mean_abs_devs_mean)
print("\nMedian of Mean Absolute Deviations:", mean_abs_devs_median)
print("\nPearson's Correlation:\n", pearson_correlation)
print("\nMean of Correlation:", mean_correlation)
print("\nMedian of Correlation:", median_correlation)

#%% D3
#Q2
star_wars_1 = df.iloc[:, 273] #Column 274
star_wars_2 = df.iloc[:, 93] #Column 94
both_star_wars_ratings = df.iloc[:, [273, 93]].dropna() #Filter to those that rated both movies

#Q3
X1 = both_star_wars_ratings.iloc[:, 1].values.reshape(-1, 1) #Star Wars II ratings (X)
y1 = both_star_wars_ratings.iloc[:, 0].values #Star Wars I ratings (y)

#Build linear regression model
regressor_star_wars = LinearRegression().fit(X1, y1)

#Find beta (slope), intercept, and residuals
beta_star_wars = regressor_star_wars.coef_
intercept_star_wars = regressor_star_wars.intercept_
yhat_star_wars = beta_star_wars * X1 + intercept_star_wars
residuals_star_wars = y1 - yhat_star_wars.flatten()

#Q4
#Return results
print("Beta (Slope Coefficient):", beta_star_wars)
print("Intercept (Star Wars I vs. Star Wars II):", intercept_star_wars)
print("Residuals:", residuals_star_wars)

#Q5
titanic = df.iloc[:, 292] #Add Titanic column 292
titanic_star_wars = df.iloc[:, [273, 292]].dropna() #Filter to those that rated both Star Wars I and Titanic

#Q6
X2 = titanic_star_wars.iloc[:, 0].values.reshape(-1, 1) #Star Wars I ratings (X)
y2 = titanic_star_wars.iloc[:, 1].values #Titanic ratings (y)

#Build linear regression model
regressor_titanic_star_wars = LinearRegression().fit(X2, y2)

#Find beta (slope), intercept, and residuals
beta_t_sw = regressor_titanic_star_wars.coef_
intercept_t_sw = regressor_titanic_star_wars.intercept_
yhat_t_sw = beta_t_sw * X2 + intercept_t_sw
residuals_t_sw = y2 - yhat_t_sw.flatten()

#Q7
#Return results
print("Beta (Slope Coefficient):", beta_t_sw)
print("Intercept (Star Wars I vs. Titanic):", intercept_t_sw)
print("Residuals:", residuals_t_sw)

#A question on the quiz
all_three_ratings = df.iloc[:, [273, 93, 292]].dropna()

#%% D4
#Get the necessary columns
education = df['Education']
income = df['income']
ses = df['SES']

#Q2
#Compute the correlation coefficient
#Then print out one of the non-diagonal indices for correlation coefficient
corr = np.corrcoef(education, income)
print(corr[0][1])

#Q3
#Calculate the residuals controlling for SES
education_residual = education - np.polyval(np.polyfit(ses, education, 1), ses)
income_residual = income - np.polyval(np.polyfit(ses, income, 1), ses)

#Calculate the partial correlation between education and income
partial_corr = pearsonr(education_residual, income_residual)
print(partial_corr)

#Q4
#Set the variables controlling for SES in X
X = df[['Education', 'SES']]
y = df['income']

#Add the intercept to the independent variable
X = sm.add_constant(X)

#Build/fit the model, then print out summary
model = sm.OLS(y, X).fit()
print(model.summary())

#Make predictions for income based on education from the model
predictions = model.predict(X)
print(predictions)

#Calculate the residual between actual and predicted y values
residuals = y - predictions

#Find RMSE and print it
rmse = np.sqrt(np.mean(residuals ** 2))
print(rmse)

#%% D5
#Q2
#means = df[data].mean(axis=0, skipna=True)
#Set an empty list to store the confidence intervals of each movie
conf_int = []
for col in df[data].columns:
    #Get all ratings, calculate standard error, then 95% confidence interval
    #Then add/subtract from the mean and store them in the list
    ratings = df[col].dropna()
    n = len(ratings)
    mean = means[col]
    std_err = stats.sem(ratings)
    h = std_err * stats.t.ppf((1 + 0.95) / 2, n - 1)
    conf_int.append((mean - h, mean + h))

#Convert to array
conf_int = np.array(conf_int)
print(conf_int)

#Q3
#Sort the movies by the mean
sorted_by_mean = means.sort_values()
print(sorted_by_mean)

#Q4
#Find the widths of confidence interval and sort the movies by that order
ci_widths = conf_int[:, 1] - conf_int[:, 0]
sorted_by_widths = means.index[np.argsort(ci_widths)]
print(sorted_by_widths)

#%% D6
#Q2
#Kill Bill: Vol. 1 in Column 314, Kill Bill: Vol. 2 in Column 177, Pulp Fiction in Column 298
all_three = df.iloc[:, [313, 176, 297]].dropna()

#Q3
#Get the mean and median for all 3 movies and print them out
all_three_mean = all_three.mean(axis=0)
all_three_median = all_three.median(axis=0)

print("Mean:\n", all_three_mean)
print("Median:\n", all_three_median)

#Q4
#Separate each movie in its own column from the filtered column
kill_bill_1 = all_three.iloc[:, 0]
kill_bill_2 = all_three.iloc[:, 1]
pulp_fiction = all_three.iloc[:, 2]

#Perform an independent-samples t-test in all possible combinations and return results
#Set equal_var to 0 or False to not assume any equal variances being tested (Welch's t-test)
ind_ttest_kb1_kb2 = stats.ttest_ind(kill_bill_1, kill_bill_2, equal_var=0)
ind_ttest_kb1_pf = stats.ttest_ind(kill_bill_1, pulp_fiction, equal_var=0)
ind_ttest_kb2_pf = stats.ttest_ind(kill_bill_2, pulp_fiction, equal_var=0)

print("Independent Samples T-Test:")
print("Kill Bill: Vol. 1 vs. Kill Bill: Vol. 2\n", ind_ttest_kb1_kb2)
print("Kill Bill: Vol. 1 vs. Pulped Fiction\n", ind_ttest_kb1_pf)
print("Kill Bill: Vol. 2 vs. Pulped Fiction\n", ind_ttest_kb2_pf)

#Q5
#Perform a paired-samples t-test in all possible combinations and return results
paired_ttest_kb1_kb2 = stats.ttest_rel(kill_bill_1, kill_bill_2)
paired_ttest_kb1_pf = stats.ttest_rel(kill_bill_1, pulp_fiction)
paired_ttest_kb2_pf = stats.ttest_rel(kill_bill_2, pulp_fiction)

print("\nPaired Samples T-Test:")
print("Kill Bill: Vol. 1 vs. Kill Bill: Vol. 2\n", paired_ttest_kb1_kb2)
print("Kill Bill: Vol. 1 vs. Pulped Fiction\n", paired_ttest_kb1_pf)
print("Kill Bill: Vol. 2 vs. Pulped Fiction\n", paired_ttest_kb2_pf)

#%% D7
#Q3
ij_lost_ark = df.iloc[:, 33].dropna()
ij_last_crusade = df.iloc[:, 4].dropna()
ij_kingdom_crystal_skull = df.iloc[:, 142].dropna()
ghostbusters = df.iloc[:, 149].dropna()
wolf_wall_st = df.iloc[:, 357].dropna()
interstellar = df.iloc[:, 95].dropna()
finding_nemo = df.iloc[:, 138].dropna()

#Use non-parametric tests below assuming alpha level = 0.05 (default value)
#Q4
#Median Test between Raiders of Lost Ark and Last Crusade –
#Mann-Whitney U Test for Independent, Wilcoxon Signed-Rank Test for Dependent
paired_4 = df.iloc[:, [33, 4]].dropna()
stats_4a, p_4a = mannwhitneyu(ij_lost_ark, ij_last_crusade, alternative='two-sided')
stats_4b, p_4b = wilcoxon(paired_4.iloc[:, 0], paired_4.iloc[:, 1])
print(f"Median Test between Raiders of Lost Ark and Last Crusade: U = {stats_4a}, p = {p_4a}")
print(f"Joint Median Test between Raiders of Lost Ark and Last Crusade: Stat = {stats_4b}, p = {p_4b}")

#Q5
#Median Test between Last Crusade and Kingdom of Crystal Skull –
#Mann-Whitney U Test for Independent, Wilcoxon Signed-Rank Test for Dependent
paired_5 = df.iloc[:, [4, 142]].dropna()
stats_5a, p_5a = mannwhitneyu(ij_last_crusade, ij_kingdom_crystal_skull, alternative='two-sided')
stats_5b, p_5b = wilcoxon(paired_5.iloc[:, 0], paired_5.iloc[:, 1])
print(f"Median Test between Last Crusade and Kingdom of Crystal Skull: U = {stats_5a}, p = {p_5a}")
print(f"Joint Median Test between Last Crusade and Kingdom of Crystal Skull: Stat = {stats_5b}, p = {p_5b}")

#Q6
#Median Test between Kingdom of Crystal Skull and Ghostbusters –
#Mann-Whitney U Test for Independent, Wilcoxon Signed-Rank Test for Dependent
paired_6 = df.iloc[:, [142, 149]].dropna()
stats_6a, p_6a = mannwhitneyu(ij_kingdom_crystal_skull, ghostbusters, alternative='two-sided')
stats_6b, p_6b = wilcoxon(paired_6.iloc[:, 0], paired_6.iloc[:, 1])
print(f"Median Test between Kingdom of Crystal Skull and Ghostbusters: U = {stats_6a}, p = {p_6a}")
print(f"Joint Median Test between Kingdom of Crystal Skull and Ghostbusters: Stat = {stats_6b}, p = {p_6b}")

#Q7
#Distribution Test between Ghostbusters and Finding Nemo – KS Test
paired_7 = df.iloc[:, [149, 138]].dropna()
stats_7a, p_7a = ks_2samp(ghostbusters, finding_nemo)
stats_7b, p_7b = ks_2samp(paired_7.iloc[:, 0], paired_7.iloc[:, 1])
print(f"Distribution Test between Ghostbusters and Finding Nemo: KS = {stats_7a}, p = {p_7a}")
print(f"Joint Distribution Test between Ghostbusters and Finding Nemo: KS = {stats_7b}, p = {p_7b}")

#Q8
#Distribution Test between Finding Nemo and Interstellar – KS Test
paired_8 = df.iloc[:, [138, 95]].dropna()
stats_8a, p_8a = ks_2samp(finding_nemo, interstellar)
stats_8b, p_8b = ks_2samp(paired_8.iloc[:, 0], paired_8.iloc[:, 1])
print(f"Distribution Test between Finding Nemo and Interstellar: KS = {stats_8a}, p = {p_8a}")
print(f"Joint Distribution Test between Finding Nemo and Interstellar: KS = {stats_8b}, p = {p_8b}")

#Q9
#Distribution Test between Interstellar and Wolf of Wall Street – KS Test
paired_9 = df.iloc[:, [95, 357]].dropna()
stats_9a, p_9a = ks_2samp(interstellar, wolf_wall_st)
stats_9b, p_9b = ks_2samp(paired_9.iloc[:, 0], paired_9.iloc[:, 1])
print(f"Distribution Test between Interstellar and Wolf of Wall Street: KS = {stats_9a}, p = {p_9a}")
print(f"Joint Distribution Test between Interstellar and Wolf of Wall Street: KS = {stats_9b}, p = {p_9b}")

#Q10
#Median Test between Interstellar and Wolf of Wall Street –
#Mann-Whitney U Test for Independent, Wilcoxon Signed-Rank Test for Dependent
#Use the paired data from previous question
stats_10a, p_10a = mannwhitneyu(interstellar, wolf_wall_st, alternative='two-sided')
stats_10b, p_10b = wilcoxon(paired_9.iloc[:, 0], paired_9.iloc[:, 1])
print(f"Median Test between Interstellar and Wolf of Wall Street: U = {stats_10a}, p = {p_10a}")
print(f"Joint Median Test between Interstellar and Wolf of Wall Street: Stat = {stats_10b}, p = {p_10b}")

#%% D8
#Q2
#Calculate mean of each column
means_8_2 = df[data].mean(axis=0, skipna=True)

#Bootstrap to resample for Confidence Intervals
ci_95 = [] #Store 95% confidence interval values
ci_99 = [] #Store 99% confidence interval values

#Iterate through each column for each movie
for column in df[data].columns:
    #Get the data for each movie, removing all NaN values
    movie_column = df[data][column].dropna().values
    
    #Bootstrapping for 95% Confidence Interval and storing the values in ci_95 list
    res_95 = bootstrap((movie_column,), np.mean, n_resamples=1000, confidence_level=0.95)
    ci_95.append((res_95.confidence_interval.low, res_95.confidence_interval.high))
    
    #Bootstrapping for 99% Confidence Interval and storing the values in ci_99 list
    res_99 = bootstrap((movie_column,), np.mean, n_resamples=1000, confidence_level=0.99)
    ci_99.append((res_99.confidence_interval.low, res_99.confidence_interval.high))

#Convert the lists into an array
ci_95 = np.array(ci_95)
ci_99 = np.array(ci_99)

#Calculate the widths for each pair of data (High - Low) and store values in a separate array
ci_95_width = ci_95[:, 1] - ci_95[:, 0]
ci_99_width = ci_99[:, 1] - ci_99[:, 0]

#Combine the arrays together into a dataframe
combined_d8 = pd.DataFrame({'Movie': means_8_2.index, 'Mean': means_8_2.values, '95% Confidence Interval Low': ci_95[:, 0], '95% Confidence Interval High': ci_95[:, 1], '95% Confidence Interval Width': ci_95_width, '99% Confidence Interval Low': ci_99[:, 0], '99% Confidence Interval High': ci_99[:, 1], '99% Confidence Interval Width': ci_99_width})

#Q3
#Sort the dataframe by mean
combined_d8_sort_by_mean = combined_d8.sort_values(by='Mean', ascending=True)
mean_8 = means_8_2.mean()

#Q4
#Sort the dataframe by 95% CI width and 99% CI width
combined_d8_sort_by_95_width = combined_d8.sort_values(by='95% Confidence Interval Width', ascending=True)
combined_d8_sort_by_99_width = combined_d8.sort_values(by='99% Confidence Interval Width', ascending=True)

#%% D10
#Q2
#Retrieve the columns and drop NaN values
sen_seek = df.iloc[:, 400:420].dropna()
personality = df.iloc[:, 420:464].dropna()
mov_exp = df.iloc[:, 464:474].dropna()

#Z-score the data before doing PCA as non z-scored data would yield nonsense results
zscored_sen_seek = stats.zscore(sen_seek)
zscored_personality = stats.zscore(personality)
zscored_mov_exp = stats.zscore(mov_exp)

#Now run the PCA
pca_sen_seek = PCA().fit(zscored_sen_seek)
pca_personality = PCA().fit(zscored_personality)
pca_mov_exp = PCA().fit(zscored_mov_exp)

#Set Kaiser's Criterion to 1 and count how many components for the data
kaisersCriterion = 1
#Find the eigenvalues in decreasing order of magnitude
eigVals_sen_seek = pca_sen_seek.explained_variance_
eigVals_personality = pca_personality.explained_variance_
eigVals_mov_exp = pca_mov_exp.explained_variance_
#Find the components
kaisers_sen_seek = np.count_nonzero(eigVals_sen_seek > kaisersCriterion)
kaisers_personality = np.count_nonzero(eigVals_personality > kaisersCriterion)
kaisers_mov_exp = np.count_nonzero(eigVals_mov_exp > kaisersCriterion)

print(f"Kaiser's Critierion for Sensational Seeking: {kaisers_sen_seek}")
print(f"Kaiser's Critierion for Personality: {kaisers_personality}")
print(f"Kaiser's Critierion for Movie Experience: {kaisers_mov_exp}")

#Q3
#Find ratings for Saw and median
saw_full = df.iloc[:, 341]
saw = saw_full.dropna()
saw_median = np.median(saw)

#Label each row comparing them to median, except values that are equal
#Do it on full data, to jointly compare this with columns from Q2 in Q4
saw_labels = saw_full.apply(
    lambda x: 0 if x < saw_median else 1 if x > saw_median else np.nan)

#Q4
#Combine original columns with saw median labels then drop NaNs
sen_seek_full = df.iloc[:, 400:420]
personality_full = df.iloc[:, 420:464]
mov_exp_full = df.iloc[:, 464:474]
sen_seek_saw = pd.concat([sen_seek_full, saw_labels], axis=1).dropna()
personality_saw = pd.concat([personality_full, saw_labels], axis=1).dropna()
mov_exp_saw = pd.concat([mov_exp_full, saw_labels], axis=1).dropna()

#Find new z-scores and first principal component scores for the characteristics
new_sen_seek = sen_seek_saw.iloc[:, 0:20]
new_personality = personality_saw.iloc[:, 0:44]
new_mov_exp = mov_exp_saw.iloc[:, 0:10]

zscored2_sen_seek = stats.zscore(new_sen_seek)
zscored2_personality = stats.zscore(new_personality)
zscored2_mov_exp = stats.zscore(new_mov_exp)

pc1_sen_seek = pca_sen_seek.transform(zscored2_sen_seek)[:, 0]
pc1_personality = pca_personality.transform(zscored2_personality)[:, 0]
pc1_mov_exp = pca_mov_exp.transform(zscored2_mov_exp)[:, 0]

#Create three new dataframes that combine both principal component scores and medians
sen_seek_saw_full = pd.DataFrame({"PC_Sen_Seek": pc1_sen_seek, "Saw": sen_seek_saw.iloc[:, 20]})
personality_saw_full = pd.DataFrame({"PC_Personality": pc1_personality, "Saw": personality_saw.iloc[:, 44]})
mov_exp_saw_full = pd.DataFrame({"PC_Mov_Exp": pc1_mov_exp, "Saw": mov_exp_saw.iloc[:, 10]})

#Logistic Regression OLS that returns both beta and p-values
def logistic_regression(X, y):
    X = sm.add_constant(X)  # Add intercept
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result.params, result.pvalues

#Create the three Logistic Regression models to predict liking/disliking
X1 = sen_seek_saw_full['PC_Sen_Seek']
y1 = sen_seek_saw_full['Saw']
sen_seek_saw_beta, sen_seek_saw_pval = logistic_regression(X1, y1)

X2 = personality_saw_full['PC_Personality']
y2 = personality_saw_full['Saw']
personality_saw_beta, personality_saw_pval = logistic_regression(X2, y2)

X3 = mov_exp_saw_full['PC_Mov_Exp']
y3 = mov_exp_saw_full['Saw']
mov_exp_saw_beta, mov_exp_saw_pval = logistic_regression(X3, y3)

print(f"Sensation Seeking Beta: {sen_seek_saw_beta['PC_Sen_Seek']:.2f}, p-value: {sen_seek_saw_pval['PC_Sen_Seek']:.3f}")
print(f"Personality Beta: {personality_saw_beta['PC_Personality']:.2f}, p-value: {personality_saw_pval['PC_Personality']:.3f}")
print(f"Movie Experience Beta: {mov_exp_saw_beta['PC_Mov_Exp']:.2f}, p-value: {mov_exp_saw_pval['PC_Mov_Exp']:.3f}")



