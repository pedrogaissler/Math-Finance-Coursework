import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

bonds = np.array([np.array([0.25, "CA135087M359", np.datetime64("2021-05-14"), np.datetime64("2023-08-01"),
                            97.8, 97.801, 97.833, 97.841, 97.828, 97.845, 97.866, 97.903, 97.909, 97.903]),
np.array([2.25, "CA135087J546", np.datetime64("2018-10-05"), np.datetime64("2024-03-01"),
          97.95, 97.86, 97.97, 97.91, 97.87, 97.86, 97.87, 97.95, 97.93, 97.84]),
np.array([1.5, "CA135087J967", np.datetime64("2019-04-05"), np.datetime64("2024-09-01"),
          96.27, 96.35, 96.34, 96.46, 96.39, 96.34, 96.22, 96.26, 96.4, 96.31]),
np.array([1.25, "CA135087K528", np.datetime64("2019-10-11"), np.datetime64("2025-03-01"),
          95.405, 95.455, 95.62, 95.595, 95.43, 95.39, 95.34, 95.45, 95.485, 95.305]),
np.array([0.5, "CA135087K940", np.datetime64("2020-04-03"), np.datetime64("2025-09-01"),
          92.765, 92.805, 93.035, 93.06, 92.885, 92.84, 92.79, 92.935, 92.965, 92.76]),
np.array([0.25, "CA135087L518", np.datetime64("2020-10-09"), np.datetime64("2026-03-01"),
          91.095, 91.20, 91.485, 91.53, 91.29, 91.265, 91.215, 91.375, 91.42, 91.165]),
np.array([1, "CA135087L930", np.datetime64("2021-04-06"), np.datetime64("2026-09-01"),
          92.755, 92.87, 93.27, 93.29, 92.945, 92.935, 92.915, 93.005, 92.975, 92.77]),
np.array([1.25, "CA135087M847", np.datetime64("2021-10-15"), np.datetime64("2027-03-01"),
          93.175, 93.3, 93.75, 93.78, 93.375, 93.355, 93.345, 93.44, 93.385, 93.15]),
np.array([2.75, "CA135087N837", np.datetime64("2022-05-13"), np.datetime64("2027-09-01"),
          99.145, 99.235, 99.73, 99.70, 99.215, 99.18, 99.16, 99.255, 99.15, 98.86]),
np.array([3.5, "CA135087P576", np.datetime64("2022-10-21"), np.datetime64("2028-03-01"),
          102.79, 102.88, 103.425, 103.375, 102.805, 102.765, 102.765, 102.855, 102.725, 102.405])])

days = np.array([np.datetime64("2023-01-16"), np.datetime64("2023-01-17"), np.datetime64("2023-01-18"),
                 np.datetime64("2023-01-19"), np.datetime64("2023-01-20"), np.datetime64("2023-01-23"),
                 np.datetime64("2023-01-24"), np.datetime64("2023-01-25"), np.datetime64("2023-01-26"),
                 np.datetime64("2023-01-27")])

maturities = bonds[:, 3]
prices = bonds[:, (4, 5, 6, 7, 8, 9, 10, 11, 12, 13)]

# Getting last coupon payments paid, to calculate accrued interest
payment = np.column_stack(maturities).T
while (payment > days[0]).any():
    payment = payment - np.timedelta64(180, "D")*(payment > days[0])

# CAN 3.5 Mar 28 coupon has issue date after the last estimated coupon payment date. We consider its issue
# date as being the last coupon payment date
payment[-1][0] = np.datetime64("2022-10-21")

# Calculating accrued interest
times = np.array([np.array([(days[day] - payment[bond]).astype(int) for bond in range(10)]) for day in range(10)])/365
accrued_int = bonds[:, 0]*times[:, :, 0]

# Calculating time until maturity
time_until_mat = np.array([np.array([(bonds[bond, 3] - days[day]).astype(int)
                                     for day in range(10)]) for bond in range(10)])/365

yield_days = np.zeros((10, 10))  # Initialising yield rates every day
forward_days = np.zeros((10, 4))  # Initialising forward rates for 1yr-1yr, 1yr-2yr, 1yr-3yr, 1yr-4yr
spot_days = np.zeros((10, 10))  # Initialising spot rates every day

for day in range(yield_days.shape[0]):
    for bond in range(yield_days.shape[1]):
        # Looping over yields
        # Creating a vector of all realised coupons
        coupon = np.append(np.repeat(bonds[bond, 0] / 2, bond + 1), bonds[bond, 0] / 2 + 100)
        power = np.append(0, time_until_mat[:, day])  # Time until maturity for each bond at day <day>
        # Creating a fair zero value function to minimise
        inner_prod = lambda x: coupon.T @ (np.repeat(np.exp(-x), bond + 2) ** power[: bond + 2]) \
                               - accrued_int[day, bond] - prices[bond, day]
        yield_days[day, bond] = sc.optimize.fsolve(inner_prod, x0 = 0.1)

        # Looping over spots
        # Considering only realised coupons at time <day>
        indicator = np.array([i < bond + 1 for i in range(spot_days.shape[1] + 1)]) / 2
        spot_scale = np.append(0, spot_days[day])  # Dummy variable for spots at a certain day
        discount_vec = np.exp(-spot_scale * np.append(0, time_until_mat[:, day]))  # Vector of discounting factors
        # Zero coupon price
        adjusted_price = prices[bond, day] + accrued_int[day, bond] - (bonds[bond, 0] * indicator) @ discount_vec
        # Zero coupon rate
        spot_days[day, bond] = -np.log(adjusted_price / (100 + bonds[bond, 0] / 2)) / time_until_mat[bond, day]

    for forward in range(forward_days.shape[1]):
        # Looping over forwards
        latest = time_until_mat[2 * (forward + 1) + 1, day]  # Last realised zero rate date
        earliest = time_until_mat[2 * forward + 1, day]  # First realised zero rate date
        time_difference = latest - earliest
        latest_spot = spot_days[day, 2 * (forward + 1) + 1]  # Last realised zero rate
        earliest_spot = spot_days[day, 2 * forward + 1]  # First realised zero rate
        forward_days[day, forward] = (latest_spot * latest - earliest_spot * earliest) / time_difference

# Estimating spots and yields between now and in 6 months with linear interpolation
mvals_yield_slope = 2 * (yield_days[:, 1] - yield_days[:, 0])
mvals_spot_slope = 2 * (spot_days[:, 1] - spot_days[:, 0])
mvals_fill_yield = yield_days[:, 1] - mvals_yield_slope
mvals_fill_spot = spot_days[:, 1] - mvals_spot_slope

# Appending the estimated values for yields and forwards between now and in 6 months
yield_days = np.column_stack([mvals_fill_yield, yield_days])
spot_days = np.column_stack([mvals_fill_spot, spot_days])

# Plotting yields
axis = np.array([year/2 for year in range(11)])
for day in range(10):
    plt.plot(axis, yield_days[day, :], label = days[day])
plt.title("0-5 Years Yield (YTM) Curves")
plt.legend()
plt.xlabel("Years")
plt.ylabel("Yield to Maturity Rate")
plt.show()

# Plotting spots
for day in range(10):
    plt.plot(axis, spot_days[day, :], label = days[day])
plt.title("0-5 Years Spot Rate Curves")
plt.legend()
plt.xlabel("Years")
plt.ylabel("Spot Rate")
plt.show()

# Plotting forwards
axis = np.array([2 + year for year in range(4)])
for day in range(10):
    plt.plot(axis, forward_days[day, :], label = days[day])
plt.title("2-5 Years 1-Year Forward Curves")
plt.legend()
plt.xlabel("Years")
plt.ylabel("Forward Rate")
plt.show()

# Choosing 1, 2, 3, 4, 5 year yields yields (columns: 2, 4, 6, 8, 10)
yield_days = yield_days[:, (2, 4, 6, 8, 10)]

# Creating daily log returns of yields and forwards
log_yields = np.array([np.log(yield_days[day, :]/yield_days[day - 1, :]) for day in range(1, yield_days.shape[0])])
log_forwards = np.array([np.log(forward_days[day, :]/forward_days[day - 1, :]) for day in range(1, forward_days.shape[0])])

# Calculating their sample covariances
cov_yields = np.cov(log_yields.T)
cov_forwards = np.cov(log_forwards.T)

# Calculating eigenvalues and eigenvectors of covariances
yield_eig = np.linalg.eig(cov_yields)
forward_eig = np.linalg.eig(cov_forwards)

print(log_yields, "Daily Log Returns of Yields")
print(log_forwards, "Daily Log Returns of Forwards")

print(cov_yields, "Covariances of Daily Log Returns of Yields")
print(cov_forwards, "Covariances of Daily Log Returns of Forwards")

print(yield_eig, "Eigenvalues and Eigenvectors of Covariance Matrix of Daily Log Returns of Yields")
print(forward_eig, "Eigenvalues and Eigenvectors of Covariance Matrix of Daily Log Returns of Forwards")

print(np.trace(cov_yields), "Trace of the Covariance Matrix of Daily Log Returns of Yields")
print(np.trace(cov_forwards), "Trace of the Covariance Matrix of Daily Log Returns of Forwards")

print(yield_eig[1][0] + log_yields.mean(0), "First Principal Component of daily log returns of yields, "
                                            "after adding time series mean to eigenvector")
print(forward_eig[1][0] + log_forwards.mean(0), "First Principal Component of daily log returns of forwards, "
                                                "after adding time series mean to eigenvector")
