# [[file:carbon_expectation.org::*Code][Code:1]]
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 500)

###########################################
# input data specific to the Yuba project #
###########################################
net_emissions = np.array([1221, 1355, 3394, 5905, 9539, 12300, 15618, 18891]) # 8 timesteps (year 5 ... 40)

# carbon stocks from FVS simulations; columns are simulation timesteps, rows are wildfire years; the values where wildfire timestep > sim timestep are not needed
c_b_df = pd.DataFrame({"WF": ["NOWF", "WF24", "WF29", "WF34", "WF39", "WF44", "WF49", "WF54", "WF59", "WF64"],
                       0:  [47121511, 41735415, 47121511, 47121511, 47121511, 47121511, 47121511, 47121511, 47121511, 47121511],
                       5:  [51910402, 42803181, 46087206, 51910402, 51910402, 51910402, 51910402, 51910402, 51910402, 51910402],
                       10: [56980196, 44444962, 47529288, 50318851, 56980196, 56980196, 56980196, 56980196, 56980196, 56980196],
                       15: [62188494, 46070930, 49431594, 51369360, 54705984, 62188494, 62188494, 62188494, 62188494, 62188494],
                       20: [67471620, 47886448, 51394522, 52923693, 55288959, 59280747, 67471620, 67471620, 67471620, 67471620],
                       25: [72737463, 49849730, 53498553, 54410660, 56308804, 59346363, 63860571, 72737463, 72737463, 72737463],
                       30: [77914663, 52041020, 55821834, 56144648, 57221106, 59738735, 63657885, 68376796, 77914663, 77914663],
                       35: [82865616, 54425580, 58271719, 58103287, 58300674, 60089892, 63757500, 68087556, 72609031, 82865616],
                       40: [87630606, 56892290, 60792791, 60345526, 59755082, 60571424, 63810526, 68038155, 72020578, 76662523]}).set_index("WF")

c_p_df = pd.DataFrame({"WF": ["NOWF", "WF24", "WF29", "WF34", "WF39", "WF44", "WF49", "WF54", "WF59", "WF64"],
                       0:  [47118283, 41732572, 47118283, 47118283, 47118283, 47118283, 47118283, 47118283, 47118283, 47118283],
                       5:  [51905750, 42799733, 46083189, 51905750, 51905750, 51905750, 51905750, 51905750, 51905750, 51905750],
                       10: [56974474, 44440852, 47526026, 50314545, 56974474, 56974474, 56974474, 56974474, 56974474, 56974474],
                       15: [62181700, 46065930, 49428959, 51366414, 54700688, 62181700, 62181700, 62181700, 62181700, 62181700],
                       20: [67463499, 47880443, 51392319, 52922112, 55284740, 59274804, 67463499, 67463499, 67463499, 67463499],
                       25: [72727523, 49842731, 53496814, 54410230, 56305315, 59343083, 63852758, 72727523, 72727523, 72727523],
                       30: [77903457, 52033086, 55820453, 56144838, 57218453, 59738276, 63650989, 68368026, 77903457, 77903457],
                       35: [82853536, 54416620, 58270716, 58104052, 58298768, 60092432, 63751007, 68079148, 72599659, 82853536],
                       40: [87617543, 56882588, 60792249, 60346532, 59753532, 60576835, 63804388, 68029603, 72011353, 76652291]}).set_index("WF")

ABP = 0.028
delta_t = 5
###########################################

print(f"Baseline carbon stocks\n {c_b_df}")
#print(f"Project carbon stocks\n {c_p_df}")

# current method
# temporarily use a placeholder of 1 for the NOWF values
naive_probs = pd.DataFrame.from_dict({interval - 5: [1] + [ABP] + [ABP * 5 if wf_yr < (interval -5) / 5 else 0 for wf_yr in np.arange(8)] for interval in np.arange(5, 50, 5)})
naive_probs.iloc[0] = 1 - naive_probs.iloc[1:].sum()
# crudely update the NOWF values so that prob(NOWF) can't be negative
naive_probs.iloc[0][naive_probs.iloc[0] < 0] = 0

# Poisson process method
# preserve NOWF and WF probabilities in a single dict (following REM notation for time periods)
poisson_probs = {interval - 5: [np.exp(-ABP * interval)] + [np.exp(-ABP * wf_yr * delta_t) * ABP * delta_t if wf_yr * delta_t < interval else 0 for wf_yr in np.arange(9)] for interval in np.arange(5, 50, 5)}

def abridged_FMU_calcs(probs: pd.DataFrame, c_b_df: pd.DataFrame, c_p_df: pd.DataFrame, ABP: float, delta_t: int):
    '''
    Print a bunch of intermediate outputs and then the final FMU values.
    '''

    expectation_wts = pd.DataFrame.from_dict(probs)
    expectation_wts["WF"] = ["NOWF", "WF24", "WF29", "WF34", "WF39", "WF44", "WF49", "WF54", "WF59" , "WF64"]
    expectation_wts = expectation_wts.set_index("WF")

    print(f"Expectation weights\n {expectation_wts}")
    print(f"Shouldn't each of these sum to 1?\n {expectation_wts.sum()}")

    # baseline and project expected carbon stocks; we only need the diagonals
    c_b_exp = pd.Series(np.diag(c_b_df.T.dot(expectation_wts)))
    print(f"Expected Baseline carbon stocks\n {c_b_exp}")
    c_p_exp = pd.Series(np.diag(c_p_df.T.dot(expectation_wts)))
    print(f"Expected Project carbon stocks\n {c_p_exp}")

    net_c_stocks = c_p_exp - c_b_exp
    print(f"Net expected carbon stocks\n {net_c_stocks}")

    # final calculations; all that matters is the final (gross) FMU value
    fmu = net_emissions + net_c_stocks[1:]
    print(f"FMU values\n {fmu}")

#abridged_FMU_calcs(naive_probs, c_b_df, c_p_df, ABP, delta_t) # FMU = 14,444
abridged_FMU_calcs(poisson_probs, c_b_df, c_p_df, ABP, delta_t) # FMU = 12,262
# Code:1 ends here
