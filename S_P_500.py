
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math

# Other packages
import numpy as np
import pandas as pd
import re
import scipy as sp
import xgboost as xgb
import matplotlib.pyplot as plt
from numpy import concatenate
from pandas import DataFrame, Series, read_csv, scatter_matrix

import warnings
warnings.filterwarnings("ignore")

import quandl
quandl.ApiConfig.api_key = '4rXnmHqzy9AuBuWVdy9e'
quandl.ApiConfig.api_version = '2015-04-09'


# In[24]:


Activision_Blizzard = quandl.get("WIKI/ATVI",start_date="2006-10-20", end_date="2013-10-20") #Communication Services
Alphabet_Inc_Class_C = quandl.get("WIKI/GOOG",start_date="2006-10-20", end_date="2013-10-20")
Alphabet_Inc_Class_A = quandl.get("WIKI/GOOGL",start_date="2006-10-20", end_date="2013-10-20")
ATT  = quandl.get("WIKI/T",start_date="2006-10-20", end_date="2013-10-20")
CBS = quandl.get("WIKI/CBS",start_date="2006-10-20", end_date="2013-10-20")
CenturyLink = quandl.get("WIKI/CTL",start_date="2006-10-20", end_date="2013-10-20")
Charter_Communications = quandl.get("WIKI/CHTR",start_date="2006-10-20", end_date="2013-10-20")
Comcast = quandl.get("WIKI/CMCSA",start_date="2006-10-20", end_date="2013-10-20")
Discovery_Class_A = quandl.get("WIKI/DISCA",start_date="2006-10-20", end_date="2013-10-20")
Discovery_Class_C = quandl.get("WIKI/DISCK",start_date="2006-10-20", end_date="2013-10-20")


#indexing

CBS.reset_index(inplace=True)
Activision_Blizzard.reset_index(inplace=True)
Alphabet_Inc_Class_C.reset_index(inplace=True)
Alphabet_Inc_Class_A.reset_index(inplace=True)
ATT.reset_index(inplace=True)
CenturyLink.reset_index(inplace=True)
Charter_Communications.reset_index(inplace=True)
Comcast.reset_index(inplace=True)
Discovery_Class_A.reset_index(inplace=True)
Discovery_Class_C.reset_index(inplace=True)

#Data Management
Activision_Blizzard = Activision_Blizzard [['Date','Open','Close']]
Activision_Blizzard.rename(columns={'Open': 'Activision_Blizzard_Open'},inplace=True)
Activision_Blizzard.rename(columns={'Close': 'Activision_Blizzard_Close'},inplace=True)



CBS = CBS [['Date','Open','Close']]
CBS.rename(columns={'Open': 'CBS_Open'},inplace=True)
CBS.rename(columns={'Close': 'CBS_Close'},inplace=True)

Alphabet_Inc_Class_C = Alphabet_Inc_Class_C [['Date','Open','Close']]
Alphabet_Inc_Class_C.rename(columns={'Open': 'Alphabet_Inc_Class_C_Open'},inplace=True)
Alphabet_Inc_Class_C.rename(columns={'Close': 'Alphabet_Inc_Class_C_Close'},inplace=True)

Alphabet_Inc_Class_A = Alphabet_Inc_Class_A [['Date','Open','Close']]
Alphabet_Inc_Class_A.rename(columns={'Open': 'Alphabet_Inc_Class_A_Open'},inplace=True)
Alphabet_Inc_Class_A.rename(columns={'Close': 'Alphabet_Inc_Class_A_Close'},inplace=True)

ATT = ATT [['Date','Open','Close']]
ATT.rename(columns={'Open': 'ATT_Open'},inplace=True)
ATT.rename(columns={'Close': 'ATT_Close'},inplace=True)

CenturyLink = CenturyLink [['Date','Open','Close']]
CenturyLink.rename(columns={'Open': 'CenturyLink_Open'},inplace=True)
CenturyLink.rename(columns={'Close': 'CenturyLink_Close'},inplace=True)

Charter_Communications = Charter_Communications [['Date','Open','Close']]
Charter_Communications.rename(columns={'Open': 'Charter_Communications_Open'},inplace=True)
Charter_Communications.rename(columns={'Close': 'Charter_Communications_Close'},inplace=True)


Comcast = Comcast [['Date','Open','Close']]
Comcast.rename(columns={'Open': 'Comcast_Open'},inplace=True)
Comcast.rename(columns={'Close': 'Comcast_Close'},inplace=True)


Discovery_Class_A = Discovery_Class_A [['Date','Open','Close']]
Discovery_Class_A.rename(columns={'Open': 'Discovery_Class_A_Open'},inplace=True)
Discovery_Class_A.rename(columns={'Close': 'Discovery_Class_A_Close'},inplace=True)


Discovery_Class_C = Discovery_Class_C [['Date','Open','Close']]
Discovery_Class_C.rename(columns={'Open': 'Discovery_Class_C_Open'},inplace=True)
Discovery_Class_C.rename(columns={'Close': 'Discovery_Class_C_Close'},inplace=True)



# In[3]:


#Activision_Blizzard.shape
#CBS.shape
#CenturyLink.shape
#ATT.shape
#Alphabet_Inc_Class_C.shape
#Alphabet_Inc_Class_A.shape
#Comcast.shape
#Charter_Communications.shape
#Discovery_Class_A.shape
#Discovery_Class_A.shape

#Merging
#Communication_Services_1= (Activision_Blizzard).merge(CBS).merge(CenturyLink).merge(ATT).merge(Alphabet_Inc_Class_C).merge(Alphabet_Inc_Class_A).merge(Comcast).merge(Discovery_Class_A).merge(Discovery_Class_C).merge(Dish).merge(Electronic_Arts).merge(Facebook).merge(Interpublic_Group).merge(Netflix).merge(News_Corp_Class_A).merge(News_Corp_Class_B).merge(Omnicom_Group).merge(Take_Two_Interactive).merge(Twitter).merge(TripAdvisor).merge(TwentyFirst_Century_Fox_A).merge(TwentyFirst_Century_Fox_B).merge(Verizon_Communications).merge(Viacom).merge(The_Walt_Disney)
#Communication_Services_1
#Communication_Services_1


# In[8]:


Dish = quandl.get("WIKI/DISH",start_date="2006-10-20", end_date="2013-10-20") #Communication Services
Electronic_Arts = quandl.get("WIKI/EA",start_date="2006-10-20", end_date="2013-10-20")
Facebook = quandl.get("WIKI/FB",start_date="2006-10-20", end_date="2013-10-20")
Interpublic_Group = quandl.get("WIKI/IPG",start_date="2006-10-20", end_date="2013-10-20")
Netflix = quandl.get("WIKI/NFLX",start_date="2006-10-20", end_date="2013-10-20")
News_Corp_Class_A = quandl.get("WIKI/NWSA",start_date="2006-10-20", end_date="2013-10-20")
News_Corp_Class_B = quandl.get("WIKI/NWS",start_date="2006-10-20", end_date="2013-10-20")
Omnicom_Group = quandl.get("WIKI/OMC",start_date="2006-10-20", end_date="2013-10-20")
Take_Two_Interactive = quandl.get("WIKI/TTWO",start_date="2006-10-20", end_date="2013-10-20")
Twitter = quandl.get("WIKI/TWTR",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Dish.reset_index(inplace=True) ##Communication Services
Electronic_Arts.reset_index(inplace=True)
Facebook.reset_index(inplace=True)
Interpublic_Group.reset_index(inplace=True)
Netflix.reset_index(inplace=True)
News_Corp_Class_A.reset_index(inplace=True)
News_Corp_Class_B.reset_index(inplace=True)
Omnicom_Group.reset_index(inplace=True)
Take_Two_Interactive.reset_index(inplace=True)
Twitter.reset_index(inplace=True)

#Data Management
Dish = Dish [['Date','Open','Close']]
Dish.rename(columns={'Open': 'Dish_Open'},inplace=True)
Dish.rename(columns={'Close': 'Dish_Close'},inplace=True)


Electronic_Arts = Electronic_Arts [['Date','Open','Close']]
Electronic_Arts.rename(columns={'Open': 'Electronic_Arts_Open'},inplace=True)
Electronic_Arts.rename(columns={'Close': 'Electronic_Arts_Close'},inplace=True)

Facebook = Facebook [['Date','Open','Close']]
Facebook.rename(columns={'Open': 'Facebook_Open'},inplace=True)
Facebook.rename(columns={'Close': 'Facebook_Close'},inplace=True)

Interpublic_Group = Interpublic_Group [['Date','Open','Close']]
Interpublic_Group.rename(columns={'Open': 'Interpublic_Group_Open'},inplace=True)
Interpublic_Group.rename(columns={'Close': 'Interpublic_Group_Close'},inplace=True)

Netflix = Netflix [['Date','Open','Close']]
Netflix.rename(columns={'Open': 'Netflix_Open'},inplace=True)
Netflix.rename(columns={'Close': 'Netflix_Close'},inplace=True)

News_Corp_Class_A = News_Corp_Class_A [['Date','Open','Close']]
News_Corp_Class_A.rename(columns={'Open': 'News_Corp_Class_A_Open'},inplace=True)
News_Corp_Class_A.rename(columns={'Close': 'News_Corp_Class_A_Close'},inplace=True)

News_Corp_Class_B = News_Corp_Class_B [['Date','Open','Close']]
News_Corp_Class_B.rename(columns={'Open': 'News_Corp_Class_B_Open'},inplace=True)
News_Corp_Class_B.rename(columns={'Close': 'News_Corp_Class_B_Close'},inplace=True)

Omnicom_Group = Omnicom_Group [['Date','Open','Close']]
Omnicom_Group.rename(columns={'Open': 'Omnicom_Group_Open'},inplace=True)
Omnicom_Group.rename(columns={'Close': 'Omnicom_Group_Close'},inplace=True)


Take_Two_Interactive = Take_Two_Interactive [['Date','Open','Close']]
Take_Two_Interactive.rename(columns={'Open': 'Take_Two_Interactive_Open'},inplace=True)
Take_Two_Interactive.rename(columns={'Close': 'Take_Two_Interactive_Close'},inplace=True)


Twitter = Twitter [['Date','Open','Close']]
Twitter.rename(columns={'Open': 'Twitter_Open'},inplace=True)
Twitter.rename(columns={'Close': 'Twitter_Close'},inplace=True)



# In[9]:


TripAdvisor = quandl.get("WIKI/TRIP",start_date="2006-10-20", end_date="2013-10-20") #Communication Services
TwentyFirst_Century_Fox_A  = quandl.get("WIKI/FOXA",start_date="2006-10-20", end_date="2013-10-20")
TwentyFirst_Century_Fox_B  = quandl.get("WIKI/FOX",start_date="2006-10-20", end_date="2013-10-20")
Verizon_Communications = quandl.get("WIKI/VZ",start_date="2006-10-20", end_date="2013-10-20")
Viacom  = quandl.get("WIKI/VIAB",start_date="2006-10-20", end_date="2013-10-20")
The_Walt_Disney = quandl.get("WIKI/DIS",start_date="2006-10-20", end_date="2013-10-20")




#indexing
TripAdvisor.reset_index(inplace=True)
TwentyFirst_Century_Fox_B.reset_index(inplace=True)
TwentyFirst_Century_Fox_A.reset_index(inplace=True)
Verizon_Communications.reset_index(inplace=True)
Viacom.reset_index(inplace=True)
The_Walt_Disney.reset_index(inplace=True)

#Data Management
TripAdvisor = TripAdvisor [['Date','Open','Close']]
TripAdvisor.rename(columns={'Open': 'TripAdvisor_Open'},inplace=True)
TripAdvisor.rename(columns={'Close': 'TripAdvisor_Close'},inplace=True)


TwentyFirst_Century_Fox_B.reset_index(inplace=True)
TwentyFirst_Century_Fox_B = TwentyFirst_Century_Fox_B [['Date','Open','Close']]
TwentyFirst_Century_Fox_B.rename(columns={'Open': 'TwentyFirst_Century_Fox_B_Open'},inplace=True)
TwentyFirst_Century_Fox_B.rename(columns={'Close': 'TwentyFirst_Century_Fox_B_Close'},inplace=True)

TwentyFirst_Century_Fox_A.reset_index(inplace=True)
TwentyFirst_Century_Fox_A = TwentyFirst_Century_Fox_A [['Date','Open','Close']]
TwentyFirst_Century_Fox_A.rename(columns={'Open': 'TwentyFirst_Century_Fox_A_Open'},inplace=True)
TwentyFirst_Century_Fox_A.rename(columns={'Close': 'TwentyFirst_Century_Fox_A_Close'},inplace=True)

Verizon_Communications.reset_index(inplace=True)
Verizon_Communications = Verizon_Communications [['Date','Open','Close']]
Verizon_Communications.rename(columns={'Open': 'Verizon_Communications_Open'},inplace=True)
Verizon_Communications.rename(columns={'Close': 'Verizon_Communications_Close'},inplace=True)

Viacom.reset_index(inplace=True)
Viacom = Viacom [['Date','Open','Close']]
Viacom.rename(columns={'Open': 'Viacom_Open'},inplace=True)
Viacom.rename(columns={'Close': 'Viacom_Close'},inplace=True)

The_Walt_Disney.reset_index(inplace=True)
The_Walt_Disney = The_Walt_Disney [['Date','Open','Close']]
The_Walt_Disney.rename(columns={'Open': 'The_Walt_Disney_Open'},inplace=True)
The_Walt_Disney.rename(columns={'Close': 'The_Walt_Disney_Close'},inplace=True)





# In[6]:


#Communication_Services1 = pd.concat([Activision_Blizzard [['Date','Activision_Blizzard_Close']], CBS['CBS_Close'],Alphabet_Inc_Class_C['Alphabet_Inc_Class_C_Close'],Alphabet_Inc_Class_A['Alphabet_Inc_Class_A_Close'],ATT ['ATT_Close'],CenturyLink['CenturyLink_Close'],Charter_Communications['Charter_Communications_Close'],Comcast['Comcast_Close'],Discovery_Class_A['Discovery_Class_A_Close'],Discovery_Class_C['Discovery_Class_C_Close']],Dish['Dish_Close'],Electronic_Arts['Electronic_Arts_Close'],Facebook['Facebook_Close'],Interpublic_Group['Interpublic_Group_Close'],Netflix['Netflix_Close'],News_Corp_Class_A['News_Corp_Class_A_Close'], News_Corp_Class_B['News_Corp_Class_B_Close'],Omnicom_Group['Omnicom_Group_Close'],Take_Two_Interactive['Take_Two_Interactive_Close'],Twitter['Twitter_Close'],
 #                            TripAdvisor['TripAdvisor_Close'],TwentyFirst_Century_Fox_B['TwentyFirst_Century_Fox_B_Close'], TwentyFirst_Century_Fox_A['TwentyFirst_Century_Fox_A_Close'],Verizon_Communications['Verizon_Communications_Close'],Viacom['Viacom_Close'],The_Walt_Disney['The_Walt_Disney_Close']], axis=1)
#Alphabet_Inc_Class_C[['Close']],Alphabet_Inc_Class_A[['Close']]])

Communication_Services= (Activision_Blizzard).merge(CBS).merge(CenturyLink).merge(ATT).merge(Alphabet_Inc_Class_A).merge(Comcast).merge(Discovery_Class_A).merge(Discovery_Class_C).merge(Dish).merge(Electronic_Arts).merge(Facebook).merge(Interpublic_Group).merge(Netflix).merge(News_Corp_Class_A).merge(News_Corp_Class_B).merge(Omnicom_Group).merge(Take_Two_Interactive).merge(Twitter).merge(TripAdvisor).merge(TwentyFirst_Century_Fox_A).merge(TwentyFirst_Century_Fox_B).merge(Verizon_Communications).merge(Viacom).merge(The_Walt_Disney)
#.merge(Alphabet_Inc_Class_C)

Communication_Services

#Check null Values
#Communication_Services.isnull().sum()


Communication_Services_FFT = (Activision_Blizzard_fft_df).merge(ATT_fft_df).merge(CBS_fft_df).merge(CenturyLink_fft_df).merge(Comcast_fft_df).merge(Discovery_Class_A_fft_df)


# In[ ]:


Communication_Services 


# In[10]:


Advance_Auto_Parts = quandl.get("WIKI/AAP",start_date="2006-10-20", end_date="2013-10-20") #Consumer Discretionary
Amazon = quandl.get("WIKI/AMZN",start_date="2006-10-20", end_date="2013-10-20")
Aptiv = quandl.get("WIKI/APTV",start_date="2006-10-20", end_date="2013-10-20")
AutoZone = quandl.get("WIKI/AZO",start_date="2006-10-20", end_date="2013-10-20")
Best_Buy_Co  = quandl.get("WIKI/BBY",start_date="2006-10-20", end_date="2013-10-20")
BlockHR = quandl.get("WIKI/HRB",start_date="2006-10-20", end_date="2013-10-20")
Booking_Holdings = quandl.get("WIKI/BKNG",start_date="2006-10-20", end_date="2013-10-20")
BorgWarner = quandl.get("WIKI/BWA",start_date="2006-10-20", end_date="2013-10-20")
#Capri_Holdings = quandl.get("WIKI/CPRI",start_date="2006-10-20", end_date="2013-10-20")
Carmax  = quandl.get("WIKI/KMX",start_date="2006-10-20", end_date="2013-10-20")


#indexing
Advance_Auto_Parts.reset_index(inplace=True)
Amazon.reset_index(inplace=True)
Aptiv.reset_index(inplace=True)
AutoZone.reset_index(inplace=True)
Best_Buy_Co.reset_index(inplace=True)
BlockHR.reset_index(inplace=True)
Booking_Holdings.reset_index(inplace=True)
BorgWarner.reset_index(inplace=True)
#Capri_Holdings.reset_index(inplace=True)
Carmax.reset_index(inplace= True)


#Data Management
Advance_Auto_Parts = Advance_Auto_Parts [['Date','Open','Close']]
Advance_Auto_Parts.rename(columns={'Open': 'Advance_Auto_Parts_Open'},inplace=True)
Advance_Auto_Parts.rename(columns={'Close': 'Advance_Auto_Parts_Close'},inplace=True)

Amazon = Amazon [['Date','Open','Close']]
Amazon.rename(columns={'Open': 'Amazon_Open'},inplace=True)
Amazon.rename(columns={'Close': 'Amazon_Close'},inplace=True)

Aptiv = Aptiv [['Date','Open','Close']]
Aptiv.rename(columns={'Open': 'Aptiv_Open'},inplace=True)
Aptiv.rename(columns={'Close': 'Aptiv_Close'},inplace=True)


AutoZone = AutoZone [['Date','Open','Close']]
AutoZone.rename(columns={'Open': 'AutoZone_Open'},inplace=True)
AutoZone.rename(columns={'Close': 'AutoZone_Close'},inplace=True)


Best_Buy_Co = Best_Buy_Co [['Date','Open','Close']]
Best_Buy_Co.rename(columns={'Open': 'Best_Buy_Co_Open'},inplace=True)
Best_Buy_Co.rename(columns={'Close': 'Best_Buy_Co_Close'},inplace=True)


BlockHR = BlockHR [['Date','Open','Close']]
BlockHR.rename(columns={'Open': 'BlockHR_Open'},inplace=True)
BlockHR.rename(columns={'Close': 'BlockHR_Close'},inplace=True)


Booking_Holdings = Booking_Holdings [['Date','Open','Close']]
Booking_Holdings.rename(columns={'Open': 'Booking_Holdings_Open'},inplace=True)
Booking_Holdings.rename(columns={'Close': 'Booking_Holdings_Close'},inplace=True)


BorgWarner = BorgWarner [['Date','Open','Close']]
BorgWarner.rename(columns={'Open': 'BorgWarner_Open'},inplace=True)
BorgWarner.rename(columns={'Close': 'BorgWarner_Close'},inplace=True)

#Capri_Holdings.reset_index(inplace=True)
#Aptiv = Aptiv [['Date','Open','Close']]
#Aptiv.rename(columns={'Open': 'Aptiv_Open'},inplace=True)
#Aptiv.rename(columns={'Close': 'Aptiv_Close'},inplace=True)

Carmax = Carmax [['Date','Open','Close']]
Carmax.rename(columns={'Open': 'Carmax_Open'},inplace=True)
Carmax.rename(columns={'Close': 'Carmax_Close'},inplace= True)



# In[11]:


Carnival = quandl.get("WIKI/CCL",start_date="2006-10-20", end_date="2013-10-20") #Consumer Discretionary
Chipotle_Mexican_Grill  = quandl.get("WIKI/CMG",start_date="2006-10-20", end_date="2013-10-20")
D_R_Horton  = quandl.get("WIKI/DHI",start_date="2006-10-20", end_date="2013-10-20")
Darden_Restaurants  = quandl.get("WIKI/DRI",start_date="2006-10-20", end_date="2013-10-20")
Dollar_General = quandl.get("WIKI/DG",start_date="2006-10-20", end_date="2013-10-20")
Dollar_Tree = quandl.get("WIKI/DLTR",start_date="2006-10-20", end_date="2013-10-20")
eBay = quandl.get("WIKI/EBAY",start_date="2006-10-20", end_date="2013-10-20")
Expedia_Group = quandl.get("WIKI/EXPE",start_date="2006-10-20", end_date="2013-10-20")
Foot_Locker = quandl.get("WIKI/FL",start_date="2006-10-20", end_date="2013-10-20")
Ford_Motor = quandl.get("WIKI/F",start_date="2006-10-20", end_date="2013-10-20")


#indexing
Carnival.reset_index(inplace=True)
Chipotle_Mexican_Grill.reset_index(inplace=True)
D_R_Horton.reset_index(inplace=True)
Darden_Restaurants.reset_index(inplace=True)
Dollar_General.reset_index(inplace=True)
Dollar_Tree.reset_index(inplace=True)
eBay.reset_index(inplace=True)
Expedia_Group.reset_index(inplace=True)
Foot_Locker.reset_index(inplace=True)
Ford_Motor.reset_index(inplace=True)


#Removing/Renaming columns

Carnival = Carnival [['Date','Open','Close']]
Carnival.rename(columns={'Open': 'Carnival_Open'},inplace=True)
Carnival.rename(columns={'Close': 'Carnival_Close'},inplace=True)

Chipotle_Mexican_Grill = Chipotle_Mexican_Grill [['Date','Open','Close']]
Chipotle_Mexican_Grill.rename(columns={'Open': 'Chipotle_Mexican_Grill_Open'},inplace=True)
Chipotle_Mexican_Grill.rename(columns={'Close': 'Chipotle_Mexican_Grill_Close'},inplace=True)


D_R_Horton = D_R_Horton [['Date','Open','Close']]
D_R_Horton.rename(columns={'Open': 'D_R_Horton_Open'},inplace=True)
D_R_Horton.rename(columns={'Close': 'D_R_Horton_Close'},inplace=True)

Darden_Restaurants = Darden_Restaurants [['Date','Open','Close']]
Darden_Restaurants.rename(columns={'Open': 'Darden_Restaurants_Open'},inplace=True)
Darden_Restaurants.rename(columns={'Close': 'Darden_Restaurants_Close'},inplace=True)

Dollar_General = Dollar_General [['Date','Open','Close']]
Dollar_General.rename(columns={'Open': 'Dollar_General_Open'},inplace=True)
Dollar_General.rename(columns={'Close': 'Dollar_General_Close'},inplace=True)


Dollar_Tree = Dollar_Tree [['Date','Open','Close']]
Dollar_Tree.rename(columns={'Open': 'Dollar_Tree_Open'},inplace=True)
Dollar_Tree.rename(columns={'Close': 'Dollar_Tree_Close'},inplace=True)

eBay = eBay [['Date','Open','Close']]
eBay.rename(columns={'Open': 'eBay_Open'},inplace=True)
eBay.rename(columns={'Close': 'eBay_Close'},inplace=True)

Expedia_Group = Expedia_Group [['Date','Open','Close']]
Expedia_Group.rename(columns={'Open': 'Expedia_Group_Open'},inplace=True)
Expedia_Group.rename(columns={'Close': 'Expedia_Group_Close'},inplace=True)


Foot_Locker = Foot_Locker [['Date','Open','Close']]
Foot_Locker.rename(columns={'Open': 'Foot_Locker_Open'},inplace=True)
Foot_Locker.rename(columns={'Close': 'Foot_Locker_Close'},inplace=True)


Ford_Motor = Ford_Motor [['Date','Open','Close']]
Ford_Motor.rename(columns={'Open': 'Ford_Motor_Open'},inplace=True)
Ford_Motor.rename(columns={'Close': 'Ford_Motor_Close'},inplace=True)


# In[12]:


Gap = quandl.get("WIKI/GPS",start_date="2006-10-20", end_date="2013-10-20") #Consumer Discretionary
Garmin = quandl.get("WIKI/GRMN",start_date="2006-10-20", end_date="2013-10-20")
General_Motors = quandl.get("WIKI/GM",start_date="2006-10-20", end_date="2013-10-20")
Genuine_Parts = quandl.get("WIKI/GPC",start_date="2006-10-20", end_date="2013-10-20")
Goodyear_Tire_Rubber = quandl.get("WIKI/GT",start_date="2006-10-20", end_date="2013-10-20")
Hanesbrands = quandl.get("WIKI/HBI",start_date="2006-10-20", end_date="2013-10-20")
Harley_Davidson = quandl.get("WIKI/HOG",start_date="2006-10-20", end_date="2013-10-20")
Hasbro = quandl.get("WIKI/HAS",start_date="2006-10-20", end_date="2013-10-20")
Hilton_Worldwide_Holdings = quandl.get("WIKI/HLT",start_date="2006-10-20", end_date="2013-10-20")
Home_Depot = quandl.get("WIKI/HD",start_date="2006-10-20", end_date="2013-10-20")





#indexing
Gap.reset_index(inplace=True) #Consumer Discretionary
Garmin.reset_index(inplace=True)
General_Motors.reset_index(inplace=True)
Genuine_Parts.reset_index(inplace=True)
Goodyear_Tire_Rubber.reset_index(inplace=True)
Hanesbrands.reset_index(inplace=True)
Harley_Davidson.reset_index(inplace=True)
Hasbro.reset_index(inplace=True)
Hilton_Worldwide_Holdings.reset_index(inplace=True)
Home_Depot.reset_index(inplace=True)

#Removing/Renaming columns
Gap = Gap [['Date','Open','Close']]
Gap.rename(columns={'Open': 'Gap_Open'},inplace=True)
Gap.rename(columns={'Close': 'Gap_Close'},inplace=True)

Garmin = Garmin [['Date','Open','Close']]
Garmin.rename(columns={'Open': 'Garmin_Open'},inplace=True)
Garmin.rename(columns={'Close': 'Garmin_Close'},inplace=True)

General_Motors = General_Motors [['Date','Open','Close']]
General_Motors.rename(columns={'Open': 'General_Motors_Open'},inplace=True)
General_Motors.rename(columns={'Close': 'General_Motors_Close'},inplace=True)

Genuine_Parts = Genuine_Parts [['Date','Open','Close']]
Genuine_Parts.rename(columns={'Open': 'Genuine_Parts_Open'},inplace=True)
Genuine_Parts.rename(columns={'Close': 'Genuine_Parts_Close'},inplace=True)

Goodyear_Tire_Rubber = Goodyear_Tire_Rubber [['Date','Open','Close']]
Goodyear_Tire_Rubber.rename(columns={'Open': 'Goodyear_Tire_Rubber_Open'},inplace=True)
Goodyear_Tire_Rubber.rename(columns={'Close': 'Goodyear_Tire_Rubber_Close'},inplace=True)

Hanesbrands = Hanesbrands [['Date','Open','Close']]
Hanesbrands.rename(columns={'Open': 'Hanesbrands_Open'},inplace=True)
Hanesbrands.rename(columns={'Close': 'Hanesbrands_Close'},inplace=True)

Harley_Davidson = Harley_Davidson [['Date','Open','Close']]
Harley_Davidson.rename(columns={'Open': 'Harley_Davidson_Open'},inplace=True)
Harley_Davidson.rename(columns={'Close': 'Harley_Davidson_Close'},inplace=True)

Hasbro = Hasbro [['Date','Open','Close']]
Hasbro.rename(columns={'Open': 'Hasbro_Open'},inplace=True)
Hasbro.rename(columns={'Close': 'Hasbro_Close'},inplace=True)

Hilton_Worldwide_Holdings = Hilton_Worldwide_Holdings [['Date','Open','Close']]
Hilton_Worldwide_Holdings.rename(columns={'Open': 'Hilton_Worldwide_Holdings_Open'},inplace=True)
Hilton_Worldwide_Holdings.rename(columns={'Close': 'Hilton_Worldwide_Holdings_Close'},inplace=True)

Home_Depot = Home_Depot [['Date','Open','Close']]
Home_Depot.rename(columns={'Open': 'Home_Depot_Open'},inplace=True)
Home_Depot.rename(columns={'Close': 'Home_Depot_Close'},inplace=True)


# In[13]:


Kohls = quandl.get("WIKI/KSS",start_date="2006-10-20", end_date="2013-10-20") #Consumer Discretionary
L_Brands = quandl.get("WIKI/LB",start_date="2006-10-20", end_date="2013-10-20")
Leggett_Platt  = quandl.get("WIKI/LEG",start_date="2006-10-20", end_date="2013-10-20")
Lennar = quandl.get("WIKI/LEN",start_date="2006-10-20", end_date="2013-10-20")
LKQ = quandl.get("WIKI/LKQ",start_date="2006-10-20", end_date="2013-10-20")
Lowes_Cos  = quandl.get("WIKI/LOW",start_date="2006-10-20", end_date="2013-10-20")
Macys = quandl.get("WIKI/M",start_date="2006-10-20", end_date="2013-10-20")
Marriott_Int = quandl.get("WIKI/MAR",start_date="2006-10-20", end_date="2013-10-20")
Mattel  = quandl.get("WIKI/MAT",start_date="2006-10-20", end_date="2013-10-20")
McDonalds  = quandl.get("WIKI/MCD",start_date="2006-10-20", end_date="2013-10-20")




#indexing
Kohls.reset_index(inplace=True)
L_Brands.reset_index(inplace=True)
Leggett_Platt.reset_index(inplace=True)
Lennar.reset_index(inplace=True)
LKQ.reset_index(inplace=True)
Lowes_Cos.reset_index(inplace=True)
Macys.reset_index(inplace=True)
Marriott_Int.reset_index(inplace=True)
Mattel.reset_index(inplace=True)
McDonalds.reset_index(inplace=True)

#Removing/Renaming columns
Kohls = Kohls [['Date','Open','Close']]
Kohls.rename(columns={'Open': 'Kohls_Open'},inplace=True)
Kohls.rename(columns={'Close': 'Kohls_Close'},inplace=True)

L_Brands = L_Brands [['Date','Open','Close']]
L_Brands.rename(columns={'Open': 'L_Brands_Open'},inplace=True)
L_Brands.rename(columns={'Close': 'L_Brands_Close'},inplace=True)

Leggett_Platt = Leggett_Platt [['Date','Open','Close']]
Leggett_Platt.rename(columns={'Open': 'Leggett_Platt_Open'},inplace=True)
Leggett_Platt.rename(columns={'Close': 'Leggett_Platt_Close'},inplace=True)

Lennar = Lennar [['Date','Open','Close']]
Lennar.rename(columns={'Open': 'Lennar_Open'},inplace=True)
Lennar.rename(columns={'Close': 'Lennar_Close'},inplace=True)

LKQ = LKQ [['Date','Open','Close']]
LKQ.rename(columns={'Open': 'LKQ_Open'},inplace=True)
LKQ.rename(columns={'Close': 'LKQ_Close'},inplace=True)

Lowes_Cos = Lowes_Cos [['Date','Open','Close']]
Lowes_Cos.rename(columns={'Open': 'Lowes_Cos_Open'},inplace=True)
Lowes_Cos.rename(columns={'Close': 'Lowes_Cos_Close'},inplace=True)

Macys = Macys [['Date','Open','Close']]
Macys.rename(columns={'Open': 'Macys_Open'},inplace=True)
Macys.rename(columns={'Close': 'Macys_Close'},inplace=True)

Marriott_Int = Marriott_Int [['Date','Open','Close']]
Marriott_Int.rename(columns={'Open': 'Marriott_Int_Open'},inplace=True)
Marriott_Int.rename(columns={'Close': 'Marriott_Int_Close'},inplace=True)

Mattel = Mattel [['Date','Open','Close']]
Mattel.rename(columns={'Open': 'Mattel_Open'},inplace=True)
Mattel.rename(columns={'Close': 'Mattel_Close'},inplace=True)


McDonalds = McDonalds [['Date','Open','Close']]
McDonalds.rename(columns={'Open': 'McDonalds_Open'},inplace=True)
McDonalds.rename(columns={'Close': 'McDonalds_Close'},inplace=True)


# In[14]:


MGM_Resorts_International  = quandl.get("WIKI/MGM",start_date="2006-10-20", end_date="2013-10-20") #Consumer Discretionary
Mohawk_Industries  = quandl.get("WIKI/MHK",start_date="2006-10-20", end_date="2013-10-20")
Newell_Brands  = quandl.get("WIKI/NWL",start_date="2006-10-20", end_date="2013-10-20")
Nike = quandl.get("WIKI/NKE",start_date="2006-10-20", end_date="2013-10-20")
Nordstrom = quandl.get("WIKI/JWN",start_date="2006-10-20", end_date="2013-10-20")
Norwegian_Cruise_Line = quandl.get("WIKI/NCLH",start_date="2006-10-20", end_date="2013-10-20")
OReilly_Automotive = quandl.get("WIKI/ORLY",start_date="2006-10-20", end_date="2013-10-20")
Polo_Ralph_Lauren = quandl.get("WIKI/RL",start_date="2006-10-20", end_date="2013-10-20")
Pulte_Homes  = quandl.get("WIKI/PHM",start_date="2006-10-20", end_date="2013-10-20")
PVH_Corp  = quandl.get("WIKI/PVH",start_date="2006-10-20", end_date="2013-10-20")


#indexing
MGM_Resorts_International.reset_index(inplace=True)
Mohawk_Industries.reset_index(inplace=True)
Newell_Brands.reset_index(inplace=True)
Nike.reset_index(inplace=True)
Nordstrom.reset_index(inplace=True)
Norwegian_Cruise_Line.reset_index(inplace=True)
OReilly_Automotive.reset_index(inplace=True)
Polo_Ralph_Lauren.reset_index(inplace=True)
Pulte_Homes.reset_index(inplace=True)
PVH_Corp.reset_index(inplace=True)


#Removing/Renaming columns
MGM_Resorts_International = MGM_Resorts_International [['Date','Open','Close']]
MGM_Resorts_International.rename(columns={'Open': 'MGM_Resorts_International_Open'},inplace=True)
MGM_Resorts_International.rename(columns={'Close': 'MGM_Resorts_International_Close'},inplace=True)

Mohawk_Industries.reset_index(inplace=True)
Mohawk_Industries = Mohawk_Industries [['Date','Open','Close']]
Mohawk_Industries.rename(columns={'Open': 'Mohawk_Industries_Open'},inplace=True)
Mohawk_Industries.rename(columns={'Close': 'Mohawk_Industries_Close'},inplace=True)


Newell_Brands.reset_index(inplace=True)
Newell_Brands = Newell_Brands [['Date','Open','Close']]
Newell_Brands.rename(columns={'Open': 'Newell_Brands_Open'},inplace=True)
Newell_Brands.rename(columns={'Close': 'Newell_Brands_Close'},inplace=True)


Nike.reset_index(inplace=True)
Nike = Nike [['Date','Open','Close']]
Nike.rename(columns={'Open': 'Nike_Open'},inplace=True)
Nike.rename(columns={'Close': 'Nike_Close'},inplace=True)


Nordstrom.reset_index(inplace=True)
Nordstrom = Nordstrom [['Date','Open','Close']]
Nordstrom.rename(columns={'Open': 'Nordstrom_Open'},inplace=True)
Nordstrom.rename(columns={'Close': 'Nordstrom_Close'},inplace=True)


Norwegian_Cruise_Line.reset_index(inplace=True)
Norwegian_Cruise_Line = Norwegian_Cruise_Line [['Date','Open','Close']]
Norwegian_Cruise_Line.rename(columns={'Open': 'Norwegian_Cruise_Line_Open'},inplace=True)
Norwegian_Cruise_Line.rename(columns={'Close': 'Norwegian_Cruise_Line_Close'},inplace=True)


OReilly_Automotive.reset_index(inplace=True)
OReilly_Automotive = OReilly_Automotive [['Date','Open','Close']]
OReilly_Automotive.rename(columns={'Open': 'OReilly_Automotive_Open'},inplace=True)
OReilly_Automotive.rename(columns={'Close': 'OReilly_Automotive_Close'},inplace=True)

Polo_Ralph_Lauren.reset_index(inplace=True)
Polo_Ralph_Lauren = Polo_Ralph_Lauren [['Date','Open','Close']]
Polo_Ralph_Lauren.rename(columns={'Open': 'Polo_Ralph_Lauren_Open'},inplace=True)
Polo_Ralph_Lauren.rename(columns={'Close': 'Polo_Ralph_Lauren_Close'},inplace=True)


Pulte_Homes.reset_index(inplace=True)
Pulte_Homes = Pulte_Homes [['Date','Open','Close']]
Pulte_Homes.rename(columns={'Open': 'Pulte_Homes_Open'},inplace=True)
Pulte_Homes.rename(columns={'Close': 'Pulte_Homes_Close'},inplace=True)


PVH_Corp.reset_index(inplace=True)
PVH_Corp = PVH_Corp [['Date','Open','Close']]
PVH_Corp.rename(columns={'Open': 'PVH_Corp_Open'},inplace=True)
PVH_Corp.rename(columns={'Close': 'PVH_Corp_Close'},inplace=True)



# In[15]:


Ross_Stores  = quandl.get("WIKI/ROST",start_date="2006-10-20", end_date="2013-10-20")  #Consumer Discretionary
Royal_Caribbean_Cruises = quandl.get("WIKI/RCL",start_date="2006-10-20", end_date="2013-10-20")
Starbucks = quandl.get("WIKI/SBUX",start_date="2006-10-20", end_date="2013-10-20")
Tapestry  = quandl.get("WIKI/TPR",start_date="2006-10-20", end_date="2013-10-20")
Target = quandl.get("WIKI/TGT",start_date="2006-10-20", end_date="2013-10-20")
Tiffany = quandl.get("WIKI/TIF",start_date="2006-10-20", end_date="2013-10-20")
TJX_Companies = quandl.get("WIKI/TJX",start_date="2006-10-20", end_date="2013-10-20")
Tractor_Supply_Company = quandl.get("WIKI/TSCO",start_date="2006-10-20", end_date="2013-10-20")
Ulta_Beauty = quandl.get("WIKI/ULTA",start_date="2006-10-20", end_date="2013-10-20")
Under_Armour_Class_C = quandl.get("WIKI/UA",start_date="2006-10-20", end_date="2013-10-20")




#indexing
Ross_Stores.reset_index(inplace=True)
Royal_Caribbean_Cruises.reset_index(inplace=True)
Starbucks.reset_index(inplace=True)
Tapestry.reset_index(inplace=True)
Target.reset_index(inplace=True)
Tiffany.reset_index(inplace=True)
TJX_Companies.reset_index(inplace=True)
Tractor_Supply_Company.reset_index(inplace=True)
Ulta_Beauty.reset_index(inplace=True)
Under_Armour_Class_C.reset_index(inplace=True)

#Removing/Renaming columns
Ross_Stores = Ross_Stores [['Date','Open','Close']]
Ross_Stores.rename(columns={'Open': 'Ross_Stores_Open'},inplace=True)
Ross_Stores.rename(columns={'Close': 'Ross_Stores_Close'},inplace=True)

Royal_Caribbean_Cruises = Royal_Caribbean_Cruises [['Date','Open','Close']]
Royal_Caribbean_Cruises.rename(columns={'Open': 'Royal_Caribbean_Cruises_Open'},inplace=True)
Royal_Caribbean_Cruises.rename(columns={'Close': 'Royal_Caribbean_Cruises_Close'},inplace=True)

Starbucks = Starbucks [['Date','Open','Close']]
Starbucks.rename(columns={'Open': 'Starbucks_Open'},inplace=True)
Starbucks.rename(columns={'Close': 'Starbucks_Close'},inplace=True)

Tapestry = Tapestry [['Date','Open','Close']]
Tapestry.rename(columns={'Open': 'Tapestry_Open'},inplace=True)
Tapestry.rename(columns={'Close': 'Tapestry_Close'},inplace=True)

Target = Target [['Date','Open','Close']]
Target.rename(columns={'Open': 'Target_Open'},inplace=True)
Target.rename(columns={'Close': 'Target_Close'},inplace=True)

Tiffany = Tiffany [['Date','Open','Close']]
Tiffany.rename(columns={'Open': 'Tiffany_Open'},inplace=True)
Tiffany.rename(columns={'Close': 'Tiffany_Close'},inplace=True)

TJX_Companies = TJX_Companies [['Date','Open','Close']]
TJX_Companies.rename(columns={'Open': 'TJX_Companies_Open'},inplace=True)
TJX_Companies.rename(columns={'Close': 'TJX_Companies_Close'},inplace=True)

Tractor_Supply_Company = Tractor_Supply_Company [['Date','Open','Close']]
Tractor_Supply_Company.rename(columns={'Open': 'Tractor_Supply_Company_Open'},inplace=True)
Tractor_Supply_Company.rename(columns={'Close': 'Tractor_Supply_Company_Close'},inplace=True)

Ulta_Beauty = Ulta_Beauty [['Date','Open','Close']]
Ulta_Beauty.rename(columns={'Open': 'Ulta_Beauty_Open'},inplace=True)
Ulta_Beauty.rename(columns={'Close': 'Ulta_Beauty_Close'},inplace=True)

Under_Armour_Class_C = Under_Armour_Class_C [['Date','Open','Close']]
Under_Armour_Class_C.rename(columns={'Open': 'Under_Armour_Class_C_Open'},inplace=True)
Under_Armour_Class_C.rename(columns={'Close': 'Under_Armour_Class_C_Close'},inplace=True)


# In[16]:


Under_Armour_Class_A  = quandl.get("WIKI/UAA",start_date="2006-10-20", end_date="2013-10-20") #Consumer Discretionary
V_F_Corp = quandl.get("WIKI/VFC",start_date="2006-10-20", end_date="2013-10-20")
Whirlpool = quandl.get("WIKI/WHR",start_date="2006-10-20", end_date="2013-10-20")
Wynn_Resorts = quandl.get("WIKI/WYNN",start_date="2006-10-20", end_date="2013-10-20")
Yum_Brands  = quandl.get("WIKI/YUM",start_date="2006-10-20", end_date="2013-10-20")


#indexing
Under_Armour_Class_A.reset_index(inplace=True)
V_F_Corp.reset_index(inplace=True)
Whirlpool.reset_index(inplace=True)
Wynn_Resorts.reset_index(inplace=True)
Yum_Brands.reset_index(inplace=True)

#Removing/Renaming columns
Under_Armour_Class_A = Under_Armour_Class_A [['Date','Open','Close']]
Under_Armour_Class_A.rename(columns={'Open': 'Under_Armour_Class_A_Open'},inplace=True)
Under_Armour_Class_A.rename(columns={'Close': 'Under_Armour_Class_A_Close'},inplace=True)


V_F_Corp = V_F_Corp [['Date','Open','Close']]
V_F_Corp.rename(columns={'Open': 'V_F_Corp_Open'},inplace=True)
V_F_Corp.rename(columns={'Close': 'V_F_Corp_Close'},inplace=True)

Whirlpool = Whirlpool [['Date','Open','Close']]
Whirlpool.rename(columns={'Open': 'Whirlpool_Open'},inplace=True)
Whirlpool.rename(columns={'Close': 'Whirlpool_Close'},inplace=True)

Wynn_Resorts = Wynn_Resorts [['Date','Open','Close']]
Wynn_Resorts.rename(columns={'Open': 'Wynn_Resorts_Open'},inplace=True)
Wynn_Resorts.rename(columns={'Close': 'Wynn_Resorts_Close'},inplace=True)


Yum_Brands = Yum_Brands [['Date','Open','Close']]
Yum_Brands.rename(columns={'Open': 'Yum_Brands_Open'},inplace=True)
Yum_Brands.rename(columns={'Close': 'Yum_Brands_Close'},inplace=True)



# In[14]:


Advance_Auto_Parts.shape
Amazon.shape
Aptiv.shape
AutoZone.shape
Best_Buy_Co.shape
BlockHR.shape
Booking_Holdings.shape
BorgWarner.shape
Carmax.shape
Carnival.shape
Chipotle_Mexican_Grill.shape
D_R_Horton.shape
Darden_Restaurants.shape
Dollar_General.shape
Dollar_Tree.shape
eBay.shape
Expedia_Group.shape
Foot_Locker.shape
Ford_Motor.shape
Gap.shape
Garmin.shape
General_Motors.shape
Genuine_Parts.shape
Goodyear_Tire_Rubber.shape
Hanesbrands.shape
Harley_Davidson.shape
Hasbro.shape
Hilton_Worldwide_Holdings.shape
Home_Depot.shape
Kohls.shape
L_Brands.shape
Leggett_Platt.shape
Lennar.shape
LKQ.shape
Lowes_Cos.shape
Macys.shape
Marriott_Int.shape
Mattel.shape
McDonalds.shape
MGM_Resorts_International.shape
Mohawk_Industries.shape
Newell_Brands.shape
Nike.shape
Nordstrom.shape
Norwegian_Cruise_Line.shape
OReilly_Automotive.shape
Polo_Ralph_Lauren.shape
Pulte_Homes.shape
PVH_Corp.shape
Ross_Stores.shape
Royal_Caribbean_Cruises.shape
Starbucks.shape
Tapestry.shape
#Target.reset_index(inplace=True)
#Tiffany.reset_index(inplace=True)
#TJX_Companies.reset_index(inplace=True)
#Tractor_Supply_Company.reset_index(inplace=True)
#Ulta_Beauty.reset_index(inplace=True)
#Under_Armour_Class_C.reset_index(inplace=True)



# In[17]:


#Consumer_Discretionary = pd.concat([Advance_Auto_Parts[['Date','Advance_Auto_Parts_Open','Advance_Auto_Parts_Close']], Amazon[['Amazon_Open','Amazon_Close']], Aptiv[['Aptiv_Open', 'Aptiv_Close']], AutoZone[['AutoZone_Open','AutoZone_Close']],Best_Buy_Co[['Best_Buy_Co_Open','Best_Buy_Co_Close']],BlockHR[['BlockHR_Open','BlockHR_Close']],Booking_Holdings[['Booking_Holdings_Open','Booking_Holdings_Close']],BorgWarner[['BorgWarner_Open','BorgWarner_Close']],Carmax[['Carmax_Open','Carmax_Close']],Carnival[['Carnival_Open','Carnival_Close']],Chipotle_Mexican_Grill[['Chipotle_Mexican_Grill_Open','Chipotle_Mexican_Grill_Close']],D_R_Horton[['D_R_Horton_Open','D_R_Horton_Close']],Darden_Restaurants[['Darden_Restaurants_Open','Darden_Restaurants_Close']],Dollar_General[['Dollar_General_Open','Dollar_General_Close']],Dollar_Tree[['Dollar_Tree_Open','Dollar_Tree_Close']],eBay[['eBay_Open','eBay_Close']], Expedia_Group[['Expedia_Group_Open','Expedia_Group_Close']],Foot_Locker[['Foot_Locker_Open','Foot_Locker_Close']],Ford_Motor[['Ford_Motor_Open','Ford_Motor_Close']],Gap[['Gap_Open','Gap_Close']],Garmin[['Garmin_Open','Garmin_Close']],General_Motors[['General_Motors_Open','General_Motors_Close']], Genuine_Parts[['Genuine_Parts_Open','Genuine_Parts_Close']],Goodyear_Tire_Rubber[['Goodyear_Tire_Rubber_Open','Goodyear_Tire_Rubber_Close']],Hanesbrands[['Hanesbrands_Open','Hanesbrands_Close']],Harley_Davidson[['Harley_Davidson_Open','Harley_Davidson_Close']],Hasbro[['Hasbro_Open','Hasbro_Close']],Hilton_Worldwide_Holdings[['Hilton_Worldwide_Holdings_Open','Hilton_Worldwide_Holdings_Close']], Home_Depot[['Home_Depot_Open','Home_Depot_Close']],Kohls[['Kohls_Open','Kohls_Close']],L_Brands[['L_Brands_Open','L_Brands_Close']],Leggett_Platt[['Leggett_Platt_Open','Leggett_Platt_Close']],Lennar[['Lennar_Open','Lennar_Close']],LKQ[['LKQ_Open','LKQ_Close']], Lowes_Cos[['Lowes_Cos_Open','Lowes_Cos_Close']],Macys[['Macys_Open','Macys_Close']],Marriott_Int[['Marriott_Int_Open','Marriott_Int_Close']],Mattel[['Mattel_Open','Mattel_Close']],McDonalds[['McDonalds_Open','McDonalds_Close']],MGM_Resorts_International[['MGM_Resorts_International_Open','MGM_Resorts_International_Close']], Mohawk_Industries[['Mohawk_Industries_Open','Mohawk_Industries_Close']],Newell_Brands[['Newell_Brands_Open','Newell_Brands_Close']],Nike[['Nike_Open','Nike_Close']],Nordstrom[['Nordstrom_Open','Nordstrom_Close']],Norwegian_Cruise_Line[['Norwegian_Cruise_Line_Open','Norwegian_Cruise_Line_Close']],OReilly_Automotive[['OReilly_Automotive_Open','OReilly_Automotive_Close']], Polo_Ralph_Lauren[['Polo_Ralph_Lauren_Open','Polo_Ralph_Lauren_Close']],Pulte_Homes[['Pulte_Homes_Open','Pulte_Homes_Close']],PVH_Corp[['PVH_Corp_Open','PVH_Corp_Close']],Ross_Stores[['Ross_Stores_Open','Ross_Stores_Close']],Royal_Caribbean_Cruises[['Royal_Caribbean_Cruises_Open','Royal_Caribbean_Cruises_Close']],Starbucks[['Starbucks_Open','Starbucks_Close']], Tapestry[['Tapestry_Open','Tapestry_Close']],Target[['Target_Open','Target_Close']],Tiffany[['Tiffany_Open','Tiffany_Close']],TJX_Companies[['TJX_Companies_Open','TJX_Companies_Close']],Tractor_Supply_Company[['Tractor_Supply_Company_Open','Tractor_Supply_Company_Close']],Ulta_Beauty[['Ulta_Beauty_Open','Ulta_Beauty_Close']], Under_Armour_Class_C[['Under_Armour_Class_C_Open','Under_Armour_Class_C_Close']],Under_Armour_Class_A[['Under_Armour_Class_A_Open','Under_Armour_Class_A_Close']],V_F_Corp[['V_F_Corp_Open','V_F_Corp_Open']],Whirlpool[['Whirlpool_Open','Whirlpool_Close']],Wynn_Resorts[['Wynn_Resorts_Open','Wynn_Resorts_Close']],Yum_Brands[['Yum_Brands_Open','Yum_Brands_Close']]],axis=1)


Consumer_Discretionary= (Advance_Auto_Parts).merge(Amazon).merge(AutoZone).merge(Best_Buy_Co).merge(BlockHR).merge(BorgWarner).merge(Carmax).merge(Carnival).merge(Chipotle_Mexican_Grill).merge(D_R_Horton).merge(Darden_Restaurants).merge(Dollar_General).merge(Dollar_Tree).merge(eBay).merge(Expedia_Group).merge(Foot_Locker).merge(Ford_Motor).merge(Gap).merge(Garmin).merge(General_Motors).merge(Genuine_Parts).merge(Goodyear_Tire_Rubber).merge(Hanesbrands).merge(Harley_Davidson).merge(Hasbro).merge(Home_Depot).merge(Kohls).merge(L_Brands).merge(Leggett_Platt).merge(Lennar).merge(LKQ).merge(Lowes_Cos).merge(Macys).merge(Marriott_Int).merge(Mattel).merge(McDonalds).merge(MGM_Resorts_International).merge(Mohawk_Industries).merge(Newell_Brands).merge(Nike).merge(Nordstrom).merge(Norwegian_Cruise_Line).merge(OReilly_Automotive).merge(Polo_Ralph_Lauren).merge(Pulte_Homes).merge(PVH_Corp).merge(Ross_Stores).merge(Royal_Caribbean_Cruises).merge(Starbucks).merge(Target).merge(Tiffany).merge(TJX_Companies).merge(Tractor_Supply_Company).merge(Ulta_Beauty).merge(Under_Armour_Class_A).merge(V_F_Corp).merge(Whirlpool).merge(Wynn_Resorts).merge(Yum_Brands)
#Aptiv
#Booking_Holdings
#Hilton_Worldwide_Holdings
#tepestry
#Under_Armour_Class_C

Consumer_Discretionary

#Consumer_Discretionary.shape
#Consumer_Discretionary.isna
#Consumer_Discretionary.isnull().sum()




# In[18]:


Altria_Group  = quandl.get("WIKI/MO",start_date="2006-10-20", end_date="2013-10-20") #Consumer Staples
Archer_Daniels_Midland_Co = quandl.get("WIKI/ADM",start_date="2006-10-20", end_date="2013-10-20")
Brown_Forman = quandl.get("WIKI/BF_B",start_date="2006-10-20", end_date="2013-10-20")
Campbell_Soup = quandl.get("WIKI/CPB",start_date="2006-10-20", end_date="2013-10-20")
Church_Dwight  = quandl.get("WIKI/CHD",start_date="2006-10-20", end_date="2013-10-20")
The_Clorox_Company = quandl.get("WIKI/CLX",start_date="2006-10-20", end_date="2013-10-20")
Coca_Cola_Company = quandl.get("WIKI/KO",start_date="2006-10-20", end_date="2013-10-20")
Colgate_Palmolive = quandl.get("WIKI/CL",start_date="2006-10-20", end_date="2013-10-20")
Conagra_Brands = quandl.get("WIKI/CAG",start_date="2006-10-20", end_date="2013-10-20")
Constellation_Brands  = quandl.get("WIKI/STZ",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Altria_Group.reset_index(inplace=True)
Archer_Daniels_Midland_Co.reset_index(inplace=True)
Brown_Forman.reset_index(inplace=True)
Campbell_Soup.reset_index(inplace=True)
Church_Dwight.reset_index(inplace=True)
The_Clorox_Company.reset_index(inplace=True)
Coca_Cola_Company.reset_index(inplace=True)
Colgate_Palmolive.reset_index(inplace=True)
Conagra_Brands.reset_index(inplace=True)
Constellation_Brands.reset_index(inplace=True)

#Removing/Renaming columns
Altria_Group = Altria_Group [['Date','Open','Close']]
Altria_Group.rename(columns={'Open': 'Altria_Group_Open'},inplace=True)
Altria_Group.rename(columns={'Close': 'Altria_Group_Close'},inplace=True)

Archer_Daniels_Midland_Co = Archer_Daniels_Midland_Co [['Date','Open','Close']]
Archer_Daniels_Midland_Co.rename(columns={'Open': 'Archer_Daniels_Midland_Co_Open'},inplace=True)
Archer_Daniels_Midland_Co.rename(columns={'Close': 'Archer_Daniels_Midland_Co_Close'},inplace=True)

Brown_Forman = Brown_Forman [['Date','Open','Close']]
Brown_Forman.rename(columns={'Open': 'Brown_Forman_Open'},inplace=True)
Brown_Forman.rename(columns={'Close': 'Brown_Forman_Close'},inplace=True)

Campbell_Soup = Campbell_Soup [['Date','Open','Close']]
Campbell_Soup.rename(columns={'Open': 'Campbell_Soup_Open'},inplace=True)
Campbell_Soup.rename(columns={'Close': 'Campbell_Soup_Close'},inplace=True)

Church_Dwight = Church_Dwight [['Date','Open','Close']]
Church_Dwight.rename(columns={'Open': 'Church_Dwight_Open'},inplace=True)
Church_Dwight.rename(columns={'Close': 'Church_Dwight_Close'},inplace=True)

The_Clorox_Company = The_Clorox_Company [['Date','Open','Close']]
The_Clorox_Company.rename(columns={'Open': 'The_Clorox_Company_Open'},inplace=True)
The_Clorox_Company.rename(columns={'Close': 'The_Clorox_Company_Close'},inplace=True)

Coca_Cola_Company = Coca_Cola_Company [['Date','Open','Close']]
Coca_Cola_Company.rename(columns={'Open': 'Coca_Cola_Company_Open'},inplace=True)
Coca_Cola_Company.rename(columns={'Close': 'Coca_Cola_Company_Close'},inplace=True)

Colgate_Palmolive = Colgate_Palmolive [['Date','Open','Close']]
Colgate_Palmolive.rename(columns={'Open': 'Colgate_Palmolive_Open'},inplace=True)
Colgate_Palmolive.rename(columns={'Close': 'Colgate_Palmolive_Close'},inplace=True)

Conagra_Brands = Conagra_Brands [['Date','Open','Close']]
Conagra_Brands.rename(columns={'Open': 'Conagra_Brands_Open'},inplace=True)
Conagra_Brands.rename(columns={'Close': 'Conagra_Brands_Close'},inplace=True)

Constellation_Brands = Constellation_Brands [['Date','Open','Close']]
Constellation_Brands.rename(columns={'Open': 'Constellation_Brands_Open'},inplace=True)
Constellation_Brands.rename(columns={'Close': 'Constellation_Brands_Close'},inplace=True)


# In[19]:


Costco_Wholesale = quandl.get("WIKI/COST",start_date="2006-10-20", end_date="2013-10-20") #Consumer Staples
Coty  = quandl.get("WIKI/COTY",start_date="2006-10-20", end_date="2013-10-20")
Estee_Lauder = quandl.get("WIKI/EL",start_date="2006-10-20", end_date="2013-10-20")
General_Mills = quandl.get("WIKI/GIS",start_date="2006-10-20", end_date="2013-10-20")
The_Hershey_Company = quandl.get("WIKI/HSY",start_date="2006-10-20", end_date="2013-10-20")
Hormel_Foods = quandl.get("WIKI/HRL",start_date="2006-10-20", end_date="2013-10-20")
JM_Smucker  = quandl.get("WIKI/SJM",start_date="2006-10-20", end_date="2013-10-20")
Kellogg = quandl.get("WIKI/K",start_date="2006-10-20", end_date="2013-10-20")
Kimberly_Clark = quandl.get("WIKI/KMB",start_date="2006-10-20", end_date="2013-10-20")
Kraft_Heinz= quandl.get("WIKI/KHC",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Costco_Wholesale.reset_index(inplace=True)
Coty.reset_index(inplace=True)
Estee_Lauder.reset_index(inplace=True)
General_Mills.reset_index(inplace=True)
The_Hershey_Company.reset_index(inplace=True)
Hormel_Foods.reset_index(inplace=True)
JM_Smucker.reset_index(inplace=True)
Kellogg.reset_index(inplace=True)
Kimberly_Clark.reset_index(inplace=True)
Kraft_Heinz.reset_index(inplace=True)

#Removing/Renaming columns
Costco_Wholesale = Costco_Wholesale [['Date','Open','Close']]
Costco_Wholesale.rename(columns={'Open': 'Costco_Wholesale_Open'},inplace=True)
Costco_Wholesale.rename(columns={'Close': 'Costco_Wholesale_Close'},inplace=True)

Coty = Coty [['Date','Open','Close']]
Coty.rename(columns={'Open': 'Coty_Open'},inplace=True)
Coty.rename(columns={'Close': 'Coty_Close'},inplace=True)

Estee_Lauder = Estee_Lauder [['Date','Open','Close']]
Estee_Lauder.rename(columns={'Open': 'Estee_Lauder_Open'},inplace=True)
Estee_Lauder.rename(columns={'Close': 'Estee_Lauder_Close'},inplace=True)

General_Mills = General_Mills [['Date','Open','Close']]
General_Mills.rename(columns={'Open': 'General_Mills_Open'},inplace=True)
General_Mills.rename(columns={'Close': 'General_Mills_Close'},inplace=True)

The_Hershey_Company = The_Hershey_Company [['Date','Open','Close']]
The_Hershey_Company.rename(columns={'Open': 'The_Hershey_Company_Open'},inplace=True)
The_Hershey_Company.rename(columns={'Close': 'The_Hershey_Company_Close'},inplace=True)

Hormel_Foods = Hormel_Foods [['Date','Open','Close']]
Hormel_Foods.rename(columns={'Open': 'Hormel_Foods_Open'},inplace=True)
Hormel_Foods.rename(columns={'Close': 'Hormel_Foods_Close'},inplace=True)

JM_Smucker = JM_Smucker [['Date','Open','Close']]
JM_Smucker.rename(columns={'Open': 'JM_Smucker_Open'},inplace=True)
JM_Smucker.rename(columns={'Close': 'JM_Smucker_Close'},inplace=True)

Kellogg = Kellogg [['Date','Open','Close']]
Kellogg.rename(columns={'Open': 'Kellogg_Open'},inplace=True)
Kellogg.rename(columns={'Close': 'Kellogg_Close'},inplace=True)

Kimberly_Clark = Kimberly_Clark [['Date','Open','Close']]
Kimberly_Clark.rename(columns={'Open': 'Kimberly_Clark_Open'},inplace=True)
Kimberly_Clark.rename(columns={'Close': 'Kimberly_Clark_Close'},inplace=True)

Kraft_Heinz = Kraft_Heinz [['Date','Open','Close']]
Kraft_Heinz.rename(columns={'Open': 'Kraft_Heinz_Open'},inplace=True)
Kraft_Heinz.rename(columns={'Close': 'Kraft_Heinz_Close'},inplace=True)


# In[20]:


Kroger = quandl.get("WIKI/KR",start_date="2006-10-20", end_date="2013-10-20")#Consumer Staples
#Lamb_Weston_Holdings = quandl.get("WIKI/LW",start_date="2006-10-20", end_date="2013-10-20")
McCormick   = quandl.get("WIKI/MKC",start_date="2006-10-20", end_date="2013-10-20")
Molson_Coors_Brewing_Company = quandl.get("WIKI/TAP",start_date="2006-10-20", end_date="2013-10-20")
Mondelez_International = quandl.get("WIKI/MDLZ",start_date="2006-10-20", end_date="2013-10-20")
Monster_Beverage = quandl.get("WIKI/MNST",start_date="2006-10-20", end_date="2013-10-20")
PepsiCo  = quandl.get("WIKI/PEP",start_date="2006-10-20", end_date="2013-10-20")
Philip_Morris_International  = quandl.get("WIKI/PM",start_date="2006-10-20", end_date="2013-10-20")
Procter_Gamble = quandl.get("WIKI/PG",start_date="2006-10-20", end_date="2013-10-20")
Sysco = quandl.get("WIKI/SYY",start_date="2006-10-20", end_date="2013-10-20")




#indexing
Kroger.reset_index(inplace=True)
#Lamb_Weston_Holdings.reset_index(inplace=True)
McCormick.reset_index(inplace=True)
Molson_Coors_Brewing_Company.reset_index(inplace=True)
Mondelez_International.reset_index(inplace=True)
Monster_Beverage.reset_index(inplace=True)
PepsiCo.reset_index(inplace=True)
Philip_Morris_International.reset_index(inplace=True)
Procter_Gamble.reset_index(inplace=True)
Sysco.reset_index(inplace=True)

#Removing/Renaming columns
Kroger = Kroger [['Date','Open','Close']]
Kroger.rename(columns={'Open': 'Kroger_Open'},inplace=True)
Kroger.rename(columns={'Close': 'Kroger_Close'},inplace=True)

#Lamb_Weston_Holdings = Lamb_Weston_Holdings [['Date','Open','Close']]
#Lamb_Weston_Holdings.rename(columns={'Open': 'Lamb_Weston_Holdings_Open'},inplace=True)
#Lamb_Weston_Holdings.rename(columns={'Close': 'Lamb_Weston_Holdings_Close'},inplace=True)

McCormick = McCormick [['Date','Open','Close']]
McCormick.rename(columns={'Open': 'McCormick_Open'},inplace=True)
McCormick.rename(columns={'Close': 'McCormick_Close'},inplace=True)

Molson_Coors_Brewing_Company = Molson_Coors_Brewing_Company [['Date','Open','Close']]
Molson_Coors_Brewing_Company.rename(columns={'Open': 'Molson_Coors_Brewing_Company_Open'},inplace=True)
Molson_Coors_Brewing_Company.rename(columns={'Close': 'Molson_Coors_Brewing_Company_Close'},inplace=True)

Mondelez_International = Mondelez_International [['Date','Open','Close']]
Mondelez_International.rename(columns={'Open': 'Mondelez_International_Open'},inplace=True)
Mondelez_International.rename(columns={'Close': 'Mondelez_International_Close'},inplace=True)

Monster_Beverage = Monster_Beverage [['Date','Open','Close']]
Monster_Beverage.rename(columns={'Open': 'Monster_Beverage_Open'},inplace=True)
Monster_Beverage.rename(columns={'Close': 'Monster_Beverage_Close'},inplace=True)

PepsiCo = PepsiCo [['Date','Open','Close']]
PepsiCo.rename(columns={'Open': 'PepsiCo_Open'},inplace=True)
PepsiCo.rename(columns={'Close': 'PepsiCo_Close'},inplace=True)

Philip_Morris_International = Philip_Morris_International [['Date','Open','Close']]
Philip_Morris_International.rename(columns={'Open': 'Philip_Morris_International_Open'},inplace=True)
Philip_Morris_International.rename(columns={'Close': 'Philip_Morris_International_Close'},inplace=True)

Procter_Gamble = Procter_Gamble [['Date','Open','Close']]
Procter_Gamble.rename(columns={'Open': 'Procter_Gamble_Open'},inplace=True)
Procter_Gamble.rename(columns={'Close': 'Procter_Gamble_Close'},inplace=True)

Sysco = Sysco [['Date','Open','Close']]
Sysco.rename(columns={'Open': 'Sysco_Open'},inplace=True)
Sysco.rename(columns={'Close': 'Sysco_Close'},inplace=True)


# In[21]:


Tyson_Foods = quandl.get("WIKI/TSN",start_date="2006-10-20", end_date="2013-10-20")#Consumer Staples
Walmart = quandl.get("WIKI/WMT",start_date="2006-10-20", end_date="2013-10-20")
Walgreens_Boots_Alliance = quandl.get("WIKI/WBA",start_date="2006-10-20", end_date="2013-10-20")




#indexing
Tyson_Foods.reset_index(inplace=True)
Walmart.reset_index(inplace=True)
Walgreens_Boots_Alliance.reset_index(inplace=True)

#Removing/Renaming columns
Tyson_Foods = Tyson_Foods [['Date','Open','Close']]
Tyson_Foods.rename(columns={'Open': 'Tyson_Foods_Open'},inplace=True)
Tyson_Foods.rename(columns={'Close': 'Tyson_Foods_Close'},inplace=True)

Walmart = Walmart [['Date','Open','Close']]
Walmart.rename(columns={'Open': 'Walmart_Open'},inplace=True)
Walmart.rename(columns={'Close': 'Walmart_Close'},inplace=True)

Walgreens_Boots_Alliance = Walgreens_Boots_Alliance [['Date','Open','Close']]
Walgreens_Boots_Alliance.rename(columns={'Open': 'Walgreens_Boots_Alliance_Open'},inplace=True)
Walgreens_Boots_Alliance.rename(columns={'Close': 'Walgreens_Boots_Alliance_Close'},inplace=True)


# In[22]:


#Consumer_Staples = pd.concat([Altria_Group[['Date','Altria_Group_Open','Altria_Group_Close']], Archer_Daniels_Midland_Co[['Archer_Daniels_Midland_Co_Open','Archer_Daniels_Midland_Co_Close']], Brown_Forman[['Brown_Forman_Open', 'Brown_Forman_Close']], Campbell_Soup[['Campbell_Soup_Open','Campbell_Soup_Close']],Church_Dwight[['Church_Dwight_Open','Church_Dwight_Close']],The_Clorox_Company[['The_Clorox_Company_Open','The_Clorox_Company_Close']],Coca_Cola_Company[['Coca_Cola_Company_Open','Coca_Cola_Company_Close']],Colgate_Palmolive[['Colgate_Palmolive_Open','Colgate_Palmolive_Close']],Conagra_Brands[['Conagra_Brands_Open','Conagra_Brands_Close']],Constellation_Brands[['Constellation_Brands_Open','Constellation_Brands_Close']],Costco_Wholesale[['Costco_Wholesale_Open','Costco_Wholesale_Close']],Coty[['Coty_Open','Coty_Close']],Estee_Lauder[['Estee_Lauder_Open','Estee_Lauder_Close']],General_Mills[['General_Mills_Open','General_Mills_Close']],The_Hershey_Company[['The_Hershey_Company_Open','The_Hershey_Company_Close']],Hormel_Foods[['Hormel_Foods_Open','Hormel_Foods_Close']], JM_Smucker[['JM_Smucker_Open','JM_Smucker_Close']],Kellogg[['Kellogg_Open','Kellogg_Close']],Kimberly_Clark[['Kimberly_Clark_Open','Kimberly_Clark_Close']],Kraft_Heinz[['Kraft_Heinz_Open','Kraft_Heinz_Close']],Kroger[['Kroger_Open','Kroger_Close']],McCormick[['McCormick_Open','McCormick_Close']], Molson_Coors_Brewing_Company[['Molson_Coors_Brewing_Company_Open','Molson_Coors_Brewing_Company_Close']],Mondelez_International[['Mondelez_International_Open','Mondelez_International_Close']],Monster_Beverage[['Monster_Beverage_Open','Monster_Beverage_Close']],PepsiCo[['PepsiCo_Open','PepsiCo_Close']],Philip_Morris_International[['Philip_Morris_International_Open','Philip_Morris_International_Close']],Procter_Gamble[['Procter_Gamble_Open','Procter_Gamble_Close']], Sysco[['Sysco_Open','Sysco_Close']],Tyson_Foods[['Tyson_Foods_Open','Tyson_Foods_Close']],Walmart[['Walmart_Open','Walmart_Close']],Walgreens_Boots_Alliance[['Walgreens_Boots_Alliance_Open','Walgreens_Boots_Alliance_Close']]],axis=1)
Consumer_Staples= (Altria_Group).merge(Archer_Daniels_Midland_Co).merge(Brown_Forman).merge(Campbell_Soup).merge(Church_Dwight).merge(The_Clorox_Company).merge(Coca_Cola_Company).merge(Colgate_Palmolive).merge(Conagra_Brands).merge(Constellation_Brands).merge(Costco_Wholesale).merge(Coty).merge(General_Mills).merge(The_Hershey_Company).merge(Hormel_Foods).merge(JM_Smucker).merge(Kellogg).merge(Kimberly_Clark).merge(Kroger).merge(McCormick).merge(Molson_Coors_Brewing_Company).merge(Mondelez_International).merge(Monster_Beverage).merge(PepsiCo).merge(Philip_Morris_International).merge(Procter_Gamble).merge(Sysco).merge(Tyson_Foods).merge(Walmart).merge(Walgreens_Boots_Alliance)

#Kraft_Heinz
Consumer_Staples

#Consumer_Staples.isnull().sum()


# In[25]:


Anadarko_Petroleum = quandl.get("WIKI/APC",start_date="2006-10-20", end_date="2013-10-20") #Energy
Apache_Corporation = quandl.get("WIKI/APA",start_date="2006-10-20", end_date="2013-10-20")
Baker_Hughes = quandl.get("WIKI/BHGE",start_date="2006-10-20", end_date="2013-10-20")
Cabot_Oil_Gas = quandl.get("WIKI/COG",start_date="2006-10-20", end_date="2013-10-20")
Chevron = quandl.get("WIKI/CVX",start_date="2006-10-20", end_date="2013-10-20")
Cimarex_Energy   = quandl.get("WIKI/XEC",start_date="2006-10-20", end_date="2013-10-20")
Concho_Resources = quandl.get("WIKI/CXO",start_date="2006-10-20", end_date="2013-10-20")
ConocoPhillips = quandl.get("WIKI/COP",start_date="2006-10-20", end_date="2013-10-20")
Devon_Energy = quandl.get("WIKI/DVN",start_date="2006-10-20", end_date="2013-10-20")
Diamondback_Energy = quandl.get("WIKI/FANG",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Anadarko_Petroleum.reset_index(inplace=True)
Apache_Corporation.reset_index(inplace=True)
Baker_Hughes.reset_index(inplace=True)
Cabot_Oil_Gas.reset_index(inplace=True)
Devon_Energy.reset_index(inplace=True)
Chevron.reset_index(inplace=True)
ConocoPhillips.reset_index(inplace=True)
Concho_Resources.reset_index(inplace=True)
Diamondback_Energy.reset_index(inplace=True)
Cimarex_Energy.reset_index(inplace=True)

#Removing/Renaming columns
Anadarko_Petroleum = Anadarko_Petroleum [['Date','Open','Close']]
Anadarko_Petroleum.rename(columns={'Open': 'Anadarko_Petroleum_Open'},inplace=True)
Anadarko_Petroleum.rename(columns={'Close': 'Anadarko_Petroleum_Close'},inplace=True)

Apache_Corporation = Apache_Corporation [['Date','Open','Close']]
Apache_Corporation.rename(columns={'Open': 'Apache_Corporation_Open'},inplace=True)
Apache_Corporation.rename(columns={'Close': 'Apache_Corporation_Close'},inplace=True)

Baker_Hughes = Baker_Hughes [['Date','Open','Close']]
Baker_Hughes.rename(columns={'Open': 'Baker_Hughes_Open'},inplace=True)
Baker_Hughes.rename(columns={'Close': 'Baker_Hughes_Close'},inplace=True)

Cabot_Oil_Gas = Cabot_Oil_Gas [['Date','Open','Close']]
Cabot_Oil_Gas.rename(columns={'Open': 'Cabot_Oil_Gas_Open'},inplace=True)
Cabot_Oil_Gas.rename(columns={'Close': 'Cabot_Oil_Gas_Close'},inplace=True)

Devon_Energy = Devon_Energy [['Date','Open','Close']]
Devon_Energy.rename(columns={'Open': 'Devon_Energy_Open'},inplace=True)
Devon_Energy.rename(columns={'Close': 'Devon_Energy_Close'},inplace=True)

Chevron = Chevron [['Date','Open','Close']]
Chevron.rename(columns={'Open': 'Chevron_Open'},inplace=True)
Chevron.rename(columns={'Close': 'Chevron_Close'},inplace=True)

ConocoPhillips = ConocoPhillips [['Date','Open','Close']]
ConocoPhillips.rename(columns={'Open': 'ConocoPhillips_Open'},inplace=True)
ConocoPhillips.rename(columns={'Close': 'ConocoPhillips_Close'},inplace=True)

Concho_Resources = Concho_Resources [['Date','Open','Close']]
Concho_Resources.rename(columns={'Open': 'Concho_Resources_Open'},inplace=True)
Concho_Resources.rename(columns={'Close': 'Concho_Resources_Close'},inplace=True)

Diamondback_Energy = Diamondback_Energy [['Date','Open','Close']]
Diamondback_Energy.rename(columns={'Open': 'Diamondback_Energy_Open'},inplace=True)
Diamondback_Energy.rename(columns={'Close': 'Diamondback_Energy_Close'},inplace=True)

Cimarex_Energy = Cimarex_Energy [['Date','Open','Close']]
Cimarex_Energy.rename(columns={'Open': 'Cimarex_Energy_Open'},inplace=True)
Cimarex_Energy.rename(columns={'Close': 'Cimarex_Energy_Close'},inplace=True)


# In[26]:


EOG_Resources = quandl.get("WIKI/EOG",start_date="2006-10-20", end_date="2013-10-20") #Energy
Exxon_Mobil = quandl.get("WIKI/XOM",start_date="2006-10-20", end_date="2013-10-20")
Halliburton = quandl.get("WIKI/HAL",start_date="2006-10-20", end_date="2013-10-20")
Helmerich_Payne = quandl.get("WIKI/HP",start_date="2006-10-20", end_date="2013-10-20")
Hess_Corporation = quandl.get("WIKI/HES",start_date="2006-10-20", end_date="2013-10-20")
Kinder_Morgan = quandl.get("WIKI/KMI",start_date="2006-10-20", end_date="2013-10-20")
Marathon_Oil= quandl.get("WIKI/MRO",start_date="2006-10-20", end_date="2013-10-20")
Marathon_Petroleum = quandl.get("WIKI/MPC",start_date="2006-10-20", end_date="2013-10-20")
National_Oilwell_Varco = quandl.get("WIKI/NOV",start_date="2006-10-20", end_date="2013-10-20")
Newfield_Exploration = quandl.get("WIKI/NFX",start_date="2006-10-20", end_date="2013-10-20")


#indexing
EOG_Resources.reset_index(inplace=True)
Exxon_Mobil.reset_index(inplace=True)
Halliburton.reset_index(inplace=True)
Hess_Corporation.reset_index(inplace=True)
Kinder_Morgan.reset_index(inplace=True)
Marathon_Oil.reset_index(inplace=True)
Marathon_Petroleum.reset_index(inplace=True)
National_Oilwell_Varco.reset_index(inplace=True)
Newfield_Exploration.reset_index(inplace=True)
Helmerich_Payne.reset_index(inplace=True)


#Removing/Renaming columns
EOG_Resources = EOG_Resources [['Date','Open','Close']]
EOG_Resources.rename(columns={'Open': 'EOG_Resources_Open'},inplace=True)
EOG_Resources.rename(columns={'Close': 'EOG_Resources_Close'},inplace=True)

Exxon_Mobil = Exxon_Mobil [['Date','Open','Close']]
Exxon_Mobil.rename(columns={'Open': 'Exxon_Mobil_Open'},inplace=True)
Exxon_Mobil.rename(columns={'Close': 'Exxon_Mobil_Close'},inplace=True)

Halliburton = Halliburton [['Date','Open','Close']]
Halliburton.rename(columns={'Open': 'Halliburton_Open'},inplace=True)
Halliburton.rename(columns={'Close': 'Halliburton_Close'},inplace=True)

Hess_Corporation = Hess_Corporation [['Date','Open','Close']]
Hess_Corporation.rename(columns={'Open': 'Hess_Corporation_Open'},inplace=True)
Hess_Corporation.rename(columns={'Close': 'Hess_Corporation_Close'},inplace=True)

Kinder_Morgan = Kinder_Morgan [['Date','Open','Close']]
Kinder_Morgan.rename(columns={'Open': 'Kinder_Morgan_Open'},inplace=True)
Kinder_Morgan.rename(columns={'Close': 'Kinder_Morgan_Close'},inplace=True)

Marathon_Oil = Marathon_Oil [['Date','Open','Close']]
Marathon_Oil.rename(columns={'Open': 'Marathon_Oil_Open'},inplace=True)
Marathon_Oil.rename(columns={'Close': 'Marathon_Oil_Close'},inplace=True)

Marathon_Petroleum = Marathon_Petroleum [['Date','Open','Close']]
Marathon_Petroleum.rename(columns={'Open': 'Marathon_Petroleum_Open'},inplace=True)
Marathon_Petroleum.rename(columns={'Close': 'Marathon_Petroleum_Close'},inplace=True)

National_Oilwell_Varco = National_Oilwell_Varco [['Date','Open','Close']]
National_Oilwell_Varco.rename(columns={'Open': 'National_Oilwell_Varco_Open'},inplace=True)
National_Oilwell_Varco.rename(columns={'Close': 'National_Oilwell_Varco_Close'},inplace=True)

Newfield_Exploration = Newfield_Exploration [['Date','Open','Close']]
Newfield_Exploration.rename(columns={'Open': 'Newfield_Exploration_Open'},inplace=True)
Newfield_Exploration.rename(columns={'Close': 'Newfield_Exploration_Close'},inplace=True)

Helmerich_Payne = Helmerich_Payne [['Date','Open','Close']]
Helmerich_Payne.rename(columns={'Open': 'Helmerich_Payne_Open'},inplace=True)
Helmerich_Payne.rename(columns={'Close': 'Helmerich_Payne_Close'},inplace=True)


# In[27]:


Noble_Energy = quandl.get("WIKI/NBL",start_date="2006-10-20", end_date="2013-10-20") #Energy
Occidental_Petroleum = quandl.get("WIKI/OXY",start_date="2006-10-20", end_date="2013-10-20")
ONEOK  = quandl.get("WIKI/OKE",start_date="2006-10-20", end_date="2013-10-20")
Phillips_SixtySix = quandl.get("WIKI/PSX",start_date="2006-10-20", end_date="2013-10-20")
Pioneer_Natural_Resources  = quandl.get("WIKI/PXD",start_date="2006-10-20", end_date="2013-10-20")
Schlumberger = quandl.get("WIKI/SLB",start_date="2006-10-20", end_date="2013-10-20")
TechnipFMC = quandl.get("WIKI/FTI",start_date="2006-10-20", end_date="2013-10-20")
Valero_Energy = quandl.get("WIKI/VLO",start_date="2006-10-20", end_date="2013-10-20")
Williams_Cos  = quandl.get("WIKI/WMB",start_date="2006-10-20", end_date="2013-10-20")

#indexing
Noble_Energy.reset_index(inplace=True)
Occidental_Petroleum.reset_index(inplace=True)
ONEOK.reset_index(inplace=True)
Phillips_SixtySix.reset_index(inplace=True)
Pioneer_Natural_Resources.reset_index(inplace=True)
Schlumberger.reset_index(inplace=True)
TechnipFMC.reset_index(inplace=True)
Valero_Energy.reset_index(inplace=True)
Williams_Cos.reset_index(inplace=True)


#Removing/Renaming columns
Noble_Energy = Noble_Energy [['Date','Open','Close']]
Noble_Energy.rename(columns={'Open': 'Noble_Energy_Open'},inplace=True)
Noble_Energy.rename(columns={'Close': 'Noble_Energy_Close'},inplace=True)

Occidental_Petroleum = Occidental_Petroleum [['Date','Open','Close']]
Occidental_Petroleum.rename(columns={'Open': 'Occidental_Petroleum_Open'},inplace=True)
Occidental_Petroleum.rename(columns={'Close': 'Occidental_Petroleum_Close'},inplace=True)

ONEOK = ONEOK [['Date','Open','Close']]
ONEOK.rename(columns={'Open': 'ONEOK_Open'},inplace=True)
ONEOK.rename(columns={'Close': 'ONEOK_Close'},inplace=True)

Phillips_SixtySix = Phillips_SixtySix [['Date','Open','Close']]
Phillips_SixtySix.rename(columns={'Open': 'Phillips_SixtySix_Open'},inplace=True)
Phillips_SixtySix.rename(columns={'Close': 'Phillips_SixtySix_Close'},inplace=True)

Pioneer_Natural_Resources = Pioneer_Natural_Resources [['Date','Open','Close']]
Pioneer_Natural_Resources.rename(columns={'Open': 'Pioneer_Natural_Resources_Open'},inplace=True)
Pioneer_Natural_Resources.rename(columns={'Close': 'Pioneer_Natural_Resources_Close'},inplace=True)

Schlumberger = Schlumberger [['Date','Open','Close']]
Schlumberger.rename(columns={'Open': 'Schlumberger_Open'},inplace=True)
Schlumberger.rename(columns={'Close': 'Schlumberger_Close'},inplace=True)

TechnipFMC = TechnipFMC [['Date','Open','Close']]
TechnipFMC.rename(columns={'Open': 'TechnipFMC_Open'},inplace=True)
TechnipFMC.rename(columns={'Close': 'TechnipFMC_Close'},inplace=True)

Valero_Energy = Valero_Energy [['Date','Open','Close']]
Valero_Energy.rename(columns={'Open': 'Valero_Energy_Open'},inplace=True)
Valero_Energy.rename(columns={'Close': 'Valero_Energy_Close'},inplace=True)

Williams_Cos = Williams_Cos [['Date','Open','Close']]
Williams_Cos.rename(columns={'Open': 'Williams_Cos_Open'},inplace=True)
Williams_Cos.rename(columns={'Close': 'Williams_Cos_Close'},inplace=True)


# In[28]:


#Energy = pd.concat([Anadarko_Petroleum[['Date','Anadarko_Petroleum_Open','Anadarko_Petroleum_Close']], Apache_Corporation[['Apache_Corporation_Open','Apache_Corporation_Close']], Baker_Hughes[['Baker_Hughes_Open', 'Baker_Hughes_Close']], Cabot_Oil_Gas[['Cabot_Oil_Gas_Open','Cabot_Oil_Gas_Close']],Chevron[['Chevron_Open','Chevron_Close']],Cimarex_Energy[['Cimarex_Energy_Open','Cimarex_Energy_Close']],Concho_Resources[['Concho_Resources_Open','Concho_Resources_Close']],ConocoPhillips[['ConocoPhillips_Open','ConocoPhillips_Close']],Devon_Energy[['Devon_Energy_Open','Devon_Energy_Close']],Diamondback_Energy[['Diamondback_Energy_Open','Diamondback_Energy_Close']],EOG_Resources[['EOG_Resources_Open','EOG_Resources_Close']],Exxon_Mobil[['Exxon_Mobil_Open','Exxon_Mobil_Close']],Halliburton[['Halliburton_Open','Halliburton_Close']],Helmerich_Payne[['Helmerich_Payne_Open','Helmerich_Payne_Close']],Hess_Corporation[['Hess_Corporation_Open','Hess_Corporation_Close']],Kinder_Morgan[['Kinder_Morgan_Open','Kinder_Morgan_Close']], Marathon_Oil[['Marathon_Oil_Open','Marathon_Oil_Close']],Marathon_Petroleum[['Marathon_Petroleum_Open','Marathon_Petroleum_Close']],National_Oilwell_Varco[['National_Oilwell_Varco_Open','National_Oilwell_Varco_Close']],Newfield_Exploration[['Newfield_Exploration_Open','Newfield_Exploration_Close']],Noble_Energy[['Noble_Energy_Open','Noble_Energy_Close']],Occidental_Petroleum[['Occidental_Petroleum_Open','Occidental_Petroleum_Close']], ONEOK[['ONEOK_Open','ONEOK_Close']],Phillips_SixtySix[['Phillips_SixtySix_Open','Phillips_SixtySix_Close']],Pioneer_Natural_Resources[['Pioneer_Natural_Resources_Open','Pioneer_Natural_Resources_Close']],Schlumberger[['Schlumberger_Open','Schlumberger_Close']],TechnipFMC[['TechnipFMC_Open','TechnipFMC_Close']],Valero_Energy[['Valero_Energy_Open','Valero_Energy_Close']], Williams_Cos[['Williams_Cos_Open','Williams_Cos_Close']]],axis=1)

Energy = (Anadarko_Petroleum).merge(Apache_Corporation).merge(Campbell_Soup).merge(Cabot_Oil_Gas).merge(Chevron).merge(Cimarex_Energy).merge(Concho_Resources).merge(ConocoPhillips).merge(Devon_Energy).merge(Diamondback_Energy).merge(EOG_Resources).merge(Exxon_Mobil).merge(Halliburton).merge(Helmerich_Payne).merge(Hess_Corporation).merge(Kinder_Morgan).merge(Marathon_Oil).merge(Marathon_Petroleum).merge(National_Oilwell_Varco).merge(Newfield_Exploration).merge(Noble_Energy).merge(Occidental_Petroleum).merge(ONEOK).merge(Phillips_SixtySix).merge(Schlumberger).merge(TechnipFMC).merge(Valero_Energy).merge(Williams_Cos)
#Baker_Hughes
Energy


# In[29]:


Affiliated_Managers_Group = quandl.get("WIKI/AMG",start_date="2006-10-20", end_date="2013-10-20") #Financials
AFLAC = quandl.get("WIKI/AFL",start_date="2006-10-20", end_date="2013-10-20")
Allstate_Corp = quandl.get("WIKI/ALL",start_date="2006-10-20", end_date="2013-10-20")
American_Express = quandl.get("WIKI/AXP",start_date="2006-10-20", end_date="2013-10-20")
American_International_Group = quandl.get("WIKI/AIG",start_date="2006-10-20", end_date="2013-10-20")
Ameriprise_Financial = quandl.get("WIKI/AMP",start_date="2006-10-20", end_date="2013-10-20")
Aon = quandl.get("WIKI/AON",start_date="2006-10-20", end_date="2013-10-20")
Arthur_J_Gallagher= quandl.get("WIKI/AJG",start_date="2006-10-20", end_date="2013-10-20")
Assurant = quandl.get("WIKI/AIZ",start_date="2006-10-20", end_date="2013-10-20")
Bank_of_America = quandl.get("WIKI/BAC",start_date="2006-10-20", end_date="2013-10-20")


#indexing
Affiliated_Managers_Group.reset_index(inplace=True)
AFLAC.reset_index(inplace=True)
Allstate_Corp.reset_index(inplace=True)
American_International_Group.reset_index(inplace=True)
American_Express.reset_index(inplace=True)
Ameriprise_Financial.reset_index(inplace=True)
Aon.reset_index(inplace=True)
Arthur_J_Gallagher.reset_index(inplace=True)
Assurant.reset_index(inplace=True)
Bank_of_America.reset_index(inplace=True)


#Removing/Renaming columns
Affiliated_Managers_Group = Affiliated_Managers_Group [['Date','Open','Close']]
Affiliated_Managers_Group.rename(columns={'Open': 'Affiliated_Managers_Group_Open'},inplace=True)
Affiliated_Managers_Group.rename(columns={'Close': 'Affiliated_Managers_Group_Close'},inplace=True)

AFLAC = AFLAC [['Date','Open','Close']]
AFLAC.rename(columns={'Open': 'AFLAC_Open'},inplace=True)
AFLAC.rename(columns={'Close': 'AFLAC_Close'},inplace=True)

Allstate_Corp = Allstate_Corp [['Date','Open','Close']]
Allstate_Corp.rename(columns={'Open': 'Allstate_Corp_Open'},inplace=True)
Allstate_Corp.rename(columns={'Close': 'Allstate_Corp_Close'},inplace=True)

American_International_Group = American_International_Group [['Date','Open','Close']]
American_International_Group.rename(columns={'Open': 'American_International_Group_Open'},inplace=True)
American_International_Group.rename(columns={'Close': 'American_International_Group_Close'},inplace=True)

American_Express = American_Express [['Date','Open','Close']]
American_Express.rename(columns={'Open': 'American_Express_Open'},inplace=True)
American_Express.rename(columns={'Close': 'American_Express_Close'},inplace=True)

Ameriprise_Financial = Ameriprise_Financial [['Date','Open','Close']]
Ameriprise_Financial.rename(columns={'Open': 'Ameriprise_Financial_Open'},inplace=True)
Ameriprise_Financial.rename(columns={'Close': 'Ameriprise_Financial_Close'},inplace=True)

Aon = Aon [['Date','Open','Close']]
Aon.rename(columns={'Open': 'Aon_Open'},inplace=True)
Aon.rename(columns={'Close': 'Aon_Close'},inplace=True)

Arthur_J_Gallagher = Arthur_J_Gallagher [['Date','Open','Close']]
Arthur_J_Gallagher.rename(columns={'Open': 'Arthur_J_Gallagher_Open'},inplace=True)
Arthur_J_Gallagher.rename(columns={'Close': 'Arthur_J_Gallagher_Close'},inplace=True)

Assurant = Assurant [['Date','Open','Close']]
Assurant.rename(columns={'Open': 'Assurant_Open'},inplace=True)
Assurant.rename(columns={'Close': 'Assurant_Close'},inplace=True)

Bank_of_America = Bank_of_America [['Date','Open','Close']]
Bank_of_America.rename(columns={'Open': 'Bank_of_America_Open'},inplace=True)
Bank_of_America.rename(columns={'Close': 'Bank_of_America_Close'},inplace=True)


# In[30]:


The_Bank_of_NewYork = quandl.get("WIKI/BK",start_date="2006-10-20", end_date="2013-10-20") #Financials
BBT = quandl.get("WIKI/BBT",start_date="2006-10-20", end_date="2013-10-20")
Berkshire_Hathaway = quandl.get("WIKI/BRK_B",start_date="2006-10-20", end_date="2013-10-20")
BlackRock = quandl.get("WIKI/BLK",start_date="2006-10-20", end_date="2013-10-20")
Brighthouse_Financial = quandl.get("WIKI/BHF",start_date="2006-10-20", end_date="2013-10-20")
Capital_One_Financial = quandl.get("WIKI/COF",start_date="2006-10-20", end_date="2013-10-20")
Cboe_Global_Markets = quandl.get("WIKI/CBOE",start_date="2006-10-20", end_date="2013-10-20")
Charles_Schwab_Corporation = quandl.get("WIKI/SCHW",start_date="2006-10-20", end_date="2013-10-20")
Chubb = quandl.get("WIKI/CB",start_date="2006-10-20", end_date="2013-10-20")
Cincinnati_Financial = quandl.get("WIKI/CINF",start_date="2006-10-20", end_date="2013-10-20")




#indexing
The_Bank_of_NewYork.reset_index(inplace=True)
BBT.reset_index(inplace=True)
Berkshire_Hathaway.reset_index(inplace=True)
BlackRock.reset_index(inplace=True)
Brighthouse_Financial.reset_index(inplace=True)
Capital_One_Financial.reset_index(inplace=True)
Cboe_Global_Markets.reset_index(inplace=True)
Charles_Schwab_Corporation.reset_index(inplace=True)
Chubb.reset_index(inplace=True)
Cincinnati_Financial.reset_index(inplace=True)

#Removing/Renaming columns
The_Bank_of_NewYork = The_Bank_of_NewYork [['Date','Open','Close']]
The_Bank_of_NewYork.rename(columns={'Open': 'The_Bank_of_NewYork_Open'},inplace=True)
The_Bank_of_NewYork.rename(columns={'Close': 'The_Bank_of_NewYork_Close'},inplace=True)

BBT = BBT [['Date','Open','Close']]
BBT.rename(columns={'Open': 'BBT_Open'},inplace=True)
BBT.rename(columns={'Close': 'BBT_Close'},inplace=True)

Berkshire_Hathaway = Berkshire_Hathaway [['Date','Open','Close']]
Berkshire_Hathaway.rename(columns={'Open': 'Berkshire_Hathaway_Open'},inplace=True)
Berkshire_Hathaway.rename(columns={'Close': 'Berkshire_Hathaway_Close'},inplace=True)


BlackRock = BlackRock [['Date','Open','Close']]
BlackRock.rename(columns={'Open': 'BlackRock_Open'},inplace=True)
BlackRock.rename(columns={'Close': 'BlackRock_Close'},inplace=True)


Brighthouse_Financial = Brighthouse_Financial [['Date','Open','Close']]
Brighthouse_Financial.rename(columns={'Open': 'Brighthouse_Financial_Open'},inplace=True)
Brighthouse_Financial.rename(columns={'Close': 'Brighthouse_Financial_Close'},inplace=True)


Capital_One_Financial = Capital_One_Financial [['Date','Open','Close']]
Capital_One_Financial.rename(columns={'Open': 'Capital_One_Financial_Open'},inplace=True)
Capital_One_Financial.rename(columns={'Close': 'Capital_One_Financial_Close'},inplace=True)

Cboe_Global_Markets = Cboe_Global_Markets [['Date','Open','Close']]
Cboe_Global_Markets.rename(columns={'Open': 'Cboe_Global_Markets_Open'},inplace=True)
Cboe_Global_Markets.rename(columns={'Close': 'Cboe_Global_Markets_Close'},inplace=True)

Charles_Schwab_Corporation = Charles_Schwab_Corporation [['Date','Open','Close']]
Charles_Schwab_Corporation.rename(columns={'Open': 'Charles_Schwab_Corporation_Open'},inplace=True)
Charles_Schwab_Corporation.rename(columns={'Close': 'Charles_Schwab_Corporation_Close'},inplace=True)


Chubb = Chubb [['Date','Open','Close']]
Chubb.rename(columns={'Open': 'Chubb_Open'},inplace=True)
Chubb.rename(columns={'Close': 'Chubb_Close'},inplace=True)


Cincinnati_Financial = Cincinnati_Financial [['Date','Open','Close']]
Cincinnati_Financial.rename(columns={'Open': 'Cincinnati_Financial_Open'},inplace=True)
Cincinnati_Financial.rename(columns={'Close': 'Cincinnati_Financial_Close'},inplace=True)


# In[31]:


Citigroup = quandl.get("WIKI/C",start_date="2006-10-20", end_date="2013-10-20")  #Financials
Citizens_Financial_Group = quandl.get("WIKI/CFG",start_date="2006-10-20", end_date="2013-10-20")
CME_Group = quandl.get("WIKI/CME",start_date="2006-10-20", end_date="2013-10-20")
Comerica = quandl.get("WIKI/CMA",start_date="2006-10-20", end_date="2013-10-20")
Discover_Financial_Services = quandl.get("WIKI/DFS",start_date="2006-10-20", end_date="2013-10-20")
ETrade = quandl.get("WIKI/ETFC",start_date="2006-10-20", end_date="2013-10-20")
Everest_Re_Group = quandl.get("WIKI/RE",start_date="2006-10-20", end_date="2013-10-20")
Fifth_Third_Bancorp = quandl.get("WIKI/FITB",start_date="2006-10-20", end_date="2013-10-20")
First_Republic_Bank = quandl.get("WIKI/FRC",start_date="2006-10-20", end_date="2013-10-20")
Franklin_Resources = quandl.get("WIKI/BEN",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Citigroup.reset_index(inplace=True)
Citizens_Financial_Group.reset_index(inplace=True)
CME_Group.reset_index(inplace=True)
Comerica.reset_index(inplace=True)
Discover_Financial_Services.reset_index(inplace=True)
ETrade.reset_index(inplace=True)
Everest_Re_Group.reset_index(inplace=True)
Fifth_Third_Bancorp.reset_index(inplace=True)
First_Republic_Bank.reset_index(inplace=True)
Franklin_Resources.reset_index(inplace=True)


#Removing/Renaming columns
Citigroup = Citigroup [['Date','Open','Close']]
Citigroup.rename(columns={'Open': 'Citigroup_Open'},inplace=True)
Citigroup.rename(columns={'Close': 'Citigroup_Close'},inplace=True)


Citizens_Financial_Group = Citizens_Financial_Group [['Date','Open','Close']]
Citizens_Financial_Group.rename(columns={'Open': 'Citizens_Financial_Group_Open'},inplace=True)
Citizens_Financial_Group.rename(columns={'Close': 'Citizens_Financial_Group_Close'},inplace=True)

CME_Group = CME_Group [['Date','Open','Close']]
CME_Group.rename(columns={'Open': 'CME_Group_Open'},inplace=True)
CME_Group.rename(columns={'Close': 'CME_Group_Close'},inplace=True)


Comerica = Comerica [['Date','Open','Close']]
Comerica.rename(columns={'Open': 'Comerica_Open'},inplace=True)
Comerica.rename(columns={'Close': 'Comerica_Close'},inplace=True)

Discover_Financial_Services = Discover_Financial_Services [['Date','Open','Close']]
Discover_Financial_Services.rename(columns={'Open': 'Discover_Financial_Services_Open'},inplace=True)
Discover_Financial_Services.rename(columns={'Close': 'Discover_Financial_Services_Close'},inplace=True)


ETrade = ETrade [['Date','Open','Close']]
ETrade.rename(columns={'Open': 'ETrade_Open'},inplace=True)
ETrade.rename(columns={'Close': 'ETrade_Close'},inplace=True)


Everest_Re_Group = Everest_Re_Group [['Date','Open','Close']]
Everest_Re_Group.rename(columns={'Open': 'Everest_Re_Group_Open'},inplace=True)
Everest_Re_Group.rename(columns={'Close': 'Everest_Re_Group_Close'},inplace=True)


Fifth_Third_Bancorp = Fifth_Third_Bancorp [['Date','Open','Close']]
Fifth_Third_Bancorp.rename(columns={'Open': 'Fifth_Third_Bancorp_Open'},inplace=True)
Fifth_Third_Bancorp.rename(columns={'Close': 'Fifth_Third_Bancorp_Close'},inplace=True)

First_Republic_Bank = First_Republic_Bank [['Date','Open','Close']]
First_Republic_Bank.rename(columns={'Open': 'First_Republic_Bank_Open'},inplace=True)
First_Republic_Bank.rename(columns={'Close': 'First_Republic_Bank_Close'},inplace=True)


Franklin_Resources = Franklin_Resources [['Date','Open','Close']]
Franklin_Resources.rename(columns={'Open': 'Franklin_Resources_Open'},inplace=True)
Franklin_Resources.rename(columns={'Close': 'Franklin_Resources_Close'},inplace=True)



# In[32]:


Goldman_Sachs_Group = quandl.get("WIKI/GS",start_date="2006-10-20", end_date="2013-10-20") #Financials
Hartford_Financial = quandl.get("WIKI/HIG",start_date="2006-10-20", end_date="2013-10-20")
Huntington_Bancshares = quandl.get("WIKI/HBAN",start_date="2006-10-20", end_date="2013-10-20")
Intercontinental_Exchange = quandl.get("WIKI/ICE",start_date="2006-10-20", end_date="2013-10-20")
Invesco = quandl.get("WIKI/IVZ",start_date="2006-10-20", end_date="2013-10-20")
#Jefferies_Financial_Group = quandl.get("WIKI/JEF",start_date="2006-10-20", end_date="2013-10-20")
JPMorgan_Chase = quandl.get("WIKI/JPM",start_date="2006-10-20", end_date="2013-10-20")
KeyCorp = quandl.get("WIKI/KEY",start_date="2006-10-20", end_date="2013-10-20")
Lincoln_National = quandl.get("WIKI/LNC",start_date="2006-10-20", end_date="2013-10-20")
Loews = quandl.get("WIKI/L",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Goldman_Sachs_Group.reset_index(inplace=True)
Hartford_Financial.reset_index(inplace=True)
Huntington_Bancshares.reset_index(inplace=True)
Intercontinental_Exchange.reset_index(inplace=True)
Invesco.reset_index(inplace=True)
#Jefferies_Financial_Group.reset_index(inplace=True)
JPMorgan_Chase.reset_index(inplace=True)
KeyCorp.reset_index(inplace=True)
Lincoln_National.reset_index(inplace=True)
Loews.reset_index(inplace=True)


#Removing/Renaming columns
Goldman_Sachs_Group = Goldman_Sachs_Group [['Date','Open','Close']]
Goldman_Sachs_Group.rename(columns={'Open': 'Goldman_Sachs_Group_Open'},inplace=True)
Goldman_Sachs_Group.rename(columns={'Close': 'Goldman_Sachs_Group_Close'},inplace=True)


Hartford_Financial = Hartford_Financial [['Date','Open','Close']]
Hartford_Financial.rename(columns={'Open': 'Hartford_Financial_Open'},inplace=True)
Hartford_Financial.rename(columns={'Close': 'Hartford_Financial_Close'},inplace=True)

Huntington_Bancshares = Huntington_Bancshares [['Date','Open','Close']]
Huntington_Bancshares.rename(columns={'Open': 'Huntington_Bancshares_Open'},inplace=True)
Huntington_Bancshares.rename(columns={'Close': 'Huntington_Bancshares_Close'},inplace=True)

Intercontinental_Exchange = Intercontinental_Exchange [['Date','Open','Close']]
Intercontinental_Exchange.rename(columns={'Open': 'Intercontinental_Exchange_Open'},inplace=True)
Intercontinental_Exchange.rename(columns={'Close': 'Intercontinental_Exchange_Close'},inplace=True)

Invesco = Invesco [['Date','Open','Close']]
Invesco.rename(columns={'Open': 'Invesco_Open'},inplace=True)
Invesco.rename(columns={'Close': 'Invesco_Close'},inplace=True)

JPMorgan_Chase = JPMorgan_Chase [['Date','Open','Close']]
JPMorgan_Chase.rename(columns={'Open': 'JPMorgan_Chase_Open'},inplace=True)
JPMorgan_Chase.rename(columns={'Close': 'JPMorgan_Chase_Close'},inplace=True)

KeyCorp = KeyCorp [['Date','Open','Close']]
KeyCorp.rename(columns={'Open': 'KeyCorp_Open'},inplace=True)
KeyCorp.rename(columns={'Close': 'KeyCorp_Close'},inplace=True)

Lincoln_National = Lincoln_National [['Date','Open','Close']]
Lincoln_National.rename(columns={'Open': 'Lincoln_National_Open'},inplace=True)
Lincoln_National.rename(columns={'Close': 'Lincoln_National_Close'},inplace=True)

Loews = Loews [['Date','Open','Close']]
Loews.rename(columns={'Open': 'Loews_Open'},inplace=True)
Loews.rename(columns={'Close': 'Loews_Close'},inplace=True)


# In[33]:


MT_Bank = quandl.get("WIKI/MTB",start_date="2006-10-20", end_date="2013-10-20")#Financials
Marsh_McLennan  = quandl.get("WIKI/MMC",start_date="2006-10-20", end_date="2013-10-20")
MetLife = quandl.get("WIKI/MET",start_date="2006-10-20", end_date="2013-10-20")
Moodys  = quandl.get("WIKI/MCO",start_date="2006-10-20", end_date="2013-10-20")
Morgan_Stanley = quandl.get("WIKI/MS",start_date="2006-10-20", end_date="2013-10-20")
MSCI = quandl.get("WIKI/MSCI",start_date="2006-10-20", end_date="2013-10-20")
Nasdaq = quandl.get("WIKI/NDAQ",start_date="2006-10-20", end_date="2013-10-20")
Northern_Trust = quandl.get("WIKI/NTRS",start_date="2006-10-20", end_date="2013-10-20")
Peoples_United_Financial = quandl.get("WIKI/PBCT",start_date="2006-10-20", end_date="2013-10-20")
PNC_Financial_Services = quandl.get("WIKI/PNC",start_date="2006-10-20", end_date="2013-10-20")



#indexing
MT_Bank.reset_index(inplace=True)
Marsh_McLennan.reset_index(inplace=True)
MetLife.reset_index(inplace=True)
Moodys.reset_index(inplace=True)
Morgan_Stanley.reset_index(inplace=True)
MSCI.reset_index(inplace=True)
Nasdaq.reset_index(inplace=True)
Northern_Trust.reset_index(inplace=True)
Peoples_United_Financial.reset_index(inplace=True)
PNC_Financial_Services.reset_index(inplace=True)

#Removing/Renaming columns
MT_Bank = MT_Bank [['Date','Open','Close']]
MT_Bank.rename(columns={'Open': 'MT_Bank_Open'},inplace=True)
MT_Bank.rename(columns={'Close': 'MT_Bank_Close'},inplace=True)

Marsh_McLennan = Marsh_McLennan [['Date','Open','Close']]
Marsh_McLennan.rename(columns={'Open': 'Marsh_McLennan_Open'},inplace=True)
Marsh_McLennan.rename(columns={'Close': 'Marsh_McLennan_Close'},inplace=True)

MetLife = MetLife [['Date','Open','Close']]
MetLife.rename(columns={'Open': 'MetLife_Open'},inplace=True)
MetLife.rename(columns={'Close': 'MetLife_Close'},inplace=True)

Moodys = Moodys [['Date','Open','Close']]
Moodys.rename(columns={'Open': 'Moodys_Open'},inplace=True)
Moodys.rename(columns={'Close': 'Moodys_Close'},inplace=True)

Morgan_Stanley = Morgan_Stanley [['Date','Open','Close']]
Morgan_Stanley.rename(columns={'Open': 'Morgan_Stanley_Open'},inplace=True)
Morgan_Stanley.rename(columns={'Close': 'Morgan_Stanley_Close'},inplace=True)

MSCI = MSCI [['Date','Open','Close']]
MSCI.rename(columns={'Open': 'MSCI_Open'},inplace=True)
MSCI.rename(columns={'Close': 'MSCI_Close'},inplace=True)

Nasdaq = Nasdaq [['Date','Open','Close']]
Nasdaq.rename(columns={'Open': 'Nasdaq_Open'},inplace=True)
Nasdaq.rename(columns={'Close': 'Nasdaq_Close'},inplace=True)

Northern_Trust = Northern_Trust [['Date','Open','Close']]
Northern_Trust.rename(columns={'Open': 'Northern_Trust_Open'},inplace=True)
Northern_Trust.rename(columns={'Close': 'Northern_Trust_Close'},inplace=True)

Peoples_United_Financial = Peoples_United_Financial [['Date','Open','Close']]
Peoples_United_Financial.rename(columns={'Open': 'Peoples_United_Financial_Open'},inplace=True)
Peoples_United_Financial.rename(columns={'Close': 'Peoples_United_Financial_Close'},inplace=True)

PNC_Financial_Services = PNC_Financial_Services [['Date','Open','Close']]
PNC_Financial_Services.rename(columns={'Open': 'PNC_Financial_Services_Open'},inplace=True)
PNC_Financial_Services.rename(columns={'Close': 'PNC_Financial_Services_Close'},inplace=True)


# In[34]:


Principal_Financial_Group  = quandl.get("WIKI/PFG",start_date="2006-10-20", end_date="2013-10-20") #Financials
Progressive = quandl.get("WIKI/PGR",start_date="2006-10-20", end_date="2013-10-20")
Prudential_Financial  = quandl.get("WIKI/PRU",start_date="2006-10-20", end_date="2013-10-20")
Raymond_James_Financial  = quandl.get("WIKI/RJF",start_date="2006-10-20", end_date="2013-10-20")
Regions_Financial  = quandl.get("WIKI/RF",start_date="2006-10-20", end_date="2013-10-20")
SP_Global = quandl.get("WIKI/SPGI",start_date="2006-10-20", end_date="2013-10-20")
State_Street  = quandl.get("WIKI/STT",start_date="2006-10-20", end_date="2013-10-20")
SunTrust_Banks = quandl.get("WIKI/STI",start_date="2006-10-20", end_date="2013-10-20")
SVB_Financial = quandl.get("WIKI/SIVB",start_date="2006-10-20", end_date="2013-10-20")
Synchrony_Financial = quandl.get("WIKI/SYF",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Principal_Financial_Group.reset_index(inplace=True)
Progressive.reset_index(inplace=True)
Prudential_Financial.reset_index(inplace=True)
Raymond_James_Financial.reset_index(inplace=True)
Regions_Financial.reset_index(inplace=True)
SP_Global.reset_index(inplace=True)
State_Street.reset_index(inplace=True)
SunTrust_Banks.reset_index(inplace=True)
SVB_Financial.reset_index(inplace=True)
Synchrony_Financial.reset_index(inplace=True)

#Removing/Renaming columns
Principal_Financial_Group = Principal_Financial_Group [['Date','Open','Close']]
Principal_Financial_Group.rename(columns={'Open': 'Principal_Financial_Group_Open'},inplace=True)
Principal_Financial_Group.rename(columns={'Close': 'Principal_Financial_Group_Close'},inplace=True)

Progressive = Progressive [['Date','Open','Close']]
Progressive.rename(columns={'Open': 'Progressive_Open'},inplace=True)
Progressive.rename(columns={'Close': 'Progressive_Close'},inplace=True)

Prudential_Financial = Prudential_Financial [['Date','Open','Close']]
Prudential_Financial.rename(columns={'Open': 'Prudential_Financial_Open'},inplace=True)
Prudential_Financial.rename(columns={'Close': 'Prudential_Financial_Close'},inplace=True)

Raymond_James_Financial = Raymond_James_Financial [['Date','Open','Close']]
Raymond_James_Financial.rename(columns={'Open': 'Raymond_James_Financial_Open'},inplace=True)
Raymond_James_Financial.rename(columns={'Close': 'Raymond_James_Financial_Close'},inplace=True)

Regions_Financial = Regions_Financial [['Date','Open','Close']]
Regions_Financial.rename(columns={'Open': 'Regions_Financial_Open'},inplace=True)
Regions_Financial.rename(columns={'Close': 'Regions_Financial_Close'},inplace=True)

SP_Global = SP_Global [['Date','Open','Close']]
SP_Global.rename(columns={'Open': 'SP_Global_Open'},inplace=True)
SP_Global.rename(columns={'Close': 'SP_Global_Close'},inplace=True)

State_Street = State_Street [['Date','Open','Close']]
State_Street.rename(columns={'Open': 'State_Street_Open'},inplace=True)
State_Street.rename(columns={'Close': 'State_Street_Close'},inplace=True)

SunTrust_Banks = SunTrust_Banks [['Date','Open','Close']]
SunTrust_Banks.rename(columns={'Open': 'SunTrust_Banks_Open'},inplace=True)
SunTrust_Banks.rename(columns={'Close': 'SunTrust_Banks_Close'},inplace=True)

SVB_Financial = SVB_Financial [['Date','Open','Close']]
SVB_Financial.rename(columns={'Open': 'SVB_Financial_Open'},inplace=True)
SVB_Financial.rename(columns={'Close': 'SVB_Financial_Close'},inplace=True)

Synchrony_Financial = Synchrony_Financial [['Date','Open','Close']]
Synchrony_Financial.rename(columns={'Open': 'Synchrony_Financial_Open'},inplace=True)
Synchrony_Financial.rename(columns={'Close': 'Synchrony_Financial_Close'},inplace=True)


# In[35]:


T_Rowe_Price_Group  = quandl.get("WIKI/TROW",start_date="2006-10-20", end_date="2013-10-20") #Financials
Torchmark = quandl.get("WIKI/TMK",start_date="2006-10-20", end_date="2013-10-20")
The_Travelers_Companies = quandl.get("WIKI/TRV",start_date="2006-10-20", end_date="2013-10-20")
US_Bancorp  = quandl.get("WIKI/USB",start_date="2006-10-20", end_date="2013-10-20")
Unum_Group = quandl.get("WIKI/UNM",start_date="2006-10-20", end_date="2013-10-20")
Wells_Fargo   = quandl.get("WIKI/WFC",start_date="2006-10-20", end_date="2013-10-20")
Willis_Towers_Watson = quandl.get("WIKI/WLTW",start_date="2006-10-20", end_date="2013-10-20")
Zions_Bancorp = quandl.get("WIKI/ZION",start_date="2006-10-20", end_date="2013-10-20")




#indexing
T_Rowe_Price_Group.reset_index(inplace=True)
Torchmark.reset_index(inplace=True)
The_Travelers_Companies.reset_index(inplace=True)
US_Bancorp.reset_index(inplace=True)
Unum_Group.reset_index(inplace=True)
Wells_Fargo.reset_index(inplace=True)
Willis_Towers_Watson.reset_index(inplace=True)
Zions_Bancorp.reset_index(inplace=True)


#Removing/Renaming columns
T_Rowe_Price_Group = T_Rowe_Price_Group [['Date','Open','Close']]
T_Rowe_Price_Group.rename(columns={'Open': 'T_Rowe_Price_Group_Open'},inplace=True)
T_Rowe_Price_Group.rename(columns={'Close': 'T_Rowe_Price_Group_Close'},inplace=True)

Torchmark = Torchmark [['Date','Open','Close']]
Torchmark.rename(columns={'Open': 'Torchmark_Open'},inplace=True)
Torchmark.rename(columns={'Close': 'Torchmark_Close'},inplace=True)

The_Travelers_Companies = The_Travelers_Companies [['Date','Open','Close']]
The_Travelers_Companies.rename(columns={'Open': 'The_Travelers_Companies_Open'},inplace=True)
The_Travelers_Companies.rename(columns={'Close': 'The_Travelers_Companies_Close'},inplace=True)

US_Bancorp = US_Bancorp [['Date','Open','Close']]
US_Bancorp.rename(columns={'Open': 'US_Bancorp_Open'},inplace=True)
US_Bancorp.rename(columns={'Close': 'US_Bancorp_Close'},inplace=True)

Unum_Group = Unum_Group [['Date','Open','Close']]
Unum_Group.rename(columns={'Open': 'Unum_Group_Open'},inplace=True)
Unum_Group.rename(columns={'Close': 'Unum_Group_Close'},inplace=True)

Wells_Fargo = Wells_Fargo [['Date','Open','Close']]
Wells_Fargo.rename(columns={'Open': 'Wells_Fargo_Open'},inplace=True)
Wells_Fargo.rename(columns={'Close': 'Wells_Fargo_Close'},inplace=True)

Willis_Towers_Watson = Willis_Towers_Watson [['Date','Open','Close']]
Willis_Towers_Watson.rename(columns={'Open': 'Willis_Towers_Watson_Open'},inplace=True)
Willis_Towers_Watson.rename(columns={'Close': 'Willis_Towers_Watson_Close'},inplace=True)

T_Rowe_Price_Group  = quandl.get("WIKI/TROW",start_date="2006-10-20", end_date="2013-10-20") #Financials
Torchmark = quandl.get("WIKI/TMK",start_date="2006-10-20", end_date="2013-10-20")
The_Travelers_Companies = quandl.get("WIKI/TRV",start_date="2006-10-20", end_date="2013-10-20")
US_Bancorp  = quandl.get("WIKI/USB",start_date="2006-10-20", end_date="2013-10-20")
Unum_Group = quandl.get("WIKI/UNM",start_date="2006-10-20", end_date="2013-10-20")
Wells_Fargo   = quandl.get("WIKI/WFC",start_date="2006-10-20", end_date="2013-10-20")
Willis_Towers_Watson = quandl.get("WIKI/WLTW",start_date="2006-10-20", end_date="2013-10-20")
Zions_Bancorp = quandl.get("WIKI/ZION",start_date="2006-10-20", end_date="2013-10-20")




#indexing
T_Rowe_Price_Group.reset_index(inplace=True)
Torchmark.reset_index(inplace=True)
The_Travelers_Companies.reset_index(inplace=True)
US_Bancorp.reset_index(inplace=True)
Unum_Group.reset_index(inplace=True)
Wells_Fargo.reset_index(inplace=True)
Willis_Towers_Watson.reset_index(inplace=True)
Zions_Bancorp.reset_index(inplace=True)


#Removing/Renaming columns
T_Rowe_Price_Group = T_Rowe_Price_Group [['Date','Open','Close']]
T_Rowe_Price_Group.rename(columns={'Open': 'T_Rowe_Price_Group_Open'},inplace=True)
T_Rowe_Price_Group.rename(columns={'Close': 'T_Rowe_Price_Group_Close'},inplace=True)

Torchmark = Torchmark [['Date','Open','Close']]
Torchmark.rename(columns={'Open': 'Torchmark_Open'},inplace=True)
Torchmark.rename(columns={'Close': 'Torchmark_Close'},inplace=True)

The_Travelers_Companies = The_Travelers_Companies [['Date','Open','Close']]
The_Travelers_Companies.rename(columns={'Open': 'The_Travelers_Companies_Open'},inplace=True)
The_Travelers_Companies.rename(columns={'Close': 'The_Travelers_Companies_Close'},inplace=True)

US_Bancorp = US_Bancorp [['Date','Open','Close']]
US_Bancorp.rename(columns={'Open': 'US_Bancorp_Open'},inplace=True)
US_Bancorp.rename(columns={'Close': 'US_Bancorp_Close'},inplace=True)

Unum_Group = Unum_Group [['Date','Open','Close']]
Unum_Group.rename(columns={'Open': 'Unum_Group_Open'},inplace=True)
Unum_Group.rename(columns={'Close': 'Unum_Group_Close'},inplace=True)

Wells_Fargo = Wells_Fargo [['Date','Open','Close']]
Wells_Fargo.rename(columns={'Open': 'Wells_Fargo_Open'},inplace=True)
Wells_Fargo.rename(columns={'Close': 'Wells_Fargo_Close'},inplace=True)

Willis_Towers_Watson = Willis_Towers_Watson [['Date','Open','Close']]
Willis_Towers_Watson.rename(columns={'Open': 'Willis_Towers_Watson_Open'},inplace=True)
Willis_Towers_Watson.rename(columns={'Close': 'Willis_Towers_Watson_Close'},inplace=True)

Zions_Bancorp = Zions_Bancorp [['Date','Open','Close']]
Zions_Bancorp.rename(columns={'Open': 'Zions_Bancorp_Open'},inplace=True)
Zions_Bancorp.rename(columns={'Close': 'Zions_Bancorp_Close'},inplace=True)


# In[36]:


Affiliated_Managers_Group.shape
AFLAC.shape
Allstate_Corp.shape
American_Express.shape
American_International_Group.shape
Ameriprise_Financial.shape
Aon.shape
Arthur_J_Gallagher.shape
Assurant.shape
Bank_of_America.shape
The_Bank_of_NewYork.shape
BBT.shape
Berkshire_Hathaway.shape
BlackRock.shape
Capital_One_Financial.shape
Cboe_Global_Markets.shape
Charles_Schwab_Corporation.shape
Chubb.shape
Cincinnati_Financial.shape
Citigroup.shape
Citizens_Financial_Group.shape
CME_Group.shape
Comerica.shape
Discover_Financial_Services.shape
ETrade.shape
Everest_Re_Group.shape
Fifth_Third_Bancorp.shape
First_Republic_Bank.shape
Franklin_Resources.shape
Goldman_Sachs_Group.shape
Hartford_Financial.shape
Huntington_Bancshares.shape
Intercontinental_Exchange.shape
Invesco.shape
JPMorgan_Chase.shape
KeyCorp.shape
Lincoln_National.shape
Loews.shape
MT_Bank.shape
Marsh_McLennan.shape
MetLife.shape
Moodys.shape
Morgan_Stanley.shape
MSCI.shape
Nasdaq.shape
Northern_Trust.shape
Peoples_United_Financial.shape
PNC_Financial_Services.shape
Principal_Financial_Group.shape
Progressive.shape
Prudential_Financial.shape
Raymond_James_Financial.shape
Regions_Financial.shape
SP_Global.shape


# In[37]:


#Financials = pd.concat([Affiliated_Managers_Group[['Date','Affiliated_Managers_Group_Open','Affiliated_Managers_Group_Close']], AFLAC[['AFLAC_Open','AFLAC_Close']], Allstate_Corp[['Allstate_Corp_Open', 'Allstate_Corp_Close']], American_Express[['American_Express_Open','American_Express_Close']],American_International_Group[['American_International_Group_Open','American_International_Group_Close']],Ameriprise_Financial[['Ameriprise_Financial_Open','Ameriprise_Financial_Close']],Aon[['Aon_Open','Aon_Close']],Arthur_J_Gallagher[['Arthur_J_Gallagher_Open','Arthur_J_Gallagher_Close']],Assurant[['Assurant_Open','Assurant_Close']],Bank_of_America[['Bank_of_America_Open','Bank_of_America_Close']],The_Bank_of_NewYork[['The_Bank_of_NewYork_Open','The_Bank_of_NewYork_Close']],BBT[['BBT_Open','BBT_Close']],Berkshire_Hathaway[['Berkshire_Hathaway_Open','Berkshire_Hathaway_Close']],BlackRock[['BlackRock_Open','BlackRock_Close']],Brighthouse_Financial[['Brighthouse_Financial_Open','Brighthouse_Financial_Close']],Capital_One_Financial[['Capital_One_Financial_Open','Capital_One_Financial_Close']], Cboe_Global_Markets[['Cboe_Global_Markets_Open','Cboe_Global_Markets_Close']],Charles_Schwab_Corporation[['Charles_Schwab_Corporation_Open','Charles_Schwab_Corporation_Close']],Chubb[['Chubb_Open','Chubb_Close']],Cincinnati_Financial[['Cincinnati_Financial_Open','Cincinnati_Financial_Close']],Citigroup[['Citigroup_Open','Citigroup_Close']],Citizens_Financial_Group[['Citizens_Financial_Group_Open','Citizens_Financial_Group_Close']], CME_Group[['CME_Group_Open','CME_Group_Close']],Comerica[['Comerica_Open','Comerica_Close']],Discover_Financial_Services[['Discover_Financial_Services_Open','Discover_Financial_Services_Close']],ETrade[['ETrade_Open','ETrade_Close']],Everest_Re_Group[['Everest_Re_Group_Open','Everest_Re_Group_Close']],Fifth_Third_Bancorp[['Fifth_Third_Bancorp_Open','Fifth_Third_Bancorp_Close']], First_Republic_Bank[['First_Republic_Bank_Open','First_Republic_Bank_Close']],Franklin_Resources[['Franklin_Resources_Open','Franklin_Resources_Close']], Goldman_Sachs_Group[['Goldman_Sachs_Group_Open', 'Goldman_Sachs_Group_Close']], Hartford_Financial[['Hartford_Financial_Open','Hartford_Financial_Close']],Huntington_Bancshares[['Huntington_Bancshares_Open','Huntington_Bancshares_Close']],Intercontinental_Exchange[['Intercontinental_Exchange_Open','Intercontinental_Exchange_Close']],Invesco[['Invesco_Open','Invesco_Close']],JPMorgan_Chase[['JPMorgan_Chase_Open','JPMorgan_Chase_Close']],KeyCorp[['KeyCorp_Open','KeyCorp_Close']],Lincoln_National[['Lincoln_National_Open','Lincoln_National_Close']],Loews[['Loews_Open','Loews_Close']],MT_Bank[['MT_Bank_Open','MT_Bank_Close']],Marsh_McLennan[['Marsh_McLennan_Open','Marsh_McLennan_Close']],MetLife[['MetLife_Open','MetLife_Close']],Moodys[['Moodys_Open','Moodys_Close']],Morgan_Stanley[['Morgan_Stanley_Open','Morgan_Stanley_Close']], MSCI[['MSCI_Open','MSCI_Close']],Nasdaq[['Nasdaq_Open','Nasdaq_Close']],Northern_Trust[['Northern_Trust_Open','Northern_Trust_Close']],Peoples_United_Financial[['Peoples_United_Financial_Open','Peoples_United_Financial_Close']],PNC_Financial_Services[['PNC_Financial_Services_Open','PNC_Financial_Services_Close']],Principal_Financial_Group[['Principal_Financial_Group_Open','Principal_Financial_Group_Close']], Progressive[['Progressive_Open','Progressive_Close']],Prudential_Financial[['Prudential_Financial_Open','Prudential_Financial_Close']],Raymond_James_Financial[['Raymond_James_Financial_Open','Raymond_James_Financial_Close']],Regions_Financial[['Regions_Financial_Open','Regions_Financial_Close']],SP_Global[['SP_Global_Open','SP_Global_Close']],State_Street[['State_Street_Open','State_Street_Close']], SunTrust_Banks[['SunTrust_Banks_Open','SunTrust_Banks_Close']],SVB_Financial[['SVB_Financial_Open','SVB_Financial_Close']],Synchrony_Financial[['Synchrony_Financial_Open','Synchrony_Financial_Close']],T_Rowe_Price_Group[['T_Rowe_Price_Group_Open','T_Rowe_Price_Group_Close']],Torchmark[['Torchmark_Open','Torchmark_Close']],The_Travelers_Companies[['The_Travelers_Companies_Open','The_Travelers_Companies_Close']],US_Bancorp[['US_Bancorp_Open','US_Bancorp_Close']], Unum_Group[['Unum_Group_Open','Unum_Group_Close']],Wells_Fargo[['Wells_Fargo_Open','Wells_Fargo_Close']],Willis_Towers_Watson[['Willis_Towers_Watson_Open','Willis_Towers_Watson_Close']],Zions_Bancorp[['Zions_Bancorp_Open','Zions_Bancorp_Close']]],axis=1)

Financials = (Affiliated_Managers_Group).merge(AFLAC).merge(Allstate_Corp).merge(American_Express).merge(American_International_Group).merge(Ameriprise_Financial).merge(Aon).merge(Arthur_J_Gallagher).merge(Assurant).merge(Bank_of_America).merge(The_Bank_of_NewYork).merge(BBT).merge(Berkshire_Hathaway).merge(BlackRock).merge(Capital_One_Financial).merge(Cboe_Global_Markets).merge(Charles_Schwab_Corporation).merge(Chubb).merge(Cincinnati_Financial).merge(Citigroup).merge(CME_Group).merge(Comerica).merge(Discover_Financial_Services).merge(ETrade).merge(Everest_Re_Group).merge(Fifth_Third_Bancorp).merge(First_Republic_Bank).merge(Franklin_Resources).merge(Goldman_Sachs_Group).merge(Hartford_Financial).merge(Huntington_Bancshares).merge(Intercontinental_Exchange).merge(Invesco).merge(JPMorgan_Chase).merge(KeyCorp).merge(Lincoln_National).merge(Loews).merge(MT_Bank).merge(Marsh_McLennan).merge(MetLife).merge(Moodys).merge(Morgan_Stanley).merge(MSCI).merge(Nasdaq).merge(Northern_Trust).merge(Peoples_United_Financial).merge(PNC_Financial_Services).merge(Principal_Financial_Group).merge(Progressive).merge(Prudential_Financial).merge(Raymond_James_Financial).merge(Regions_Financial).merge(State_Street).merge(SunTrust_Banks).merge(SVB_Financial).merge(T_Rowe_Price_Group).merge(Torchmark).merge(US_Bancorp).merge(Unum_Group).merge(Wells_Fargo).merge(Zions_Bancorp)
Financials
#Brighthouse_Financial
#SP_Global
#Synchrony_Financial
#The_Travelers_Companies
#Willis_Towers_Watson
#Citizens_Financial_Group

#Financials.isnull().sum()


# In[38]:


Abbott_Laboratories = quandl.get("WIKI/ABT",start_date="2006-10-20", end_date="2013-10-20") #HealthCare
AbbVie = quandl.get("WIKI/ABBV",start_date="2006-10-20", end_date="2013-10-20")
ABIOMED = quandl.get("WIKI/ABMD",start_date="2006-10-20", end_date="2013-10-20")
Agilent = quandl.get("WIKI/A",start_date="2006-10-20", end_date="2013-10-20")
Alexion_Pharmaceuticals = quandl.get("WIKI/ALXN",start_date="2006-10-20", end_date="2013-10-20")
Align_Technology = quandl.get("WIKI/ALGN",start_date="2006-10-20", end_date="2013-10-20")
Allergan = quandl.get("WIKI/AGN",start_date="2006-10-20", end_date="2013-10-20")
AmerisourceBergen_Corp = quandl.get("WIKI/ABC",start_date="2006-10-20", end_date="2013-10-20")
Amgen = quandl.get("WIKI/AMGN",start_date="2006-10-20", end_date="2013-10-20")
Anthem = quandl.get("WIKI/ANTM",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Abbott_Laboratories.reset_index(inplace=True)
AbbVie.reset_index(inplace=True)
ABIOMED.reset_index(inplace=True)
Agilent.reset_index(inplace=True)
Alexion_Pharmaceuticals.reset_index(inplace=True)
Align_Technology.reset_index(inplace=True)
Allergan.reset_index(inplace=True)
AmerisourceBergen_Corp.reset_index(inplace=True)
Amgen.reset_index(inplace=True)
Anthem.reset_index(inplace=True)


#Removing/Renaming columns
Abbott_Laboratories = Abbott_Laboratories [['Date','Open','Close']]
Abbott_Laboratories.rename(columns={'Open': 'Abbott_Laboratories_Open'},inplace=True)
Abbott_Laboratories.rename(columns={'Close': 'Abbott_Laboratories_Close'},inplace=True)


AbbVie = AbbVie [['Date','Open','Close']]
AbbVie.rename(columns={'Open': 'AbbVie_Open'},inplace=True)
AbbVie.rename(columns={'Close': 'AbbVie_Close'},inplace=True)


ABIOMED = ABIOMED [['Date','Open','Close']]
ABIOMED.rename(columns={'Open': 'ABIOMED_Open'},inplace=True)
ABIOMED.rename(columns={'Close': 'ABIOMED_Close'},inplace=True)

Agilent = Agilent [['Date','Open','Close']]
Agilent.rename(columns={'Open': 'Agilent_Open'},inplace=True)
Agilent.rename(columns={'Close': 'Agilent_Close'},inplace=True)

Alexion_Pharmaceuticals = Alexion_Pharmaceuticals [['Date','Open','Close']]
Alexion_Pharmaceuticals.rename(columns={'Open': 'Alexion_Pharmaceuticals_Open'},inplace=True)
Alexion_Pharmaceuticals.rename(columns={'Close': 'Alexion_Pharmaceuticals_Close'},inplace=True)

Align_Technology = Align_Technology [['Date','Open','Close']]
Align_Technology.rename(columns={'Open': 'Align_Technology_Open'},inplace=True)
Align_Technology.rename(columns={'Close': 'Align_Technology_Close'},inplace=True)

Allergan = Allergan [['Date','Open','Close']]
Allergan.rename(columns={'Open': 'Allergan_Open'},inplace=True)
Allergan.rename(columns={'Close': 'Allergan_Close'},inplace=True)

AmerisourceBergen_Corp = AmerisourceBergen_Corp [['Date','Open','Close']]
AmerisourceBergen_Corp.rename(columns={'Open': 'AmerisourceBergen_Corp_Open'},inplace=True)
AmerisourceBergen_Corp.rename(columns={'Close': 'AmerisourceBergen_Corp_Close'},inplace=True)

Amgen = Amgen [['Date','Open','Close']]
Amgen.rename(columns={'Open': 'Amgen_Open'},inplace=True)
Amgen.rename(columns={'Close': 'Amgen_Close'},inplace=True)

Anthem = Anthem [['Date','Open','Close']]
Anthem.rename(columns={'Open': 'Anthem_Open'},inplace=True)
Anthem.rename(columns={'Close': 'Anthem_Close'},inplace=True)


# In[39]:


Baxter_International = quandl.get("WIKI/BAX",start_date="2006-10-20", end_date="2013-10-20") #HealthCare
Becton_Dickinson = quandl.get("WIKI/BDX",start_date="2006-10-20", end_date="2013-10-20")
Biogen = quandl.get("WIKI/BIIB",start_date="2006-10-20", end_date="2013-10-20")
Boston_Scientific = quandl.get("WIKI/BSX",start_date="2006-10-20", end_date="2013-10-20")
Bristol_Myers_Squibb = quandl.get("WIKI/BMY",start_date="2006-10-20", end_date="2013-10-20")
Cardinal_Health = quandl.get("WIKI/CAH",start_date="2006-10-20", end_date="2013-10-20")
Celgene = quandl.get("WIKI/CELG",start_date="2006-10-20", end_date="2013-10-20")
Centene_Corporation  = quandl.get("WIKI/CNC",start_date="2006-10-20", end_date="2013-10-20")
Cerner = quandl.get("WIKI/CERN",start_date="2006-10-20", end_date="2013-10-20")
CIGNA  = quandl.get("WIKI/CI",start_date="2006-10-20", end_date="2013-10-20")




#indexing
Baxter_International.reset_index(inplace=True)
Becton_Dickinson.reset_index(inplace=True)
Biogen.reset_index(inplace=True)
Boston_Scientific.reset_index(inplace=True)
Bristol_Myers_Squibb.reset_index(inplace=True)
Cardinal_Health.reset_index(inplace=True)
Celgene.reset_index(inplace=True)
Centene_Corporation.reset_index(inplace=True)
#Jefferies_Financial_Group = quandl.get("WIKI/JEF",start_date="2006-10-20", end_date="2013-10-20")
Cerner.reset_index(inplace=True)
CIGNA.reset_index(inplace=True)

#Removing/Renaming columns
Baxter_International = Baxter_International [['Date','Open','Close']]
Baxter_International.rename(columns={'Open': 'Baxter_International_Open'},inplace=True)
Baxter_International.rename(columns={'Close': 'Baxter_International_Close'},inplace=True)

Becton_Dickinson = Becton_Dickinson [['Date','Open','Close']]
Becton_Dickinson.rename(columns={'Open': 'Becton_Dickinson_Open'},inplace=True)
Becton_Dickinson.rename(columns={'Close': 'Becton_Dickinson_Close'},inplace=True)

Biogen = Biogen [['Date','Open','Close']]
Biogen.rename(columns={'Open': 'Biogen_Open'},inplace=True)
Biogen.rename(columns={'Close': 'Biogen_Close'},inplace=True)

Boston_Scientific = Boston_Scientific [['Date','Open','Close']]
Boston_Scientific.rename(columns={'Open': 'Boston_Scientific_Open'},inplace=True)
Boston_Scientific.rename(columns={'Close': 'Boston_Scientific_Close'},inplace=True)

Bristol_Myers_Squibb = Bristol_Myers_Squibb [['Date','Open','Close']]
Bristol_Myers_Squibb.rename(columns={'Open': 'Bristol_Myers_Squibb_Open'},inplace=True)
Bristol_Myers_Squibb.rename(columns={'Close': 'Bristol_Myers_Squibb_Close'},inplace=True)

Cardinal_Health = Cardinal_Health [['Date','Open','Close']]
Cardinal_Health.rename(columns={'Open': 'Cardinal_Health_Open'},inplace=True)
Cardinal_Health.rename(columns={'Close': 'Cardinal_Health_Close'},inplace=True)

Celgene = Celgene [['Date','Open','Close']]
Celgene.rename(columns={'Open': 'Celgene_Open'},inplace=True)
Celgene.rename(columns={'Close': 'Celgene_Close'},inplace=True)

Centene_Corporation = Centene_Corporation [['Date','Open','Close']]
Centene_Corporation.rename(columns={'Open': 'Centene_Corporation_Open'},inplace=True)
Centene_Corporation.rename(columns={'Close': 'Centene_Corporation_Close'},inplace=True)

Cerner = Cerner [['Date','Open','Close']]
Cerner.rename(columns={'Open': 'Cerner_Open'},inplace=True)
Cerner.rename(columns={'Close': 'Cerner_Close'},inplace=True)

CIGNA = CIGNA [['Date','Open','Close']]
CIGNA.rename(columns={'Open': 'CIGNA_Open'},inplace=True)
CIGNA.rename(columns={'Close': 'CIGNA_Close'},inplace=True)


# In[40]:


The_Cooper_Companies = quandl.get("WIKI/COO",start_date="2006-10-20", end_date="2013-10-20") #HelathCare
CVS_Health = quandl.get("WIKI/CVS",start_date="2006-10-20", end_date="2013-10-20")
Danaher = quandl.get("WIKI/DHR",start_date="2006-10-20", end_date="2013-10-20")
DaVita = quandl.get("WIKI/DVA",start_date="2006-10-20", end_date="2013-10-20")
Dentsply_Sirona  = quandl.get("WIKI/XRAY",start_date="2006-10-20", end_date="2013-10-20")
Edwards_Lifesciences = quandl.get("WIKI/EW",start_date="2006-10-20", end_date="2013-10-20")
Gilead_Sciences = quandl.get("WIKI/GILD",start_date="2006-10-20", end_date="2013-10-20")
HCA_Holdings = quandl.get("WIKI/HCA",start_date="2006-10-20", end_date="2013-10-20")
Henry_Schein = quandl.get("WIKI/HSIC",start_date="2006-10-20", end_date="2013-10-20")
Hologic = quandl.get("WIKI/HOLX",start_date="2006-10-20", end_date="2013-10-20")



#indexing
The_Cooper_Companies.reset_index(inplace=True)
CVS_Health.reset_index(inplace=True)
Danaher.reset_index(inplace=True)
Dentsply_Sirona.reset_index(inplace=True)
Edwards_Lifesciences.reset_index(inplace=True)
Gilead_Sciences.reset_index(inplace=True)
HCA_Holdings.reset_index(inplace=True)
Henry_Schein.reset_index(inplace=True)
Hologic.reset_index(inplace=True)
DaVita.reset_index(inplace=True)


#Removing/Renaming columns
The_Cooper_Companies = The_Cooper_Companies [['Date','Open','Close']]
The_Cooper_Companies.rename(columns={'Open': 'The_Cooper_Companies_Open'},inplace=True)
The_Cooper_Companies.rename(columns={'Close': 'The_Cooper_Companies_Close'},inplace=True)

CVS_Health = CVS_Health [['Date','Open','Close']]
CVS_Health.rename(columns={'Open': 'CVS_Health_Open'},inplace=True)
CVS_Health.rename(columns={'Close': 'CVS_Health_Close'},inplace=True)

Danaher = Danaher [['Date','Open','Close']]
Danaher.rename(columns={'Open': 'Danaher_Open'},inplace=True)
Danaher.rename(columns={'Close': 'Danaher_Close'},inplace=True)

Dentsply_Sirona = Dentsply_Sirona [['Date','Open','Close']]
Dentsply_Sirona.rename(columns={'Open': 'Dentsply_Sirona_Open'},inplace=True)
Dentsply_Sirona.rename(columns={'Close': 'Dentsply_Sirona_Close'},inplace=True)

Edwards_Lifesciences = Edwards_Lifesciences [['Date','Open','Close']]
Edwards_Lifesciences.rename(columns={'Open': 'Edwards_Lifesciences_Open'},inplace=True)
Edwards_Lifesciences.rename(columns={'Close': 'Edwards_Lifesciences_Close'},inplace=True)

Gilead_Sciences = Gilead_Sciences [['Date','Open','Close']]
Gilead_Sciences.rename(columns={'Open': 'Gilead_Sciences_Open'},inplace=True)
Gilead_Sciences.rename(columns={'Close': 'Gilead_Sciences_Close'},inplace=True)

HCA_Holdings = HCA_Holdings [['Date','Open','Close']]
HCA_Holdings.rename(columns={'Open': 'HCA_Holdings_Open'},inplace=True)
HCA_Holdings.rename(columns={'Close': 'HCA_Holdings_Close'},inplace=True)

Henry_Schein = Henry_Schein [['Date','Open','Close']]
Henry_Schein.rename(columns={'Open': 'Henry_Schein_Open'},inplace=True)
Henry_Schein.rename(columns={'Close': 'Henry_Schein_Close'},inplace=True)

Hologic = Hologic [['Date','Open','Close']]
Hologic.rename(columns={'Open': 'Hologic_Open'},inplace=True)
Hologic.rename(columns={'Close': 'Hologic_Close'},inplace=True)

DaVita = DaVita [['Date','Open','Close']]
DaVita.rename(columns={'Open': 'DaVita_Open'},inplace=True)
DaVita.rename(columns={'Close': 'DaVita_Close'},inplace=True)


# In[41]:


Humana = quandl.get("WIKI/HUM",start_date="2006-10-20", end_date="2013-10-20")#HealthCare
IDEXX_Laboratories = quandl.get("WIKI/IDXX",start_date="2006-10-20", end_date="2013-10-20")
Illumina  = quandl.get("WIKI/ILMN",start_date="2006-10-20", end_date="2013-10-20")
Incyte = quandl.get("WIKI/INCY",start_date="2006-10-20", end_date="2013-10-20")
Intuitive_Surgical  = quandl.get("WIKI/ISRG",start_date="2006-10-20", end_date="2013-10-20")
IQVIA_Holdings = quandl.get("WIKI/IQV",start_date="2006-10-20", end_date="2013-10-20")
Johnson_Johnson = quandl.get("WIKI/JNJ",start_date="2006-10-20", end_date="2013-10-20")
Laboratory_America_Holding  = quandl.get("WIKI/LH",start_date="2006-10-20", end_date="2013-10-20")
Lilly = quandl.get("WIKI/LLY",start_date="2006-10-20", end_date="2013-10-20")
McKesson  = quandl.get("WIKI/MCK",start_date="2006-10-20", end_date="2013-10-20")





#indexing
Humana.reset_index(inplace=True)
IDEXX_Laboratories.reset_index(inplace=True)
Illumina.reset_index(inplace=True)
Incyte.reset_index(inplace=True)
Intuitive_Surgical.reset_index(inplace=True)
IQVIA_Holdings.reset_index(inplace=True)
Johnson_Johnson.reset_index(inplace=True)
Laboratory_America_Holding.reset_index(inplace=True)
Lilly.reset_index(inplace=True)
McKesson.reset_index(inplace=True)


#Removing/Renaming columns
Humana = Humana [['Date','Open','Close']]
Humana.rename(columns={'Open': 'Humana_Open'},inplace=True)
Humana.rename(columns={'Close': 'Humana_Close'},inplace=True)

IDEXX_Laboratories = IDEXX_Laboratories [['Date','Open','Close']]
IDEXX_Laboratories.rename(columns={'Open': 'IDEXX_Laboratories_Open'},inplace=True)
IDEXX_Laboratories.rename(columns={'Close': 'IDEXX_Laboratories_Close'},inplace=True)

Illumina = Illumina [['Date','Open','Close']]
Illumina.rename(columns={'Open': 'Illumina_Open'},inplace=True)
Illumina.rename(columns={'Close': 'Illumina_Close'},inplace=True)

Incyte = Incyte [['Date','Open','Close']]
Incyte.rename(columns={'Open': 'Incyte_Open'},inplace=True)
Incyte.rename(columns={'Close': 'Incyte_Close'},inplace=True)

Intuitive_Surgical = Intuitive_Surgical [['Date','Open','Close']]
Intuitive_Surgical.rename(columns={'Open': 'Intuitive_Surgical_Open'},inplace=True)
Intuitive_Surgical.rename(columns={'Close': 'Intuitive_Surgical_Close'},inplace=True)

IQVIA_Holdings = IQVIA_Holdings [['Date','Open','Close']]
IQVIA_Holdings.rename(columns={'Open': 'IQVIA_Holdings_Open'},inplace=True)
IQVIA_Holdings.rename(columns={'Close': 'IQVIA_Holdings_Close'},inplace=True)

Johnson_Johnson = Johnson_Johnson [['Date','Open','Close']]
Johnson_Johnson.rename(columns={'Open': 'Johnson_Johnson_Open'},inplace=True)
Johnson_Johnson.rename(columns={'Close': 'Johnson_Johnson_Close'},inplace=True)

Laboratory_America_Holding = Laboratory_America_Holding [['Date','Open','Close']]
Laboratory_America_Holding.rename(columns={'Open': 'Laboratory_America_Holding_Open'},inplace=True)
Laboratory_America_Holding.rename(columns={'Close': 'Laboratory_America_Holding_Close'},inplace=True)

Lilly = Lilly [['Date','Open','Close']]
Lilly.rename(columns={'Open': 'Lilly_Open'},inplace=True)
Lilly.rename(columns={'Close': 'Lilly_Close'},inplace=True)

McKesson = McKesson [['Date','Open','Close']]
McKesson.rename(columns={'Open': 'McKesson_Open'},inplace=True)
McKesson.rename(columns={'Close': 'McKesson_Close'},inplace=True)


# In[42]:


Medtronic = quandl.get("WIKI/MDT",start_date="2006-10-20", end_date="2013-10-20") #HealthCare
Merck  = quandl.get("WIKI/MRK",start_date="2006-10-20", end_date="2013-10-20")
Mylan_NV  = quandl.get("WIKI/MYL",start_date="2006-10-20", end_date="2013-10-20")
Nektar_Therapeutics  = quandl.get("WIKI/NKTR",start_date="2006-10-20", end_date="2013-10-20")
PerkinElmer  = quandl.get("WIKI/PKI",start_date="2006-10-20", end_date="2013-10-20")
Perrigo  = quandl.get("WIKI/PRGO",start_date="2006-10-20", end_date="2013-10-20")
Pfizer = quandl.get("WIKI/PFE",start_date="2006-10-20", end_date="2013-10-20")
Quest_Diagnostics = quandl.get("WIKI/DGX",start_date="2006-10-20", end_date="2013-10-20")
Regeneron = quandl.get("WIKI/REGN",start_date="2006-10-20", end_date="2013-10-20")
ResMed = quandl.get("WIKI/RMD",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Medtronic.reset_index(inplace=True)
Merck.reset_index(inplace=True)
#Linde  = quandl.get("WIKI/LIN",start_date="2006-10-20", end_date="2013-10-20")
Mylan_NV.reset_index(inplace=True)
Nektar_Therapeutics.reset_index(inplace=True)
PerkinElmer.reset_index(inplace=True)
Perrigo.reset_index(inplace=True)
Pfizer.reset_index(inplace=True)
Quest_Diagnostics.reset_index(inplace=True)
Regeneron.reset_index(inplace=True)
ResMed.reset_index(inplace=True)

#Removing/Renaming columns
Medtronic = Medtronic [['Date','Open','Close']]
Medtronic.rename(columns={'Open': 'Medtronic_Open'},inplace=True)
Medtronic.rename(columns={'Close': 'Medtronic_Close'},inplace=True)

Merck = Merck [['Date','Open','Close']]
Merck.rename(columns={'Open': 'Merck_Open'},inplace=True)
Merck.rename(columns={'Close': 'Merck_Close'},inplace=True)

Mylan_NV = Mylan_NV [['Date','Open','Close']]
Mylan_NV.rename(columns={'Open': 'Mylan_NV_Open'},inplace=True)
Mylan_NV.rename(columns={'Close': 'Mylan_NV_Close'},inplace=True)

Nektar_Therapeutics = Nektar_Therapeutics [['Date','Open','Close']]
Nektar_Therapeutics.rename(columns={'Open': 'Nektar_Therapeutics_Open'},inplace=True)
Nektar_Therapeutics.rename(columns={'Close': 'Nektar_Therapeutics_Close'},inplace=True)

PerkinElmer = PerkinElmer [['Date','Open','Close']]
PerkinElmer.rename(columns={'Open': 'PerkinElmer_Open'},inplace=True)
PerkinElmer.rename(columns={'Close': 'PerkinElmer_Close'},inplace=True)

Perrigo = Perrigo [['Date','Open','Close']]
Perrigo.rename(columns={'Open': 'Perrigo_Open'},inplace=True)
Perrigo.rename(columns={'Close': 'Perrigo_Close'},inplace=True)

Pfizer = Pfizer [['Date','Open','Close']]
Pfizer.rename(columns={'Open': 'Pfizer_Open'},inplace=True)
Pfizer.rename(columns={'Close': 'Pfizer_Close'},inplace=True)

Quest_Diagnostics = Quest_Diagnostics [['Date','Open','Close']]
Quest_Diagnostics.rename(columns={'Open': 'Quest_Diagnostics_Open'},inplace=True)
Quest_Diagnostics.rename(columns={'Close': 'Quest_Diagnostics_Close'},inplace=True)

Regeneron = Regeneron [['Date','Open','Close']]
Regeneron.rename(columns={'Open': 'Regeneron_Open'},inplace=True)
Regeneron.rename(columns={'Close': 'Regeneron_Close'},inplace=True)

ResMed = ResMed [['Date','Open','Close']]
ResMed.rename(columns={'Open': 'ResMed_Open'},inplace=True)
ResMed.rename(columns={'Close': 'ResMed_Close'},inplace=True)


# In[43]:


Stryker = quandl.get("WIKI/SYK",start_date="2006-10-20", end_date="2013-10-20") #HealthCare
Teleflex = quandl.get("WIKI/TFX",start_date="2006-10-20", end_date="2013-10-20")
Thermo_Fisher_Scientific = quandl.get("WIKI/TMO",start_date="2006-10-20", end_date="2013-10-20")
United_Health_Group = quandl.get("WIKI/UNH",start_date="2006-10-20", end_date="2013-10-20")
Universal_Health_Services = quandl.get("WIKI/UHS",start_date="2006-10-20", end_date="2013-10-20")
Varian_Medical_Systems = quandl.get("WIKI/VAR",start_date="2006-10-20", end_date="2013-10-20")
Vertex_Pharmaceuticals = quandl.get("WIKI/VRTX",start_date="2006-10-20", end_date="2013-10-20")
Waters_Corporation  = quandl.get("WIKI/WAT",start_date="2006-10-20", end_date="2013-10-20")
WellCare = quandl.get("WIKI/WCG",start_date="2006-10-20", end_date="2013-10-20")
Zimmer_Biomet_Holdings = quandl.get("WIKI/ZBH",start_date="2006-10-20", end_date="2013-10-20")
Zoetis   = quandl.get("WIKI/ZTS",start_date="2006-10-20", end_date="2013-10-20", collaps= 'monthly')


#indexing
Stryker.reset_index(inplace=True)
Teleflex.reset_index(inplace=True)
#Lamb_Weston_Holdings = quandl.get("WIKI/LW",start_date="2006-10-20", end_date="2013-10-20")
Thermo_Fisher_Scientific.reset_index(inplace=True)
United_Health_Group.reset_index(inplace=True)
Universal_Health_Services.reset_index(inplace=True)
Varian_Medical_Systems.reset_index(inplace=True)
Vertex_Pharmaceuticals.reset_index(inplace=True)
Waters_Corporation.reset_index(inplace=True)
WellCare.reset_index(inplace=True)
Zimmer_Biomet_Holdings.reset_index(inplace=True)
Zoetis.reset_index(inplace=True)


#Removing/Renaming columns
Stryker = Stryker [['Date','Open','Close']]
Stryker.rename(columns={'Open': 'Stryker_Open'},inplace=True)
Stryker.rename(columns={'Close': 'Stryker_Close'},inplace=True)

Teleflex = Teleflex [['Date','Open','Close']]
Teleflex.rename(columns={'Open': 'Teleflex_Open'},inplace=True)
Teleflex.rename(columns={'Close': 'Teleflex_Close'},inplace=True)

Thermo_Fisher_Scientific = Thermo_Fisher_Scientific [['Date','Open','Close']]
Thermo_Fisher_Scientific.rename(columns={'Open': 'Thermo_Fisher_Scientific_Open'},inplace=True)
Thermo_Fisher_Scientific.rename(columns={'Close': 'Thermo_Fisher_Scientific_Close'},inplace=True)

United_Health_Group = United_Health_Group [['Date','Open','Close']]
United_Health_Group.rename(columns={'Open': 'United_Health_Group_Open'},inplace=True)
United_Health_Group.rename(columns={'Close': 'United_Health_Group_Close'},inplace=True)

Universal_Health_Services = Universal_Health_Services [['Date','Open','Close']]
Universal_Health_Services.rename(columns={'Open': 'Universal_Health_Servicesr_Open'},inplace=True)
Universal_Health_Services.rename(columns={'Close': 'Universal_Health_Services_Close'},inplace=True)

Varian_Medical_Systems = Varian_Medical_Systems [['Date','Open','Close']]
Varian_Medical_Systems.rename(columns={'Open': 'Varian_Medical_Systems_Open'},inplace=True)
Varian_Medical_Systems.rename(columns={'Close': 'Varian_Medical_Systems_Close'},inplace=True)

Vertex_Pharmaceuticals = Vertex_Pharmaceuticals [['Date','Open','Close']]
Vertex_Pharmaceuticals.rename(columns={'Open': 'Vertex_Pharmaceuticals_Open'},inplace=True)
Vertex_Pharmaceuticals.rename(columns={'Close': 'Vertex_Pharmaceuticals_Close'},inplace=True)

Waters_Corporation = Waters_Corporation [['Date','Open','Close']]
Waters_Corporation.rename(columns={'Open': 'Waters_Corporation_Open'},inplace=True)
Waters_Corporation.rename(columns={'Close': 'Waters_Corporation_Close'},inplace=True)

WellCare = WellCare [['Date','Open','Close']]
WellCare.rename(columns={'Open': 'WellCare_Open'},inplace=True)
WellCare.rename(columns={'Close': 'WellCare_Close'},inplace=True)

Zimmer_Biomet_Holdings = Zimmer_Biomet_Holdings [['Date','Open','Close']]
Zimmer_Biomet_Holdings.rename(columns={'Open': 'Zimmer_Biomet_Holdings_Open'},inplace=True)
Zimmer_Biomet_Holdings.rename(columns={'Close': 'Zimmer_Biomet_Holdings_Close'},inplace=True)

Zoetis = Zoetis [['Date','Open','Close']]
Zoetis.rename(columns={'Open': 'Zoetis_Open'},inplace=True)
Zoetis.rename(columns={'Close': 'Zoetis_Close'},inplace=True)


# In[44]:


Abbott_Laboratories.shape
AbbVie.shape
ABIOMED.shape
Agilent.shape
Align_Technology.shape
Allergan.shape
AmerisourceBergen_Corp.shape
Amgen.shape
Anthem.shape
Baxter_International.shape
Becton_Dickinson.shape
Biogen.shape
Boston_Scientific.shape
Bristol_Myers_Squibb.shape
Cardinal_Health.shape
Celgene.shape
Centene_Corporation.shape
Cerner.shape
CIGNA.shape
The_Cooper_Companies.shape
CVS_Health.shape
Danaher.shape
DaVita.shape
Dentsply_Sirona.shape
Edwards_Lifesciences.shape
Gilead_Sciences.shape
HCA_Holdings.shape
Henry_Schein.shape
Hologic.shape
Humana.shape
IDEXX_Laboratories.shape
Illumina.shape
Incyte.shape
Intuitive_Surgical.shape
IQVIA_Holdings.shape


# In[45]:


#HealthCare = pd.concat([Abbott_Laboratories[['Date','Abbott_Laboratories_Open','Abbott_Laboratories_Close']], AbbVie[['AbbVie_Open','AbbVie_Close']], ABIOMED[['ABIOMED_Open', 'ABIOMED_Close']], Agilent[['Agilent_Open','Agilent_Close']],Alexion_Pharmaceuticals[['Alexion_Pharmaceuticals_Open','Alexion_Pharmaceuticals_Close']],Align_Technology[['Align_Technology_Open','Align_Technology_Close']],Allergan[['Allergan_Open','Allergan_Close']],AmerisourceBergen_Corp[['AmerisourceBergen_Corp_Open','AmerisourceBergen_Corp_Close']],Amgen[['Amgen_Open','Amgen_Close']],Anthem[['Anthem_Open','Anthem_Close']],Baxter_International[['Baxter_International_Open','Baxter_International_Close']],Becton_Dickinson[['Becton_Dickinson_Open','Becton_Dickinson_Close']],Biogen[['Biogen_Open','Biogen_Close']],Boston_Scientific[['Boston_Scientific_Open','Boston_Scientific_Close']],Bristol_Myers_Squibb[['Bristol_Myers_Squibb_Open','Bristol_Myers_Squibb_Close']],Cardinal_Health[['Cardinal_Health_Open','Cardinal_Health_Close']], Celgene[['Celgene_Open','Celgene_Close']],Centene_Corporation[['Centene_Corporation_Open','Centene_Corporation_Close']],Cerner[['Cerner_Open','Cerner_Close']],CIGNA[['CIGNA_Open','CIGNA_Close']],The_Cooper_Companies[['The_Cooper_Companies_Open','The_Cooper_Companies_Close']],CVS_Health[['CVS_Health_Open','CVS_Health_Close']], Danaher[['Danaher_Open','Danaher_Close']],DaVita[['DaVita_Open','DaVita_Close']],Dentsply_Sirona[['Dentsply_Sirona_Open','Dentsply_Sirona_Close']],Edwards_Lifesciences[['Edwards_Lifesciences_Open','Edwards_Lifesciences_Close']],Gilead_Sciences[['Gilead_Sciences_Open','Gilead_Sciences_Close']],HCA_Holdings[['HCA_Holdings_Open','HCA_Holdings_Close']], Henry_Schein[['Henry_Schein_Open','Henry_Schein_Close']],Hologic[['Hologic_Open','Hologic_Close']], Humana[['Humana_Open', 'Humana_Close']], IDEXX_Laboratories[['IDEXX_Laboratories_Open','IDEXX_Laboratories_Close']],Illumina[['Illumina_Open','Illumina_Close']],Incyte[['Incyte_Open','Incyte_Close']],Intuitive_Surgical[['Intuitive_Surgical_Open','Intuitive_Surgical_Close']],IQVIA_Holdings[['IQVIA_Holdings_Open','IQVIA_Holdings_Close']],Johnson_Johnson[['Johnson_Johnson_Open','Johnson_Johnson_Close']],Laboratory_America_Holding[['Laboratory_America_Holding_Open','Laboratory_America_Holding_Close']],Lilly[['Lilly_Open','Lilly_Close']],McKesson[['McKesson_Open','McKesson_Close']],Medtronic[['Medtronic_Open','Medtronic_Close']],Merck[['Merck_Open','Merck_Close']],Mylan_NV[['Mylan_NV_Open','Mylan_NV_Close']],Nektar_Therapeutics[['Nektar_Therapeutics_Open','Nektar_Therapeutics_Close']], PerkinElmer[['PerkinElmer_Open','PerkinElmer_Close']],Perrigo[['Perrigo_Open','Perrigo_Close']],Pfizer[['Pfizer_Open','Pfizer_Close']],Quest_Diagnostics[['Quest_Diagnostics_Open','Quest_Diagnostics_Close']],Regeneron[['Regeneron_Open','Regeneron_Close']],ResMed[['ResMed_Open','ResMed_Close']], Stryker[['Stryker_Open','Stryker_Close']],Teleflex[['Teleflex_Open','Teleflex_Close']],Thermo_Fisher_Scientific[['Thermo_Fisher_Scientific_Open','Thermo_Fisher_Scientific_Close']],United_Health_Group[['United_Health_Group_Open','United_Health_Group_Close']],Varian_Medical_Systems[['Varian_Medical_Systems_Open','Varian_Medical_Systems_Close']], Vertex_Pharmaceuticals[['Vertex_Pharmaceuticals_Open','Vertex_Pharmaceuticals_Close']],Waters_Corporation[['Waters_Corporation_Open','Waters_Corporation_Close']],WellCare[['WellCare_Open','WellCare_Close']],Zimmer_Biomet_Holdings[['Zimmer_Biomet_Holdings_Open','Zimmer_Biomet_Holdings_Close']],Zoetis[['Zoetis_Open','Zoetis_Close']],The_Travelers_Companies[['The_Travelers_Companies_Open','The_Travelers_Companies_Close']],US_Bancorp[['US_Bancorp_Open','US_Bancorp_Close']], Unum_Group[['Unum_Group_Open','Unum_Group_Close']],Wells_Fargo[['Wells_Fargo_Open','Wells_Fargo_Close']],Willis_Towers_Watson[['Willis_Towers_Watson_Open','Willis_Towers_Watson_Close']],Zions_Bancorp[['Zions_Bancorp_Open','Zions_Bancorp_Close']]],axis=1)
#Universal_Health_Services
HealthCare = (Abbott_Laboratories).merge(AbbVie).merge(ABIOMED).merge(Agilent).merge(Align_Technology).merge(Allergan).merge(AmerisourceBergen_Corp).merge(Amgen).merge(Anthem).merge(Baxter_International).merge(Becton_Dickinson).merge(Biogen).merge(Boston_Scientific).merge(Bristol_Myers_Squibb).merge(Cardinal_Health).merge(Celgene).merge(Centene_Corporation).merge(Cerner).merge(CIGNA).merge(The_Cooper_Companies).merge(CVS_Health).merge(Danaher).merge(DaVita).merge(Dentsply_Sirona).merge(Edwards_Lifesciences).merge(Gilead_Sciences).merge(HCA_Holdings).merge(Henry_Schein).merge(Hologic).merge(Humana).merge(IDEXX_Laboratories).merge(Illumina).merge(Incyte).merge(Intuitive_Surgical).merge(Johnson_Johnson).merge(Laboratory_America_Holding).merge(Lilly).merge(McKesson).merge(Medtronic).merge(Merck).merge(Mylan_NV).merge(Nektar_Therapeutics).merge(PerkinElmer).merge(Perrigo).merge(Pfizer).merge(Quest_Diagnostics).merge(Regeneron).merge(ResMed).merge(Stryker).merge(Teleflex).merge(Thermo_Fisher_Scientific).merge(United_Health_Group).merge(Universal_Health_Services).merge(Varian_Medical_Systems).merge(Vertex_Pharmaceuticals).merge(Waters_Corporation).merge(WellCare).merge(Zimmer_Biomet_Holdings).merge(Zoetis)
HealthCare

#IQVIA_Holdings

#HealthCare.isnull().sum()


# In[46]:


ThreeM = quandl.get("WIKI/MMM",start_date="2006-10-20", end_date="2013-10-20") #Industrials
Alaska_Air_Group= quandl.get("WIKI/ALK",start_date="2006-10-20", end_date="2013-10-20")
Allegion = quandl.get("WIKI/ALLE",start_date="2006-10-20", end_date="2013-10-20")
American_Airlines = quandl.get("WIKI/AAL",start_date="2006-10-20", end_date="2013-10-20")
AMETEK = quandl.get("WIKI/AME",start_date="2006-10-20", end_date="2013-10-20")
A_O_Smith = quandl.get("WIKI/AOS",start_date="2006-10-20", end_date="2013-10-20")
Arconic = quandl.get("WIKI/ARNC",start_date="2006-10-20", end_date="2013-10-20")
Boeing_Company = quandl.get("WIKI/BA",start_date="2006-10-20", end_date="2013-10-20")
C_H_Robinson_Worldwide = quandl.get("WIKI/CHRW",start_date="2006-10-20", end_date="2013-10-20")
Caterpillar = quandl.get("WIKI/CAT",start_date="2006-10-20", end_date="2013-10-20")



#indexing
ThreeM.reset_index(inplace=True)
Alaska_Air_Group.reset_index(inplace=True)
Allegion.reset_index(inplace=True)
American_Airlines.reset_index(inplace=True)
AMETEK.reset_index(inplace=True)
A_O_Smith.reset_index(inplace=True)
Arconic.reset_index(inplace=True)
Boeing_Company.reset_index(inplace=True)
C_H_Robinson_Worldwide.reset_index(inplace=True)
Caterpillar.reset_index(inplace=True)

#Removing/Renaming columns
ThreeM = ThreeM [['Date','Open','Close']]
ThreeM.rename(columns={'Open': 'ThreeM_Open'},inplace=True)
ThreeM.rename(columns={'Close': 'ThreeM_Close'},inplace=True)

Alaska_Air_Group = Alaska_Air_Group [['Date','Open','Close']]
Alaska_Air_Group.rename(columns={'Open': 'Alaska_Air_Group_Open'},inplace=True)
Alaska_Air_Group.rename(columns={'Close': 'Alaska_Air_Group_Close'},inplace=True)

Allegion = Allegion [['Date','Open','Close']]
Allegion.rename(columns={'Open': 'Allegion_Open'},inplace=True)
Allegion.rename(columns={'Close': 'Allegion_Close'},inplace=True)

American_Airlines = American_Airlines [['Date','Open','Close']]
American_Airlines.rename(columns={'Open': 'American_Airlines_Open'},inplace=True)
American_Airlines.rename(columns={'Close': 'American_Airlines_Close'},inplace=True)

AMETEK = AMETEK [['Date','Open','Close']]
AMETEK.rename(columns={'Open': 'AMETEK_Open'},inplace=True)
AMETEK.rename(columns={'Close': 'AMETEK_Close'},inplace=True)

A_O_Smith = A_O_Smith [['Date','Open','Close']]
A_O_Smith.rename(columns={'Open': 'A_O_Smith_Open'},inplace=True)
A_O_Smith.rename(columns={'Close': 'A_O_Smith_Close'},inplace=True)

Arconic = Arconic [['Date','Open','Close']]
Arconic.rename(columns={'Open': 'Arconic_Open'},inplace=True)
Arconic.rename(columns={'Close': 'Arconic_Close'},inplace=True)

Boeing_Company = Boeing_Company [['Date','Open','Close']]
Boeing_Company.rename(columns={'Open': 'Boeing_Company_Open'},inplace=True)
Boeing_Company.rename(columns={'Close': 'Boeing_Company_Close'},inplace=True)

C_H_Robinson_Worldwide = C_H_Robinson_Worldwide [['Date','Open','Close']]
C_H_Robinson_Worldwide.rename(columns={'Open': 'C_H_Robinson_Worldwide_Open'},inplace=True)
C_H_Robinson_Worldwide.rename(columns={'Close': 'C_H_Robinson_Worldwide_Close'},inplace=True)

Caterpillar = Caterpillar [['Date','Open','Close']]
Caterpillar.rename(columns={'Open': 'Caterpillar_Open'},inplace=True)
Caterpillar.rename(columns={'Close': 'Caterpillar_Close'},inplace=True)



# In[47]:


Cintas_Corporation = quandl.get("WIKI/CTAS",start_date="2006-10-20", end_date="2013-10-20") #Industrials
Copart = quandl.get("WIKI/CPRT",start_date="2006-10-20", end_date="2013-10-20")
CSX = quandl.get("WIKI/CSX",start_date="2006-10-20", end_date="2013-10-20")
Cummins = quandl.get("WIKI/CMI",start_date="2006-10-20", end_date="2013-10-20")
Deere = quandl.get("WIKI/DE",start_date="2006-10-20", end_date="2013-10-20")
Delta_Air_Lines = quandl.get("WIKI/DAL",start_date="2006-10-20", end_date="2013-10-20")
Dover = quandl.get("WIKI/DOV",start_date="2006-10-20", end_date="2013-10-20")
Eaton_Corporation = quandl.get("WIKI/ETN",start_date="2006-10-20", end_date="2013-10-20")
Emerson_Electric_Company = quandl.get("WIKI/EMR",start_date="2006-10-20", end_date="2013-10-20")
Equifax = quandl.get("WIKI/EFX",start_date="2006-10-20", end_date="2013-10-20")


#indexing
Cintas_Corporation.reset_index(inplace=True)
Copart.reset_index(inplace=True)
CSX.reset_index(inplace=True)
Cummins.reset_index(inplace=True)
Deere.reset_index(inplace=True)
Delta_Air_Lines.reset_index(inplace=True)
Dover.reset_index(inplace=True)
Eaton_Corporation.reset_index(inplace=True)
Emerson_Electric_Company.reset_index(inplace=True)
Equifax.reset_index(inplace=True)


#Removing/Renaming columns
Cintas_Corporation = Cintas_Corporation [['Date','Open','Close']]
Cintas_Corporation.rename(columns={'Open': 'Cintas_Corporation_Open'},inplace=True)
Cintas_Corporation.rename(columns={'Close': 'Cintas_Corporation_Close'},inplace=True)

Copart = Copart [['Date','Open','Close']]
Copart.rename(columns={'Open': 'Copart_Open'},inplace=True)
Copart.rename(columns={'Close': 'Copart_Close'},inplace=True)

CSX = CSX [['Date','Open','Close']]
CSX.rename(columns={'Open': 'CSX_Open'},inplace=True)
CSX.rename(columns={'Close': 'CSX_Close'},inplace=True)

Cummins = Cummins [['Date','Open','Close']]
Cummins.rename(columns={'Open': 'Cummins_Open'},inplace=True)
Cummins.rename(columns={'Close': 'Cummins_Close'},inplace=True)

Deere = Deere [['Date','Open','Close']]
Deere.rename(columns={'Open': 'Deere_Open'},inplace=True)
Deere.rename(columns={'Close': 'Deere_Close'},inplace=True)

Delta_Air_Lines = Delta_Air_Lines [['Date','Open','Close']]
Delta_Air_Lines.rename(columns={'Open': 'Delta_Air_Lines_Open'},inplace=True)
Delta_Air_Lines.rename(columns={'Close': 'Delta_Air_Lines_Close'},inplace=True)

Dover = Dover [['Date','Open','Close']]
Dover.rename(columns={'Open': 'Dover_Open'},inplace=True)
Dover.rename(columns={'Close': 'Dover_Close'},inplace=True)

Eaton_Corporation = Eaton_Corporation [['Date','Open','Close']]
Eaton_Corporation.rename(columns={'Open': 'Eaton_Corporation_Open'},inplace=True)
Eaton_Corporation.rename(columns={'Close': 'Eaton_Corporation_Close'},inplace=True)

Emerson_Electric_Company = Emerson_Electric_Company [['Date','Open','Close']]
Emerson_Electric_Company.rename(columns={'Open': 'Emerson_Electric_Company_Open'},inplace=True)
Emerson_Electric_Company.rename(columns={'Close': 'Emerson_Electric_Company_Close'},inplace=True)

Equifax = Equifax [['Date','Open','Close']]
Equifax.rename(columns={'Open': 'Equifax_Open'},inplace=True)
Equifax.rename(columns={'Close': 'Equifax_Close'},inplace=True)


# In[48]:


Expeditors = quandl.get("WIKI/EXPD",start_date="2006-10-20", end_date="2013-10-20") #Industrials
Fastenal  = quandl.get("WIKI/FAST",start_date="2006-10-20", end_date="2013-10-20")
FedEx  = quandl.get("WIKI/FDX",start_date="2006-10-20", end_date="2013-10-20")
Flowserve_Corporation = quandl.get("WIKI/FLS",start_date="2006-10-20", end_date="2013-10-20")
Fluor = quandl.get("WIKI/FLR",start_date="2006-10-20", end_date="2013-10-20")
Fortive = quandl.get("WIKI/FTV",start_date="2006-10-20", end_date="2013-10-20")
Fortune_Brands_Home_Security = quandl.get("WIKI/FBHS",start_date="2006-10-20", end_date="2013-10-20")
General_Dynamics = quandl.get("WIKI/GD",start_date="2006-10-20", end_date="2013-10-20")
General_Electric = quandl.get("WIKI/GE",start_date="2006-10-20", end_date="2013-10-20")
Grainger = quandl.get("WIKI/GWW",start_date="2006-10-20", end_date="2013-10-20")


#indexing
Expeditors.reset_index(inplace=True)
Fastenal.reset_index(inplace=True)
FedEx.reset_index(inplace=True)
Flowserve_Corporation.reset_index(inplace=True)
Fluor.reset_index(inplace=True)
Fortive.reset_index(inplace=True)
Fortune_Brands_Home_Security.reset_index(inplace=True)
General_Dynamics.reset_index(inplace=True)
General_Electric.reset_index(inplace=True)
Grainger.reset_index(inplace=True)


#Removing/Renaming columns
Expeditors = Expeditors [['Date','Open','Close']]
Expeditors.rename(columns={'Open': 'Expeditors_Open'},inplace=True)
Expeditors.rename(columns={'Close': 'Expeditors_Close'},inplace=True)

Fastenal = Fastenal [['Date','Open','Close']]
Fastenal.rename(columns={'Open': 'Fastenal_Open'},inplace=True)
Fastenal.rename(columns={'Close': 'Fastenal_Close'},inplace=True)

FedEx = FedEx [['Date','Open','Close']]
FedEx.rename(columns={'Open': 'FedEx_Open'},inplace=True)
FedEx.rename(columns={'Close': 'FedEx_Close'},inplace=True)

Flowserve_Corporation = Flowserve_Corporation [['Date','Open','Close']]
Flowserve_Corporation.rename(columns={'Open': 'Flowserve_Corporation_Open'},inplace=True)
Flowserve_Corporation.rename(columns={'Close': 'Flowserve_Corporation_Close'},inplace=True)

Fluor = Fluor [['Date','Open','Close']]
Fluor.rename(columns={'Open': 'Fluor_Open'},inplace=True)
Fluor.rename(columns={'Close': 'Fluor_Close'},inplace=True)

Fortive = Fortive [['Date','Open','Close']]
Fortive.rename(columns={'Open': 'Fortive_Open'},inplace=True)
Fortive.rename(columns={'Close': 'Fortive_Close'},inplace=True)

Fortune_Brands_Home_Security = Fortune_Brands_Home_Security [['Date','Open','Close']]
Fortune_Brands_Home_Security.rename(columns={'Open': 'Fortune_Brands_Home_Security_Open'},inplace=True)
Fortune_Brands_Home_Security.rename(columns={'Close': 'Fortune_Brands_Home_Security_Close'},inplace=True)

General_Dynamics = General_Dynamics [['Date','Open','Close']]
General_Dynamics.rename(columns={'Open': 'General_Dynamics_Open'},inplace=True)
General_Dynamics.rename(columns={'Close': 'General_Dynamics_Close'},inplace=True)

General_Electric = General_Electric [['Date','Open','Close']]
General_Electric.rename(columns={'Open': 'General_Electric_Open'},inplace=True)
General_Electric.rename(columns={'Close': 'General_Electric_Close'},inplace=True)

Grainger = Grainger [['Date','Open','Close']]
Grainger.rename(columns={'Open': 'Grainger_Open'},inplace=True)
Grainger.rename(columns={'Close': 'Grainger_Close'},inplace=True)


# In[49]:


Harris = quandl.get("WIKI/HRS",start_date="2006-10-20", end_date="2013-10-20") #Industrials
Honeywell_Int = quandl.get("WIKI/HON",start_date="2006-10-20", end_date="2013-10-20")
Huntington_Ingalls_Industries = quandl.get("WIKI/HII",start_date="2006-10-20", end_date="2013-10-20")
IHS_Markit = quandl.get("WIKI/INFO",start_date="2006-10-20", end_date="2013-10-20")
Ingersoll_Rand  = quandl.get("WIKI/IR",start_date="2006-10-20", end_date="2013-10-20")
Jacobs_Engineering_Group = quandl.get("WIKI/JEC",start_date="2006-10-20", end_date="2013-10-20")
J_B_Hunt_Transport = quandl.get("WIKI/JBHT",start_date="2006-10-20", end_date="2013-10-20")
Johnson_Controls_International = quandl.get("WIKI/JCI",start_date="2006-10-20", end_date="2013-10-20")
Kansas_City_Southern = quandl.get("WIKI/KSU",start_date="2006-10-20", end_date="2013-10-20")
L_3_Communications_Holdings = quandl.get("WIKI/LLL",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Harris.reset_index(inplace=True)
Honeywell_Int.reset_index(inplace=True)
Huntington_Ingalls_Industries.reset_index(inplace=True)
L_3_Communications_Holdings.reset_index(inplace=True)
IHS_Markit.reset_index(inplace=True)
Ingersoll_Rand.reset_index(inplace=True)
Jacobs_Engineering_Group.reset_index(inplace=True)
J_B_Hunt_Transport.reset_index(inplace=True)
Johnson_Controls_International.reset_index(inplace=True)
Kansas_City_Southern.reset_index(inplace=True)


#Removing/Renaming columns
Harris = Harris [['Date','Open','Close']]
Harris.rename(columns={'Open': 'Harris_Open'},inplace=True)
Harris.rename(columns={'Close': 'Harris_Close'},inplace=True)

Honeywell_Int = Honeywell_Int [['Date','Open','Close']]
Honeywell_Int.rename(columns={'Open': 'Honeywell_Int_Open'},inplace=True)
Honeywell_Int.rename(columns={'Close': 'Honeywell_Int_Close'},inplace=True)

Huntington_Ingalls_Industries = Huntington_Ingalls_Industries [['Date','Open','Close']]
Huntington_Ingalls_Industries.rename(columns={'Open': 'Huntington_Ingalls_Industries_Open'},inplace=True)
Huntington_Ingalls_Industries.rename(columns={'Close': 'Huntington_Ingalls_Industries_Close'},inplace=True)

L_3_Communications_Holdings = L_3_Communications_Holdings [['Date','Open','Close']]
L_3_Communications_Holdings.rename(columns={'Open': 'L_3_Communications_Holdings_Open'},inplace=True)
L_3_Communications_Holdings.rename(columns={'Close': 'L_3_Communications_Holdings_Close'},inplace=True)

IHS_Markit = IHS_Markit [['Date','Open','Close']]
IHS_Markit.rename(columns={'Open': 'IHS_Markit_Open'},inplace=True)
IHS_Markit.rename(columns={'Close': 'IHS_Markit_Close'},inplace=True)

Ingersoll_Rand = Ingersoll_Rand [['Date','Open','Close']]
Ingersoll_Rand.rename(columns={'Open': 'Ingersoll_Rand_Open'},inplace=True)
Ingersoll_Rand.rename(columns={'Close': 'Ingersoll_Rand_Close'},inplace=True)

Jacobs_Engineering_Group = Jacobs_Engineering_Group [['Date','Open','Close']]
Jacobs_Engineering_Group.rename(columns={'Open': 'Jacobs_Engineering_Group_Open'},inplace=True)
Jacobs_Engineering_Group.rename(columns={'Close': 'Jacobs_Engineering_Group_Close'},inplace=True)

J_B_Hunt_Transport = J_B_Hunt_Transport [['Date','Open','Close']]
J_B_Hunt_Transport.rename(columns={'Open': 'J_B_Hunt_Transport_Open'},inplace=True)
J_B_Hunt_Transport.rename(columns={'Close': 'J_B_Hunt_Transport_Close'},inplace=True)

Johnson_Controls_International = Johnson_Controls_International [['Date','Open','Close']]
Johnson_Controls_International.rename(columns={'Open': 'Johnson_Controls_International_Open'},inplace=True)
Johnson_Controls_International.rename(columns={'Close': 'Johnson_Controls_International_Close'},inplace=True)

Kansas_City_Southern = Kansas_City_Southern [['Date','Open','Close']]
Kansas_City_Southern.rename(columns={'Open': 'Kansas_City_Southern_Open'},inplace=True)
Kansas_City_Southern.rename(columns={'Close': 'Kansas_City_Southern_Close'},inplace=True)



# In[50]:


Lockheed_Martin = quandl.get("WIKI/LMT",start_date="2006-10-20", end_date="2013-10-20") #Industrials
Masco  = quandl.get("WIKI/MAS",start_date="2006-10-20", end_date="2013-10-20")
Nielsen_Holdings = quandl.get("WIKI/NLSN",start_date="2006-10-20", end_date="2013-10-20")
Norfolk_Southern = quandl.get("WIKI/NSC",start_date="2006-10-20", end_date="2013-10-20")
Northrop_Grumman = quandl.get("WIKI/NOC",start_date="2006-10-20", end_date="2013-10-20")
PACCAR  = quandl.get("WIKI/PCAR",start_date="2006-10-20", end_date="2013-10-20")
Parker_Hannifin = quandl.get("WIKI/PH",start_date="2006-10-20", end_date="2013-10-20")
Pentair  = quandl.get("WIKI/PNR",start_date="2006-10-20", end_date="2013-10-20")
Quanta_Services = quandl.get("WIKI/PWR",start_date="2006-10-20", end_date="2013-10-20")
Raytheon  = quandl.get("WIKI/RTN",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Lockheed_Martin.reset_index(inplace=True)
Masco.reset_index(inplace=True)
Nielsen_Holdings.reset_index(inplace=True)
Northrop_Grumman.reset_index(inplace=True)
Norfolk_Southern.reset_index(inplace=True)
PACCAR.reset_index(inplace=True)
Parker_Hannifin.reset_index(inplace=True)
Pentair.reset_index(inplace=True)
Quanta_Services.reset_index(inplace=True)
Raytheon.reset_index(inplace=True)

#Removing/Renaming columns
Lockheed_Martin = Lockheed_Martin [['Date','Open','Close']]
Lockheed_Martin.rename(columns={'Open': 'Lockheed_Martin_Open'},inplace=True)
Lockheed_Martin.rename(columns={'Close': 'Lockheed_Martin_Close'},inplace=True)

Masco = Masco [['Date','Open','Close']]
Masco.rename(columns={'Open': 'Masco_Open'},inplace=True)
Masco.rename(columns={'Close': 'Masco_Close'},inplace=True)

Nielsen_Holdings = Nielsen_Holdings [['Date','Open','Close']]
Nielsen_Holdings.rename(columns={'Open': 'Nielsen_Holdings_Open'},inplace=True)
Nielsen_Holdings.rename(columns={'Close': 'Nielsen_Holdings_Close'},inplace=True)

Quanta_Services = Quanta_Services [['Date','Open','Close']]
Quanta_Services.rename(columns={'Open': 'Quanta_Services_Open'},inplace=True)
Quanta_Services.rename(columns={'Close': 'Quanta_Services_Close'},inplace=True)

Northrop_Grumman = Northrop_Grumman [['Date','Open','Close']]
Northrop_Grumman.rename(columns={'Open': 'Northrop_Grumman_Open'},inplace=True)
Northrop_Grumman.rename(columns={'Close': 'Northrop_Grumman_Close'},inplace=True)

Norfolk_Southern = Norfolk_Southern [['Date','Open','Close']]
Norfolk_Southern.rename(columns={'Open': 'Norfolk_Southern_Open'},inplace=True)
Norfolk_Southern.rename(columns={'Close': 'Norfolk_Southern_Close'},inplace=True)

PACCAR = PACCAR [['Date','Open','Close']]
PACCAR.rename(columns={'Open': 'PACCAR_Open'},inplace=True)
PACCAR.rename(columns={'Close': 'PACCAR_Close'},inplace=True)

Parker_Hannifin = Parker_Hannifin [['Date','Open','Close']]
Parker_Hannifin.rename(columns={'Open': 'Parker_Hannifin_Open'},inplace=True)
Parker_Hannifin.rename(columns={'Close': 'Parker_Hannifin_Close'},inplace=True)

Pentair = Pentair [['Date','Open','Close']]
Pentair.rename(columns={'Open': 'Pentair_Open'},inplace=True)
Pentair.rename(columns={'Close': 'Pentair_Close'},inplace=True)

Raytheon = Raytheon [['Date','Open','Close']]
Raytheon.rename(columns={'Open': 'Raytheon_Open'},inplace=True)
Raytheon.rename(columns={'Close': 'Raytheon_Close'},inplace=True)


# In[51]:


Republic_Services = quandl.get("WIKI/RSG",start_date="2006-10-20", end_date="2013-10-20") #Industrials
Robert_Half_International  = quandl.get("WIKI/RHI",start_date="2006-10-20", end_date="2013-10-20")
Rockwell_Automation = quandl.get("WIKI/ROK",start_date="2006-10-20", end_date="2013-10-20")
Rollins = quandl.get("WIKI/ROL",start_date="2006-10-20", end_date="2013-10-20")
Roper_Technologies = quandl.get("WIKI/ROP",start_date="2006-10-20", end_date="2013-10-20")
Snap_on  = quandl.get("WIKI/SNA",start_date="2006-10-20", end_date="2013-10-20")
Southwest_Airlines  = quandl.get("WIKI/LUV",start_date="2006-10-20", end_date="2013-10-20")
Stanley_Black_Decker  = quandl.get("WIKI/SWK",start_date="2006-10-20", end_date="2013-10-20")
Textron  = quandl.get("WIKI/TXT",start_date="2006-10-20", end_date="2013-10-20")
TransDigm_Group  = quandl.get("WIKI/TDG",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Republic_Services.reset_index(inplace=True)
Robert_Half_International.reset_index(inplace=True)
Rockwell_Automation.reset_index(inplace=True)
Rollins.reset_index(inplace=True)
Roper_Technologies.reset_index(inplace=True)
Snap_on.reset_index(inplace=True)
Southwest_Airlines.reset_index(inplace=True)
Stanley_Black_Decker.reset_index(inplace=True)
Textron.reset_index(inplace=True)
TransDigm_Group.reset_index(inplace=True)

#Removing/Renaming columns
Republic_Services = Republic_Services [['Date','Open','Close']]
Republic_Services.rename(columns={'Open': 'Republic_Services_Open'},inplace=True)
Republic_Services.rename(columns={'Close': 'Republic_Services_Close'},inplace=True)

Robert_Half_International = Robert_Half_International [['Date','Open','Close']]
Robert_Half_International.rename(columns={'Open': 'Robert_Half_International_Open'},inplace=True)
Robert_Half_International.rename(columns={'Close': 'Robert_Half_International_Close'},inplace=True)

Rockwell_Automation = Rockwell_Automation [['Date','Open','Close']]
Rockwell_Automation.rename(columns={'Open': 'Rockwell_Automation_Open'},inplace=True)
Rockwell_Automation.rename(columns={'Close': 'Rockwell_Automation_Close'},inplace=True)

Rollins = Rollins [['Date','Open','Close']]
Rollins.rename(columns={'Open': 'Rollins_Open'},inplace=True)
Rollins.rename(columns={'Close': 'Rollins_Close'},inplace=True)

Roper_Technologies = Roper_Technologies [['Date','Open','Close']]
Roper_Technologies.rename(columns={'Open': 'Roper_Technologies_Open'},inplace=True)
Roper_Technologies.rename(columns={'Close': 'Roper_Technologies_Close'},inplace=True)

Snap_on = Snap_on [['Date','Open','Close']]
Snap_on.rename(columns={'Open': 'Snap_on_Open'},inplace=True)
Snap_on.rename(columns={'Close': 'Snap_on_Close'},inplace=True)

Southwest_Airlines = Southwest_Airlines [['Date','Open','Close']]
Southwest_Airlines.rename(columns={'Open': 'Southwest_Airlines_Open'},inplace=True)
Southwest_Airlines.rename(columns={'Close': 'Southwest_Airlines_Close'},inplace=True)

Stanley_Black_Decker = Stanley_Black_Decker [['Date','Open','Close']]
Stanley_Black_Decker.rename(columns={'Open': 'Stanley_Black_Decker_Open'},inplace=True)
Stanley_Black_Decker.rename(columns={'Close': 'Stanley_Black_Decker_Close'},inplace=True)

Textron = Textron [['Date','Open','Close']]
Textron.rename(columns={'Open': 'Textron_Open'},inplace=True)
Textron.rename(columns={'Close': 'Textron_Close'},inplace=True)



TransDigm_Group = TransDigm_Group [['Date','Open','Close']]
TransDigm_Group.rename(columns={'Open': 'TransDigm_Group_Open'},inplace=True)
TransDigm_Group.rename(columns={'Close': 'TransDigm_Group_Close'},inplace=True)


# In[52]:


Union_Pacific = quandl.get("WIKI/UNP",start_date="2006-10-20", end_date="2013-10-20") #Industrials
United_Continental_Holdings = quandl.get("WIKI/UAL",start_date="2006-10-20", end_date="2013-10-20")
United_Parcel_Service  = quandl.get("WIKI/UPS",start_date="2006-10-20", end_date="2013-10-20")
United_Rentals = quandl.get("WIKI/URI",start_date="2006-10-20", end_date="2013-10-20")
United_Technologies = quandl.get("WIKI/UTX",start_date="2006-10-20", end_date="2013-10-20")
Verisk_Analytics = quandl.get("WIKI/VRSK",start_date="2006-10-20", end_date="2013-10-20")
Waste_Management = quandl.get("WIKI/WM",start_date="2006-10-20", end_date="2013-10-20")
Xylem = quandl.get("WIKI/XYL",start_date="2006-10-20", end_date="2013-10-20")


#indexing
Union_Pacific.reset_index(inplace=True)
United_Continental_Holdings.reset_index(inplace=True)
United_Parcel_Service.reset_index(inplace=True)
United_Rentals.reset_index(inplace=True)
United_Technologies.reset_index(inplace=True)
Verisk_Analytics.reset_index(inplace=True)
Waste_Management.reset_index(inplace=True)
Xylem.reset_index(inplace=True)

#Removing/Renaming columns
Union_Pacific = Union_Pacific [['Date','Open','Close']]
Union_Pacific.rename(columns={'Open': 'Union_Pacific_Open'},inplace=True)
Union_Pacific.rename(columns={'Close': 'Union_Pacific_Close'},inplace=True)

United_Continental_Holdings = United_Continental_Holdings [['Date','Open','Close']]
United_Continental_Holdings.rename(columns={'Open': 'United_Continental_Holdings_Open'},inplace=True)
United_Continental_Holdings.rename(columns={'Close': 'United_Continental_Holdings_Close'},inplace=True)

United_Parcel_Service = United_Parcel_Service [['Date','Open','Close']]
United_Parcel_Service.rename(columns={'Open': 'United_Parcel_Service_Open'},inplace=True)
United_Parcel_Service.rename(columns={'Close': 'United_Parcel_Service_Close'},inplace=True)

United_Rentals = United_Rentals [['Date','Open','Close']]
United_Rentals.rename(columns={'Open': 'United_Rentals_Open'},inplace=True)
United_Rentals.rename(columns={'Close': 'United_Rentals_Close'},inplace=True)

United_Technologies = United_Technologies [['Date','Open','Close']]
United_Technologies.rename(columns={'Open': 'United_Technologies_Open'},inplace=True)
United_Technologies.rename(columns={'Close': 'United_Technologies_Close'},inplace=True)

Verisk_Analytics = Verisk_Analytics [['Date','Open','Close']]
Verisk_Analytics.rename(columns={'Open': 'Verisk_Analytics_Open'},inplace=True)
Verisk_Analytics.rename(columns={'Close': 'Verisk_Analytics_Close'},inplace=True)

Waste_Management = Waste_Management [['Date','Open','Close']]
Waste_Management.rename(columns={'Open': 'Waste_Management_Open'},inplace=True)
Waste_Management.rename(columns={'Close': 'Waste_Management_Close'},inplace=True)

Xylem = Xylem [['Date','Open','Close']]
Xylem.rename(columns={'Open': 'Xylem_Open'},inplace=True)
Xylem.rename(columns={'Close': 'Xylem_Close'},inplace=True)


# In[53]:







Union_Pacific.shape
United_Continental_Holdings.shape
United_Parcel_Service.shape
United_Rentals.shape
United_Technologies.shape
Verisk_Analytics.shape
Waste_Management.shape
Xylem.shape

Republic_Services.shape
Robert_Half_International.shape
Rockwell_Automation.shape
Rollins.shape
Roper_Technologies.shape
Snap_on.shape
Southwest_Airlines.shape
Stanley_Black_Decker.shape
Textron.shape
TransDigm_Group.shape


Lockheed_Martin.shape
Masco.shape
Nielsen_Holdings.shape
Northrop_Grumman.shape
Norfolk_Southern.shape
PACCAR.shape
Parker_Hannifin.shape
Pentair.shape
Quanta_Services.shape
Raytheon.shape

Harris.shape
Honeywell_Int.shape
Huntington_Ingalls_Industries.shape
L_3_Communications_Holdings.shape
IHS_Markit.shape
Ingersoll_Rand.shape
Jacobs_Engineering_Group.shape
J_B_Hunt_Transport.shape
Johnson_Controls_International.shape
Kansas_City_Southern.shape

Expeditors.shape
Fastenal.shape
FedEx.shape
Flowserve_Corporation.shape
Fluor.shape
Fortive.shape
Fortune_Brands_Home_Security.shape
General_Dynamics.shape
General_Electric.shape
Grainger.shape


Cintas_Corporation.shape
Copart.shape
CSX.shape
Cummins.shape
Deere.shape
Delta_Air_Lines.shape
Dover.shape
Eaton_Corporation.shape
Emerson_Electric_Company.shape
Equifax.shape


ThreeM.shape
Alaska_Air_Group.shape
Allegion.shape
American_Airlines.shape
AMETEK.shape
A_O_Smith.shape
Arconic.shape
Boeing_Company.shape
C_H_Robinson_Worldwide.shape
Caterpillar.shape





# In[54]:


#Industrials = pd.concat([Alaska_Air_Group[['Date','Alaska_Air_Group_Open','Alaska_Air_Group_Close']],ThreeM[['ThreeM_Open','ThreeM_Close']], Allegion[['Allegion_Open', 'Allegion_Close']], American_Airlines[['American_Airlines_Open','American_Airlines_Close']],AMETEK[['AMETEK_Open','AMETEK_Close']],A_O_Smith[['A_O_Smith_Open','A_O_Smith_Close']],Arconic[['Arconic_Open','Arconic_Close']],Boeing_Company[['Boeing_Company_Open','Boeing_Company_Close']],C_H_Robinson_Worldwide[['C_H_Robinson_Worldwide_Open','C_H_Robinson_Worldwide_Close']],Caterpillar[['Caterpillar_Open','Caterpillar_Close']],Cintas_Corporation[['Cintas_Corporation_Open','Cintas_Corporation_Close']],Copart[['Copart_Open','Copart_Close']],CSX[['CSX_Open','CSX_Close']],Cummins[['Cummins_Open','Cummins_Close']],Deere[['Deere_Open','Deere_Close']],Delta_Air_Lines[['Delta_Air_Lines_Open','Delta_Air_Lines_Close']], Dover[['Dover_Open','Dover_Close']],Eaton_Corporation[['Eaton_Corporation_Open','Eaton_Corporation_Close']],Emerson_Electric_Company[['Emerson_Electric_Company_Open','Emerson_Electric_Company_Close']],Equifax[['Equifax_Open','Equifax_Close']],Expeditors[['Expeditors_Open','Expeditors_Close']],Fastenal[['Fastenal_Open','Fastenal_Close']], FedEx[['FedEx_Open','FedEx_Close']],Flowserve_Corporation[['Flowserve_Corporation_Open','Flowserve_Corporation_Close']],Fluor[['Fluor_Open','Fluor_Close']],Fortive[['Fortive_Open','Fortive_Close']],Fortune_Brands_Home_Security[['Fortune_Brands_Home_Security_Open','Fortune_Brands_Home_Security_Close']],General_Dynamics[['General_Dynamics_Open','General_Dynamics_Close']], General_Electric[['General_Electric_Open','General_Electric_Close']],Grainger[['Grainger_Open','Grainger_Close']], Harris[['Harris_Open', 'Harris_Close']], Honeywell_Int[['Honeywell_Int_Open','Honeywell_Int_Close']],Huntington_Ingalls_Industries[['Huntington_Ingalls_Industries_Open','Huntington_Ingalls_Industries_Close']],IHS_Markit[['IHS_Markit_Open','IHS_Markit_Close']],Ingersoll_Rand[['Ingersoll_Rand_Open','Ingersoll_Rand_Close']],Jacobs_Engineering_Group[['Jacobs_Engineering_Group_Open','Jacobs_Engineering_Group_Close']],J_B_Hunt_Transport[['J_B_Hunt_Transport_Open','J_B_Hunt_Transport_Close']],Johnson_Controls_International[['Johnson_Controls_International_Open','Johnson_Controls_International_Close']],Kansas_City_Southern[['Kansas_City_Southern_Open','Kansas_City_Southern_Close']],L_3_Communications_Holdings[['L_3_Communications_Holdings_Open','L_3_Communications_Holdings_Close']],Lockheed_Martin[['Lockheed_Martin_Open','Lockheed_Martin_Close']],Masco[['Masco_Open','Masco_Close']],Nielsen_Holdings[['Nielsen_Holdings_Open','Nielsen_Holdings_Close']],Norfolk_Southern[['Norfolk_Southern_Open','Norfolk_Southern_Close']], Northrop_Grumman[['Northrop_Grumman_Open','Northrop_Grumman_Close']],PACCAR[['PACCAR_Open','PACCAR_Close']],Parker_Hannifin[['Parker_Hannifin_Open','Parker_Hannifin_Close']],Pentair[['Pentair_Open','Pentair_Close']],Quanta_Services[['Quanta_Services_Open','Quanta_Services_Close']],Raytheon[['Raytheon_Open','Raytheon_Close']], Republic_Services[['Republic_Services_Open','Republic_Services_Close']],Robert_Half_International[['Robert_Half_International_Open','Robert_Half_International_Close']],Rockwell_Automation[['Rockwell_Automation_Open','Rockwell_Automation_Close']],Rollins[['Rollins_Open','Rollins_Close']],Roper_Technologies[['Roper_Technologies_Open','Roper_Technologies_Close']], Snap_on[['Snap_on_Open','Snap_on_Close']],Southwest_Airlines[['Southwest_Airlines_Open','Southwest_Airlines_Close']],Stanley_Black_Decker[['Stanley_Black_Decker_Open','Stanley_Black_Decker_Close']],Textron[['Textron_Open','Textron_Close']],TransDigm_Group[['TransDigm_Group_Open','TransDigm_Group_Close']],Union_Pacific[['Union_Pacific_Open','Union_Pacific_Close']],United_Continental_Holdings[['United_Continental_Holdings_Open','United_Continental_Holdings_Close']], United_Parcel_Service[['United_Parcel_Service_Open','United_Parcel_Service_Close']],United_Rentals[['United_Rentals_Open','United_Rentals_Close']],United_Technologies[['United_Technologies_Open','United_Technologies_Close']],Verisk_Analytics[['Verisk_Analytics_Open','Verisk_Analytics_Close']],Waste_Management[['Waste_Management_Open','Waste_Management_Close']],Xylem[['Xylem_Open','Xylem_Close']]],axis=1)
Industrials = (Alaska_Air_Group).merge(Allegion).merge(American_Airlines).merge(AMETEK).merge(A_O_Smith).merge(Arconic).merge(Boeing_Company).merge(C_H_Robinson_Worldwide).merge(Caterpillar).merge(Cintas_Corporation).merge(Copart).merge(CSX).merge(Cummins).merge(Deere).merge(Delta_Air_Lines).merge(Dover).merge(Eaton_Corporation).merge(Emerson_Electric_Company).merge(Equifax).merge(Expeditors).merge(Fastenal).merge(FedEx).merge(Flowserve_Corporation).merge(Fluor).merge(Fortune_Brands_Home_Security).merge(General_Dynamics).merge(General_Electric).merge(Grainger).merge(Harris).merge(Honeywell_Int).merge(Huntington_Ingalls_Industries).merge(Ingersoll_Rand).merge(Jacobs_Engineering_Group).merge(J_B_Hunt_Transport).merge(Johnson_Controls_International).merge(Kansas_City_Southern).merge(L_3_Communications_Holdings).merge(Lockheed_Martin).merge(Masco).merge(Nielsen_Holdings).merge(Norfolk_Southern).merge(Northrop_Grumman).merge(PACCAR).merge(Parker_Hannifin).merge(Pentair).merge(Quanta_Services).merge(Raytheon).merge(Republic_Services).merge(Robert_Half_International).merge(Rockwell_Automation).merge(Rollins).merge(Roper_Technologies).merge(Snap_on).merge(Southwest_Airlines).merge(Stanley_Black_Decker).merge(Textron).merge(TransDigm_Group).merge(Union_Pacific).merge(United_Continental_Holdings).merge(United_Parcel_Service).merge(United_Rentals).merge(United_Technologies).merge(Verisk_Analytics).merge(Waste_Management).merge(Xylem)

Industrials

#IHS_Markit
#Fortive
#ThreeM

#Industrials.isnull().sum()


# In[55]:


Accenture = quandl.get("WIKI/ACN",start_date="2006-10-20", end_date="2013-10-20") #Information Technology
Adobe_Systems = quandl.get("WIKI/ADBE",start_date="2006-10-20", end_date="2013-10-20")
Advanced_Micro_Devices = quandl.get("WIKI/AMD",start_date="2006-10-20", end_date="2013-10-20")
Akamai_Technologies = quandl.get("WIKI/AKAM",start_date="2006-10-20", end_date="2013-10-20")
Alliance_Data_Systems = quandl.get("WIKI/ADS",start_date="2006-10-20", end_date="2013-10-20")
Amphenol_Corp = quandl.get("WIKI/APH",start_date="2006-10-20", end_date="2013-10-20")
Analog_Devices = quandl.get("WIKI/ADI",start_date="2006-10-20", end_date="2013-10-20")
ANSYS = quandl.get("WIKI/ANSS",start_date="2006-10-20", end_date="2013-10-20")
Apple = quandl.get("WIKI/AAPL",start_date="2006-10-20", end_date="2013-10-20")
Applied_Materials = quandl.get("WIKI/AMAT",start_date="2006-10-20", end_date="2013-10-20")


#Indexing
Accenture.reset_index(inplace=True)
Adobe_Systems.reset_index(inplace=True)
Advanced_Micro_Devices.reset_index(inplace=True)
Alliance_Data_Systems.reset_index(inplace=True)
Akamai_Technologies.reset_index(inplace=True)
Amphenol_Corp.reset_index(inplace=True)
Analog_Devices.reset_index(inplace=True)
ANSYS.reset_index(inplace=True)
Apple.reset_index(inplace=True)
Applied_Materials.reset_index(inplace=True)


#Removing/Renaming columns
Accenture = Accenture [['Date','Open','Close']]
Accenture.rename(columns={'Open': 'Accenture_Open'},inplace=True)
Accenture.rename(columns={'Close': 'Accenture_Close'},inplace=True)

Adobe_Systems = Adobe_Systems [['Date','Open','Close']]
Adobe_Systems.rename(columns={'Open': 'Adobe_Systems_Open'},inplace=True)
Adobe_Systems.rename(columns={'Close': 'Adobe_Systems_Close'},inplace=True)

Advanced_Micro_Devices = Advanced_Micro_Devices [['Date','Open','Close']]
Advanced_Micro_Devices.rename(columns={'Open': 'Advanced_Micro_Devices_Open'},inplace=True)
Advanced_Micro_Devices.rename(columns={'Close': 'Advanced_Micro_Devices_Close'},inplace=True)

Alliance_Data_Systems = Alliance_Data_Systems [['Date','Open','Close']]
Alliance_Data_Systems.rename(columns={'Open': 'Alliance_Data_Systems_Open'},inplace=True)
Alliance_Data_Systems.rename(columns={'Close': 'Alliance_Data_Systems_Close'},inplace=True)

Akamai_Technologies = Akamai_Technologies [['Date','Open','Close']]
Akamai_Technologies.rename(columns={'Open': 'Akamai_Technologies_Open'},inplace=True)
Akamai_Technologies.rename(columns={'Close': 'Akamai_Technologies_Close'},inplace=True)

Amphenol_Corp = Amphenol_Corp [['Date','Open','Close']]
Amphenol_Corp.rename(columns={'Open': 'Amphenol_Corp_Open'},inplace=True)
Amphenol_Corp.rename(columns={'Close': 'Amphenol_Corp_Close'},inplace=True)

Analog_Devices = Analog_Devices [['Date','Open','Close']]
Analog_Devices.rename(columns={'Open': 'Analog_Devices_Open'},inplace=True)
Analog_Devices.rename(columns={'Close': 'Analog_Devices_Close'},inplace=True)

ANSYS = ANSYS [['Date','Open','Close']]
ANSYS.rename(columns={'Open': 'ANSYS_Open'},inplace=True)
ANSYS.rename(columns={'Close': 'ANSYS_Close'},inplace=True)

Apple = Apple [['Date','Open','Close']]
Apple.rename(columns={'Open': 'Apple_Open'},inplace=True)
Apple.rename(columns={'Close': 'Apple_Close'},inplace=True)

Applied_Materials = Applied_Materials [['Date','Open','Close']]
Applied_Materials.rename(columns={'Open': 'Applied_Materials_Open'},inplace=True)
Applied_Materials.rename(columns={'Close': 'Applied_Materials_Close'},inplace=True)


# In[56]:


#Arista_Networks = quandl.get("WIKI/ANET",start_date="2006-10-20", end_date="2013-10-20") #Information Technology
Autodesk = quandl.get("WIKI/ADSK",start_date="2006-10-20", end_date="2013-10-20")
Automatic_Data_Processing = quandl.get("WIKI/ADP",start_date="2006-10-20", end_date="2013-10-20")
Broadcom = quandl.get("WIKI/AVGO",start_date="2006-10-20", end_date="2013-10-20")
Broadridge_Financial_Solutions = quandl.get("WIKI/BR",start_date="2006-10-20", end_date="2013-10-20")
Cadence_Design_Systems = quandl.get("WIKI/CDNS",start_date="2006-10-20", end_date="2013-10-20")
Cisco_Systems = quandl.get("WIKI/CSCO",start_date="2006-10-20", end_date="2013-10-20")
Citrix_Systems  = quandl.get("WIKI/CTXS",start_date="2006-10-20", end_date="2013-10-20")
Cognizant_Technology_Solutions = quandl.get("WIKI/CTSH",start_date="2006-10-20", end_date="2013-10-20")
Corning = quandl.get("WIKI/GLW",start_date="2006-10-20", end_date="2013-10-20")


#indexing
#Arista_Networks.reset_index(inplace=True)
Autodesk.reset_index(inplace=True)
Automatic_Data_Processing.reset_index(inplace=True)
Broadcom.reset_index(inplace=True)
Broadridge_Financial_Solutions.reset_index(inplace=True)
Cadence_Design_Systems.reset_index(inplace=True)
Cisco_Systems.reset_index(inplace=True)
Citrix_Systems.reset_index(inplace=True)
Cognizant_Technology_Solutions.reset_index(inplace=True)
Corning.reset_index(inplace=True)

#Removing/Renaming columns
#Arista_Networks = Arista_Networks [['Date','Open','Close']]
#Arista_Networks.rename(columns={'Open': 'Arista_Networks_Open'},inplace=True)
#Arista_Networks.rename(columns={'Close': 'Arista_Networks_Close'},inplace=True)

Autodesk = Autodesk [['Date','Open','Close']]
Autodesk.rename(columns={'Open': 'Autodesk_Open'},inplace=True)
Autodesk.rename(columns={'Close': 'Autodesk_Close'},inplace=True)

Automatic_Data_Processing = Automatic_Data_Processing [['Date','Open','Close']]
Automatic_Data_Processing.rename(columns={'Open': 'Automatic_Data_Processing_Open'},inplace=True)
Automatic_Data_Processing.rename(columns={'Close': 'Automatic_Data_Processing_Close'},inplace=True)

Broadcom = Broadcom [['Date','Open','Close']]
Broadcom.rename(columns={'Open': 'Broadcom_Open'},inplace=True)
Broadcom.rename(columns={'Close': 'Broadcom_Close'},inplace=True)

Broadridge_Financial_Solutions = Broadridge_Financial_Solutions [['Date','Open','Close']]
Broadridge_Financial_Solutions.rename(columns={'Open': 'Broadridge_Financial_Solutions_Open'},inplace=True)
Broadridge_Financial_Solutions.rename(columns={'Close': 'Broadridge_Financial_Solutions_Close'},inplace=True)

Cadence_Design_Systems = Cadence_Design_Systems [['Date','Open','Close']]
Cadence_Design_Systems.rename(columns={'Open': 'Cadence_Design_Systems_Open'},inplace=True)
Cadence_Design_Systems.rename(columns={'Close': 'Cadence_Design_Systems_Close'},inplace=True)

Cisco_Systems = Cisco_Systems [['Date','Open','Close']]
Cisco_Systems.rename(columns={'Open': 'Cisco_Systems_Open'},inplace=True)
Cisco_Systems.rename(columns={'Close': 'Cisco_Systems_Close'},inplace=True)

Citrix_Systems = Citrix_Systems [['Date','Open','Close']]
Citrix_Systems.rename(columns={'Open': 'Citrix_Systems_Open'},inplace=True)
Citrix_Systems.rename(columns={'Close': 'Citrix_Systems_Close'},inplace=True)

Cognizant_Technology_Solutions = Cognizant_Technology_Solutions [['Date','Open','Close']]
Cognizant_Technology_Solutions.rename(columns={'Open': 'Cognizant_Technology_Solutions_Open'},inplace=True)
Cognizant_Technology_Solutions.rename(columns={'Close': 'Cognizant_Technology_Solutions_Close'},inplace=True)

Corning = Corning [['Date','Open','Close']]
Corning.rename(columns={'Open': 'Corning_Open'},inplace=True)
Corning.rename(columns={'Close': 'Corning_Close'},inplace=True)



# In[57]:


DXC_Technology = quandl.get("WIKI/DXC",start_date="2006-10-20", end_date="2013-10-20") #Information Technology
F5_Networks = quandl.get("WIKI/FFIV",start_date="2006-10-20", end_date="2013-10-20")
Fidelity_National_Information_Services  = quandl.get("WIKI/FIS",start_date="2006-10-20", end_date="2013-10-20")
Fiserv = quandl.get("WIKI/FISV",start_date="2006-10-20", end_date="2013-10-20")
FleetCor_Technologies = quandl.get("WIKI/FLT",start_date="2006-10-20", end_date="2013-10-20")
FLIR_Systems = quandl.get("WIKI/FLIR",start_date="2006-10-20", end_date="2013-10-20")
Fortinet = quandl.get("WIKI/FTNT",start_date="2006-10-20", end_date="2013-10-20")
Gartner = quandl.get("WIKI/IG",start_date="2006-10-20", end_date="2013-10-20")
Global_Payments = quandl.get("WIKI/GPN",start_date="2006-10-20", end_date="2013-10-20")
Hewlett_Packard_Enterprise = quandl.get("WIKI/HPE",start_date="2006-10-20", end_date="2013-10-20")


#indexing
DXC_Technology.reset_index(inplace=True)
F5_Networks.reset_index(inplace=True)
Fidelity_National_Information_Services.reset_index(inplace=True)
Fiserv.reset_index(inplace=True)
FleetCor_Technologies.reset_index(inplace=True)
FLIR_Systems.reset_index(inplace=True)
Fortinet.reset_index(inplace=True)
Gartner.reset_index(inplace=True)
Global_Payments.reset_index(inplace=True)
Hewlett_Packard_Enterprise.reset_index(inplace=True)

#Removing/Renaming columns
DXC_Technology = DXC_Technology [['Date','Open','Close']]
DXC_Technology.rename(columns={'Open': 'DXC_Technology_Open'},inplace=True)
DXC_Technology.rename(columns={'Close': 'DXC_Technology_Close'},inplace=True)

F5_Networks = F5_Networks [['Date','Open','Close']]
F5_Networks.rename(columns={'Open': 'F5_Networks_Open'},inplace=True)
F5_Networks.rename(columns={'Close': 'F5_Networks_Close'},inplace=True)

Fidelity_National_Information_Services = Fidelity_National_Information_Services [['Date','Open','Close']]
Fidelity_National_Information_Services.rename(columns={'Open': 'Fidelity_National_Information_Services_Open'},inplace=True)
Fidelity_National_Information_Services.rename(columns={'Close': 'Fidelity_National_Information_Services_Close'},inplace=True)

Fiserv = Fiserv [['Date','Open','Close']]
Fiserv.rename(columns={'Open': 'Fiserv_Open'},inplace=True)
Fiserv.rename(columns={'Close': 'Fiserv_Close'},inplace=True)

FleetCor_Technologies = FleetCor_Technologies [['Date','Open','Close']]
FleetCor_Technologies.rename(columns={'Open': 'FleetCor_Technologies_Open'},inplace=True)
FleetCor_Technologies.rename(columns={'Close': 'FleetCor_Technologies_Close'},inplace=True)

FLIR_Systems = FLIR_Systems [['Date','Open','Close']]
FLIR_Systems.rename(columns={'Open': 'FLIR_Systems_Open'},inplace=True)
FLIR_Systems.rename(columns={'Close': 'FLIR_Systems_Close'},inplace=True)

Fortinet = Fortinet [['Date','Open','Close']]
Fortinet.rename(columns={'Open': 'Fortinet_Open'},inplace=True)
Fortinet.rename(columns={'Close': 'Fortinet_Close'},inplace=True)

Gartner = Gartner [['Date','Open','Close']]
Gartner.rename(columns={'Open': 'Gartner_Open'},inplace=True)
Gartner.rename(columns={'Close': 'Gartner_Close'},inplace=True)

Global_Payments = Global_Payments [['Date','Open','Close']]
Global_Payments.rename(columns={'Open': 'Global_Payments_Open'},inplace=True)
Global_Payments.rename(columns={'Close': 'Global_Payments_Close'},inplace=True)

Hewlett_Packard_Enterprise = Hewlett_Packard_Enterprise [['Date','Open','Close']]
Hewlett_Packard_Enterprise.rename(columns={'Open': 'Hewlett_Packard_Enterprise_Open'},inplace=True)
Hewlett_Packard_Enterprise.rename(columns={'Close': 'Hewlett_Packard_Enterprise_Close'},inplace=True)


# In[58]:


HP = quandl.get("WIKI/HPQ",start_date="2006-10-20", end_date="2013-10-20")  #Information Technology
Intel = quandl.get("WIKI/INTC",start_date="2006-10-20", end_date="2013-10-20")
International_Business_Machines = quandl.get("WIKI/IBM",start_date="2006-10-20", end_date="2013-10-20")
Intuit = quandl.get("WIKI/INTU",start_date="2006-10-20", end_date="2013-10-20")
IPG_Photonics = quandl.get("WIKI/IPGP",start_date="2006-10-20", end_date="2013-10-20")
Jack_Henry_Associates  = quandl.get("WIKI/JKHY",start_date="2006-10-20", end_date="2013-10-20")
Juniper_Networks = quandl.get("WIKI/JNPR",start_date="2006-10-20", end_date="2013-10-20")
Keysight_Technologies = quandl.get("WIKI/KEYW",start_date="2006-10-20", end_date="2013-10-20")
KLA_Tencor_Corp  = quandl.get("WIKI/KLAC",start_date="2006-10-20", end_date="2013-10-20")
Lam_Research = quandl.get("WIKI/LRCX",start_date="2006-10-20", end_date="2013-10-20")


#indexing
HP.reset_index(inplace=True)
Intel.reset_index(inplace=True)
International_Business_Machines.reset_index(inplace=True)
Intuit.reset_index(inplace=True)
IPG_Photonics.reset_index(inplace=True)
Juniper_Networks.reset_index(inplace=True)
Keysight_Technologies.reset_index(inplace=True)
Jack_Henry_Associates.reset_index(inplace=True)
KLA_Tencor_Corp.reset_index(inplace=True)
Lam_Research.reset_index(inplace=True)

#Removing/Renaming columns
HP = HP [['Date','Open','Close']]
HP.rename(columns={'Open': 'HP_Open'},inplace=True)
HP.rename(columns={'Close': 'HP_Close'},inplace=True)

Intel = Intel [['Date','Open','Close']]
Intel.rename(columns={'Open': 'Intel_Open'},inplace=True)
Intel.rename(columns={'Close': 'Intel_Close'},inplace=True)

International_Business_Machines = International_Business_Machines [['Date','Open','Close']]
International_Business_Machines.rename(columns={'Open': 'International_Business_Machines_Open'},inplace=True)
International_Business_Machines.rename(columns={'Close': 'International_Business_Machines_Close'},inplace=True)

Intuit = Intuit [['Date','Open','Close']]
Intuit.rename(columns={'Open': 'Intuit_Open'},inplace=True)
Intuit.rename(columns={'Close': 'Intuit_Close'},inplace=True)

IPG_Photonics = IPG_Photonics [['Date','Open','Close']]
IPG_Photonics.rename(columns={'Open': 'IPG_Photonics_Open'},inplace=True)
IPG_Photonics.rename(columns={'Close': 'IPG_Photonics_Close'},inplace=True)

Juniper_Networks = Juniper_Networks [['Date','Open','Close']]
Juniper_Networks.rename(columns={'Open': 'Juniper_Networks_Open'},inplace=True)
Juniper_Networks.rename(columns={'Close': 'Juniper_Networks_Close'},inplace=True)

Keysight_Technologies = Keysight_Technologies [['Date','Open','Close']]
Keysight_Technologies.rename(columns={'Open': 'Keysight_Technologies_Open'},inplace=True)
Keysight_Technologies.rename(columns={'Close': 'Keysight_Technologies_Close'},inplace=True)

Jack_Henry_Associates = Jack_Henry_Associates [['Date','Open','Close']]
Jack_Henry_Associates.rename(columns={'Open': 'Jack_Henry_Associates_Open'},inplace=True)
Jack_Henry_Associates.rename(columns={'Close': 'Jack_Henry_Associates_Close'},inplace=True)

KLA_Tencor_Corp = KLA_Tencor_Corp [['Date','Open','Close']]
KLA_Tencor_Corp.rename(columns={'Open': 'KLA_Tencor_Corp_Open'},inplace=True)
KLA_Tencor_Corp.rename(columns={'Close': 'KLA_Tencor_Corp_Close'},inplace=True)

Lam_Research = Lam_Research [['Date','Open','Close']]
Lam_Research.rename(columns={'Open': 'Lam_Research_Open'},inplace=True)
Lam_Research.rename(columns={'Close': 'Lam_Research_Close'},inplace=True)




# In[59]:


Mastercard = quandl.get("WIKI/MA",start_date="2006-10-20", end_date="2013-10-20") #Information Technology
Microchip_Technology = quandl.get("WIKI/MCHP",start_date="2006-10-20", end_date="2013-10-20")
Micron_Technology = quandl.get("WIKI/MU",start_date="2006-10-20", end_date="2013-10-20")
Microsoft = quandl.get("WIKI/MSFT",start_date="2006-10-20", end_date="2013-10-20")
Motorola_Solutions = quandl.get("WIKI/MSI",start_date="2006-10-20", end_date="2013-10-20")
NetApp = quandl.get("WIKI/NTAP",start_date="2006-10-20", end_date="2013-10-20")
Nvidia_Corporation = quandl.get("WIKI/NVDA",start_date="2006-10-20", end_date="2013-10-20")
Oracle = quandl.get("WIKI/ORCL",start_date="2006-10-20", end_date="2013-10-20")
Paychex = quandl.get("WIKI/PAYX",start_date="2006-10-20", end_date="2013-10-20")
PayPal = quandl.get("WIKI/PYPL",start_date="2006-10-20", end_date="2013-10-20")


#indexing
Mastercard.reset_index(inplace=True)
Microchip_Technology.reset_index(inplace=True)
Microsoft.reset_index(inplace=True)
Micron_Technology.reset_index(inplace=True)
Motorola_Solutions.reset_index(inplace=True)
NetApp.reset_index(inplace=True)
Nvidia_Corporation.reset_index(inplace=True)
Oracle.reset_index(inplace=True)
Paychex.reset_index(inplace=True)
PayPal.reset_index(inplace=True)


#Removing/Renaming columns
Mastercard = Mastercard [['Date','Open','Close']]
Mastercard.rename(columns={'Open': 'Mastercard_Open'},inplace=True)
Mastercard.rename(columns={'Close': 'Mastercard_Close'},inplace=True)

Microchip_Technology = Microchip_Technology [['Date','Open','Close']]
Microchip_Technology.rename(columns={'Open': 'Microchip_Technology_Open'},inplace=True)
Microchip_Technology.rename(columns={'Close': 'Microchip_Technology_Close'},inplace=True)

Microsoft = Microsoft [['Date','Open','Close']]
Microsoft.rename(columns={'Open': 'Microsoft_Open'},inplace=True)
Microsoft.rename(columns={'Close': 'Microsoft_Close'},inplace=True)

Micron_Technology = Micron_Technology [['Date','Open','Close']]
Micron_Technology.rename(columns={'Open': 'Micron_Technology_Open'},inplace=True)
Micron_Technology.rename(columns={'Close': 'Micron_Technology_Close'},inplace=True)

Motorola_Solutions = Motorola_Solutions [['Date','Open','Close']]
Motorola_Solutions.rename(columns={'Open': 'Motorola_Solutions_Open'},inplace=True)
Motorola_Solutions.rename(columns={'Close': 'Motorola_Solutions_Close'},inplace=True)

NetApp = NetApp [['Date','Open','Close']]
NetApp.rename(columns={'Open': 'NetApp_Open'},inplace=True)
NetApp.rename(columns={'Close': 'NetApp_Close'},inplace=True)

Nvidia_Corporation = Nvidia_Corporation [['Date','Open','Close']]
Nvidia_Corporation.rename(columns={'Open': 'Nvidia_Corporation_Open'},inplace=True)
Nvidia_Corporation.rename(columns={'Close': 'Nvidia_Corporation_Close'},inplace=True)

Oracle = Oracle [['Date','Open','Close']]
Oracle.rename(columns={'Open': 'Oracle_Open'},inplace=True)
Oracle.rename(columns={'Close': 'Oracle_Close'},inplace=True)

Paychex = Paychex [['Date','Open','Close']]
Paychex.rename(columns={'Open': 'Paychex_Open'},inplace=True)
Paychex.rename(columns={'Close': 'Paychex_Close'},inplace=True)

PayPal = PayPal [['Date','Open','Close']]
PayPal.rename(columns={'Open': 'PayPal_Open'},inplace=True)
PayPal.rename(columns={'Close': 'PayPal_Close'},inplace=True)



# In[60]:


QUALCOMM = quandl.get("WIKI/QCOM",start_date="2006-10-20", end_date="2013-10-20") #Information Technology
Qorvo = quandl.get("WIKI/QRVO",start_date="2006-10-20", end_date="2013-10-20")
Red_Hat  = quandl.get("WIKI/RHT",start_date="2006-10-20", end_date="2013-10-20")
Salesforce = quandl.get("WIKI/CRM",start_date="2006-10-20", end_date="2013-10-20")
Seagate_Technology = quandl.get("WIKI/STX",start_date="2006-10-20", end_date="2013-10-20")
Skyworks_Solution = quandl.get("WIKI/SWKS",start_date="2006-10-20", end_date="2013-10-20")
Symantec = quandl.get("WIKI/SYMC",start_date="2006-10-20", end_date="2013-10-20")
Synopsys = quandl.get("WIKI/SNPS",start_date="2006-10-20", end_date="2013-10-20")
TE_Connectivity = quandl.get("WIKI/TEL",start_date="2006-10-20", end_date="2013-10-20")
Texas_Instruments = quandl.get("WIKI/TXN",start_date="2006-10-20", end_date="2013-10-20")


#indexing
QUALCOMM.reset_index(inplace=True)
Qorvo.reset_index(inplace=True)
Red_Hat.reset_index(inplace=True)
Synopsys.reset_index(inplace=True)
Salesforce.reset_index(inplace=True)
Seagate_Technology.reset_index(inplace=True)
Skyworks_Solution.reset_index(inplace=True)
Symantec.reset_index(inplace=True)
TE_Connectivity.reset_index(inplace=True)
Texas_Instruments.reset_index(inplace=True)


#Removing/Renaming columns
QUALCOMM = QUALCOMM [['Date','Open','Close']]
QUALCOMM.rename(columns={'Open': 'QUALCOMM_Open'},inplace=True)
QUALCOMM.rename(columns={'Close': 'QUALCOMM_Close'},inplace=True)

Qorvo = Qorvo [['Date','Open','Close']]
Qorvo.rename(columns={'Open': 'Qorvo_Open'},inplace=True)
Qorvo.rename(columns={'Close': 'Qorvo_Close'},inplace=True)

Red_Hat = Red_Hat [['Date','Open','Close']]
Red_Hat.rename(columns={'Open': 'Red_Hat_Open'},inplace=True)
Red_Hat.rename(columns={'Close': 'Red_Hat_Close'},inplace=True)

Synopsys = Synopsys [['Date','Open','Close']]
Synopsys.rename(columns={'Open': 'Synopsys_Open'},inplace=True)
Synopsys.rename(columns={'Close': 'Synopsys_Close'},inplace=True)

Salesforce = Salesforce [['Date','Open','Close']]
Salesforce.rename(columns={'Open': 'Salesforce_Open'},inplace=True)
Salesforce.rename(columns={'Close': 'Salesforce_Close'},inplace=True)

Seagate_Technology = Seagate_Technology [['Date','Open','Close']]
Seagate_Technology.rename(columns={'Open': 'Seagate_Technology_Open'},inplace=True)
Seagate_Technology.rename(columns={'Close': 'Seagate_Technology_Close'},inplace=True)

Skyworks_Solution = Skyworks_Solution [['Date','Open','Close']]
Skyworks_Solution.rename(columns={'Open': 'Skyworks_Solution_Open'},inplace=True)
Skyworks_Solution.rename(columns={'Close': 'Skyworks_Solution_Close'},inplace=True)

Symantec = Symantec [['Date','Open','Close']]
Symantec.rename(columns={'Open': 'Symantec_Open'},inplace=True)
Symantec.rename(columns={'Close': 'Symantec_Close'},inplace=True)

TE_Connectivity = TE_Connectivity [['Date','Open','Close']]
TE_Connectivity.rename(columns={'Open': 'TE_Connectivity_Open'},inplace=True)
TE_Connectivity.rename(columns={'Close': 'TE_Connectivity_Close'},inplace=True)

Texas_Instruments = Texas_Instruments [['Date','Open','Close']]
Texas_Instruments.rename(columns={'Open': 'Texas_Instruments_Open'},inplace=True)
Texas_Instruments.rename(columns={'Close': 'Texas_Instruments_Close'},inplace=True)


# In[61]:


Total_System_Services = quandl.get("WIKI/TSS",start_date="2006-10-20", end_date="2013-10-20") #Information Technology
Verisign  = quandl.get("WIKI/VRSN",start_date="2006-10-20", end_date="2013-10-20")
Visa = quandl.get("WIKI/V",start_date="2006-10-20", end_date="2013-10-20")
Western_Digital  = quandl.get("WIKI/WDC",start_date="2006-10-20", end_date="2013-10-20")
Western_Union  = quandl.get("WIKI/WU",start_date="2006-10-20", end_date="2013-10-20")
Xerox = quandl.get("WIKI/XRX",start_date="2006-10-20", end_date="2013-10-20")
Xilinx = quandl.get("WIKI/XLNX",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Total_System_Services.reset_index(inplace=True)
Verisign.reset_index(inplace=True)
Visa.reset_index(inplace=True)
Western_Digital.reset_index(inplace=True)
Xerox.reset_index(inplace=True)
Western_Union.reset_index(inplace=True)
Xilinx.reset_index(inplace=True)

#Removing/Renaming columns
Total_System_Services = Total_System_Services [['Date','Open','Close']]
Total_System_Services.rename(columns={'Open': 'Total_System_Services_Open'},inplace=True)
Total_System_Services.rename(columns={'Close': 'Total_System_Services_Close'},inplace=True)

Verisign = Verisign [['Date','Open','Close']]
Verisign.rename(columns={'Open': 'Verisign_Open'},inplace=True)
Verisign.rename(columns={'Close': 'Verisign_Close'},inplace=True)

Visa = Visa [['Date','Open','Close']]
Visa.rename(columns={'Open': 'Visa_Open'},inplace=True)
Visa.rename(columns={'Close': 'Visa_Close'},inplace=True)

Western_Digital = Western_Digital [['Date','Open','Close']]
Western_Digital.rename(columns={'Open': 'Western_Digital_Open'},inplace=True)
Western_Digital.rename(columns={'Close': 'Western_Digital_Close'},inplace=True)

Xerox = Xerox [['Date','Open','Close']]
Xerox.rename(columns={'Open': 'Xerox_Open'},inplace=True)
Xerox.rename(columns={'Close': 'Xerox_Close'},inplace=True)

Western_Union = Western_Union [['Date','Open','Close']]
Western_Union.rename(columns={'Open': 'Western_Union_Open'},inplace=True)
Western_Union.rename(columns={'Close': 'Western_Union_Close'},inplace=True)

Xilinx = Xilinx [['Date','Open','Close']]
Xilinx.rename(columns={'Open': 'Xilinx_Open'},inplace=True)
Xilinx.rename(columns={'Close': 'Xilinx_Close'},inplace=True)


# In[62]:


Accenture.shape
Adobe_Systems.shape
Advanced_Micro_Devices.shape
Alliance_Data_Systems.shape
Akamai_Technologies.shape
Amphenol_Corp.shape
Analog_Devices.shape
ANSYS.shape
Apple.shape
Applied_Materials.shape
Autodesk.shape
Automatic_Data_Processing.shape
Broadcom.shape
Broadridge_Financial_Solutions.shape
Cadence_Design_Systems.shape
Cisco_Systems.shape
Citrix_Systems.shape
Cognizant_Technology_Solutions.shape
Corning.shape

F5_Networks.shape
Fidelity_National_Information_Services.shape
Fiserv.shape
FleetCor_Technologies.shape
FLIR_Systems.shape
Fortinet.shape
Gartner.shape
Global_Payments.shape
Hewlett_Packard_Enterprise.shape

HP.shape
Intel.shape
International_Business_Machines.shape
Intuit.shape
IPG_Photonics.shape
Juniper_Networks.shape
Keysight_Technologies.shape
Jack_Henry_Associates.shape
KLA_Tencor_Corp.shape
Lam_Research.shape

Mastercard.shape
Microchip_Technology.shape
Microsoft.shape
Micron_Technology.shape
Motorola_Solutions.shape
NetApp.shape
Nvidia_Corporation.shape
Oracle.shape
Paychex.shape
PayPal.shape

QUALCOMM.shape
Qorvo.shape
Red_Hat.shape
#Synopsys.shape
#Salesforce.shape
#Seagate_Technology.shape
#Skyworks_Solution.shape
#Symantec.shape
#TE_Connectivity.shape
#Texas_Instruments.shape




# In[63]:


#Information_Technology = pd.concat([Accenture[['Date','Accenture_Open','Accenture_Close']], Adobe_Systems[['Adobe_Systems_Open','Adobe_Systems_Close']], Advanced_Micro_Devices[['Advanced_Micro_Devices_Open', 'Advanced_Micro_Devices_Close']], Akamai_Technologies[['Akamai_Technologies_Open','Akamai_Technologies_Close']],Alliance_Data_Systems[['Alliance_Data_Systems_Open','Alliance_Data_Systems_Close']],Amphenol_Corp[['Amphenol_Corp_Open','Amphenol_Corp_Close']],Analog_Devices[['Analog_Devices_Open','Analog_Devices_Close']],ANSYS[['ANSYS_Open','ANSYS_Close']],Apple[['Apple_Open','Apple_Close']],Applied_Materials[['Applied_Materials_Open','Applied_Materials_Close']],Autodesk[['Autodesk_Open','Autodesk_Close']],Automatic_Data_Processing[['Automatic_Data_Processing_Open','Automatic_Data_Processing_Close']],Broadcom[['Broadcom_Open','Broadcom_Close']],Broadridge_Financial_Solutions[['Broadridge_Financial_Solutions_Open','Broadridge_Financial_Solutions_Close']],Cadence_Design_Systems[['Cadence_Design_Systems_Open','Cadence_Design_Systems_Close']],Cisco_Systems[['Cisco_Systems_Open','Cisco_Systems_Close']], Citrix_Systems[['Citrix_Systems_Open','Citrix_Systems_Close']],Cognizant_Technology_Solutions[['Cognizant_Technology_Solutions_Open','Cognizant_Technology_Solutions_Close']],Corning[['Corning_Open','Corning_Close']],DXC_Technology[['DXC_Technology_Open','DXC_Technology_Close']],F5_Networks[['F5_Networks_Open','F5_Networks_Close']],Fidelity_National_Information_Services[['Fidelity_National_Information_Services_Open','Fidelity_National_Information_Services_Close']], Fiserv[['Fiserv_Open','Fiserv_Close']],FleetCor_Technologies[['FleetCor_Technologies_Open','FleetCor_Technologies_Close']],FLIR_Systems[['FLIR_Systems_Open','FLIR_Systems_Close']],Fortinet[['Fortinet_Open','Fortinet_Close']],Gartner[['Gartner_Open','Gartner_Close']],Global_Payments[['Global_Payments_Open','Global_Payments_Close']], Hewlett_Packard_Enterprise[['Hewlett_Packard_Enterprise_Open','Hewlett_Packard_Enterprise_Close']],HP[['HP_Open','HP_Close']], Intel[['Intel_Open', 'Intel_Close']], International_Business_Machines[['International_Business_Machines_Open','International_Business_Machines_Close']],Intuit[['Intuit_Open','Intuit_Close']],IPG_Photonics[['IPG_Photonics_Open','IPG_Photonics_Close']],Jack_Henry_Associates[['Jack_Henry_Associates_Open','Jack_Henry_Associates_Close']],Juniper_Networks[['Juniper_Networks_Open','Juniper_Networks_Close']],Keysight_Technologies[['Keysight_Technologies_Open','Keysight_Technologies_Close']],KLA_Tencor_Corp[['KLA_Tencor_Corp_Open','KLA_Tencor_Corp_Close']],Lam_Research[['Lam_Research_Open','Lam_Research_Close']],Mastercard[['Mastercard_Open','Mastercard_Close']],Microchip_Technology[['Microchip_Technology_Open','Microchip_Technology_Close']],Micron_Technology[['Micron_Technology_Open','Micron_Technology_Close']],Microsoft[['Microsoft_Open','Microsoft_Close']],Motorola_Solutions[['Motorola_Solutions_Open','Motorola_Solutions_Close']], NetApp[['NetApp_Open','NetApp_Close']],Nvidia_Corporation[['Nvidia_Corporation_Open','Nvidia_Corporation_Close']],Oracle[['Oracle_Open','Oracle_Close']],Paychex[['Paychex_Open','Paychex_Close']],PayPal[['PayPal_Open','PayPal_Close']],QUALCOMM[['QUALCOMM_Open','QUALCOMM_Close']], Qorvo[['Qorvo_Open','Qorvo_Close']],Red_Hat[['Red_Hat_Open','Red_Hat_Close']],Salesforce[['Salesforce_Open','Salesforce_Close']],Seagate_Technology[['Seagate_Technology_Open','Seagate_Technology_Close']],Skyworks_Solution[['Skyworks_Solution_Open','Skyworks_Solution_Close']], Symantec[['Symantec_Open','Symantec_Close']],Synopsys[['Synopsys_Open','Synopsys_Close']],TE_Connectivity[['TE_Connectivity_Open','TE_Connectivity_Close']],Texas_Instruments[['Texas_Instruments_Open','Texas_Instruments_Close']],Total_System_Services[['Total_System_Services_Open','Total_System_Services_Close']],Verisign[['Verisign_Open','Verisign_Close']],Visa[['Visa_Open','Visa_Close']], Western_Digital[['Western_Digital_Open','Western_Digital_Close']],Western_Union[['Western_Union_Open','Western_Union_Close']],Xerox[['Xerox_Open','Xerox_Close']],Xilinx[['Xilinx_Open','Xilinx_Close']]],axis=1)

Information_Technology = (Accenture).merge(Adobe_Systems).merge(Advanced_Micro_Devices).merge(Akamai_Technologies).merge(Alliance_Data_Systems).merge(Amphenol_Corp).merge(Analog_Devices).merge(ANSYS).merge(Apple).merge(Applied_Materials).merge(Autodesk).merge(Automatic_Data_Processing).merge(Broadcom).merge(Broadridge_Financial_Solutions).merge(Cadence_Design_Systems).merge(Cisco_Systems).merge(Citrix_Systems).merge(Cognizant_Technology_Solutions).merge(Corning).merge(F5_Networks).merge(Fidelity_National_Information_Services).merge(Fiserv).merge(FleetCor_Technologies).merge(FLIR_Systems).merge(Fortinet).merge(Global_Payments).merge(HP).merge(Intel).merge(International_Business_Machines).merge(Intuit).merge(IPG_Photonics).merge(Jack_Henry_Associates).merge(Juniper_Networks).merge(Keysight_Technologies).merge(KLA_Tencor_Corp).merge(Lam_Research).merge(Mastercard).merge(Microchip_Technology).merge(Micron_Technology).merge(Microsoft).merge(Motorola_Solutions).merge(NetApp).merge(Nvidia_Corporation).merge(Oracle).merge(Paychex).merge(QUALCOMM).merge(Red_Hat).merge(Salesforce).merge(Rollins).merge(Seagate_Technology).merge(Skyworks_Solution).merge(Symantec).merge(Synopsys).merge(TE_Connectivity).merge(Texas_Instruments).merge(Total_System_Services).merge(Verisign).merge(Visa).merge(Western_Digital).merge(Western_Union).merge(Xerox).merge(Xilinx)

Information_Technology

#DXC_Technology
#Gartner
#Hewlett_Packard_Enterprise
#PayPal
#Qorvo



#Information_Technology.isnull().sum()


# In[64]:


Air_Products_Chemicals = quandl.get("WIKI/APD",start_date="2006-10-20", end_date="2013-10-20") #Materials
Albemarle_Corp = quandl.get("WIKI/ALB",start_date="2006-10-20", end_date="2013-10-20")
Avery_Dennison = quandl.get("WIKI/AVY",start_date="2006-10-20", end_date="2013-10-20")
Ball= quandl.get("WIKI/BLL",start_date="2006-10-20", end_date="2013-10-20")
Celanese = quandl.get("WIKI/CE",start_date="2006-10-20", end_date="2013-10-20")
CF_Industries_Holdings = quandl.get("WIKI/CF",start_date="2006-10-20", end_date="2013-10-20")
DowDuPont = quandl.get("WIKI/DWDP",start_date="2006-10-20", end_date="2013-10-20")
Eastman_Chemical = quandl.get("WIKI/EMN",start_date="2006-10-20", end_date="2013-10-20")
Ecolab = quandl.get("WIKI/ECL",start_date="2006-10-20", end_date="2013-10-20")
FMC_Corporation = quandl.get("WIKI/FMC",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Air_Products_Chemicals.reset_index(inplace=True)
Albemarle_Corp.reset_index(inplace=True)
Avery_Dennison.reset_index(inplace=True)
Ball.reset_index(inplace=True)
Celanese.reset_index(inplace=True)
CF_Industries_Holdings.reset_index(inplace=True)
DowDuPont.reset_index(inplace=True)
Eastman_Chemical.reset_index(inplace=True)
FMC_Corporation.reset_index(inplace=True)
Ecolab.reset_index(inplace=True)

#Removing/Renaming columns
Air_Products_Chemicals = Air_Products_Chemicals [['Date','Open','Close']]
Air_Products_Chemicals.rename(columns={'Open': 'Air_Products_Chemicals_Open'},inplace=True)
Air_Products_Chemicals.rename(columns={'Close': 'Air_Products_Chemicals_Close'},inplace=True)

Albemarle_Corp = Albemarle_Corp [['Date','Open','Close']]
Albemarle_Corp.rename(columns={'Open': 'Albemarle_Corp_Open'},inplace=True)
Albemarle_Corp.rename(columns={'Close': 'Albemarle_Corp_Close'},inplace=True)

Avery_Dennison = Avery_Dennison [['Date','Open','Close']]
Avery_Dennison.rename(columns={'Open': 'Avery_Dennison_Open'},inplace=True)
Avery_Dennison.rename(columns={'Close': 'Avery_Dennison_Close'},inplace=True)

Ball = Ball [['Date','Open','Close']]
Ball.rename(columns={'Open': 'Ball_Open'},inplace=True)
Ball.rename(columns={'Close': 'Ball_Close'},inplace=True)


Celanese = Celanese [['Date','Open','Close']]
Celanese.rename(columns={'Open': 'Celanese_Open'},inplace=True)
Celanese.rename(columns={'Close': 'Celanese_Close'},inplace=True)

CF_Industries_Holdings = CF_Industries_Holdings [['Date','Open','Close']]
CF_Industries_Holdings.rename(columns={'Open': 'CF_Industries_Holdings_Open'},inplace=True)
CF_Industries_Holdings.rename(columns={'Close': 'CF_Industries_Holdings_Close'},inplace=True)

DowDuPont = DowDuPont [['Date','Open','Close']]
DowDuPont.rename(columns={'Open': 'DowDuPont_Open'},inplace=True)
DowDuPont.rename(columns={'Close': 'DowDuPont_Close'},inplace=True)

Eastman_Chemical = Eastman_Chemical [['Date','Open','Close']]
Eastman_Chemical.rename(columns={'Open': 'Eastman_Chemical_Open'},inplace=True)
Eastman_Chemical.rename(columns={'Close': 'Eastman_Chemical_Close'},inplace=True)

FMC_Corporation = FMC_Corporation [['Date','Open','Close']]
FMC_Corporation.rename(columns={'Open': 'FMC_Corporation_Open'},inplace=True)
FMC_Corporation.rename(columns={'Close': 'FMC_Corporation_Close'},inplace=True)

Ecolab = Ecolab [['Date','Open','Close']]
Ecolab.rename(columns={'Open': 'Ecolab_Open'},inplace=True)
Ecolab.rename(columns={'Close': 'Ecolab_Close'},inplace=True)


# In[65]:


Freeport_McMoRan  = quandl.get("WIKI/FCX",start_date="2006-10-20", end_date="2013-10-20") #Materials
International_Paper = quandl.get("WIKI/IP",start_date="2006-10-20", end_date="2013-10-20")
#Intl_Flavors_Fragrances = quandl.get("WIKI/IFF",start_date="2006-10-20", end_date="2013-10-20")
#Linde  = quandl.get("WIKI/LIN",start_date="2006-10-20", end_date="2013-10-20")
LyondellBasell = quandl.get("WIKI/LYB",start_date="2006-10-20", end_date="2013-10-20")
Martin_Marietta_Materials = quandl.get("WIKI/MLM",start_date="2006-10-20", end_date="2013-10-20")
The_Mosaic_Company = quandl.get("WIKI/MOS",start_date="2006-10-20", end_date="2013-10-20")
Newmont_Mining_Corporation = quandl.get("WIKI/NEM",start_date="2006-10-20", end_date="2013-10-20")
Nucor = quandl.get("WIKI/NUE",start_date="2006-10-20", end_date="2013-10-20")
Packaging_Corporation_of_America  = quandl.get("WIKI/PKG",start_date="2006-10-20", end_date="2013-10-20")



#indexing
Freeport_McMoRan.reset_index(inplace=True)
International_Paper.reset_index(inplace=True)
#Intl_Flavors_Fragrances.reset_index(inplace=True)
#Linde.reset_index(inplace=True)
LyondellBasell.reset_index(inplace=True)
Martin_Marietta_Materials.reset_index(inplace=True)
The_Mosaic_Company.reset_index(inplace=True)
Newmont_Mining_Corporation.reset_index(inplace=True)
Nucor.reset_index(inplace=True)
Packaging_Corporation_of_America.reset_index(inplace=True)


#Removing/Renaming columns
Freeport_McMoRan = Freeport_McMoRan [['Date','Open','Close']]
Freeport_McMoRan.rename(columns={'Open': 'Freeport_McMoRan_Open'},inplace=True)
Freeport_McMoRan.rename(columns={'Close': 'Freeport_McMoRan_Close'},inplace=True)

International_Paper = International_Paper [['Date','Open','Close']]
International_Paper.rename(columns={'Open': 'International_Paper_Open'},inplace=True)
International_Paper.rename(columns={'Close': 'International_Paper_Close'},inplace=True)

#Intl_Flavors_Fragrances = Intl_Flavors_Fragrances [['Date','Open','Close']]
#Intl_Flavors_Fragrances.rename(columns={'Open': 'Intl_Flavors_Fragrances_Open'},inplace=True)
#Intl_Flavors_Fragrances.rename(columns={'Close': 'Intl_Flavors_Fragrances_Close'},inplace=True)

#Linde = Linde [['Date','Open','Close']]
#Linde.rename(columns={'Open': 'Linde_Open'},inplace=True)
#Linde.rename(columns={'Close': 'Linde_Close'},inplace=True)

LyondellBasell = LyondellBasell [['Date','Open','Close']]
LyondellBasell.rename(columns={'Open': 'LyondellBasell_Open'},inplace=True)
LyondellBasell.rename(columns={'Close': 'LyondellBasell_Close'},inplace=True)

Martin_Marietta_Materials = Martin_Marietta_Materials [['Date','Open','Close']]
Martin_Marietta_Materials.rename(columns={'Open': 'Martin_Marietta_Materials_Open'},inplace=True)
Martin_Marietta_Materials.rename(columns={'Close': 'Martin_Marietta_Materials_Close'},inplace=True)

The_Mosaic_Company = The_Mosaic_Company [['Date','Open','Close']]
The_Mosaic_Company.rename(columns={'Open': 'The_Mosaic_Company_Open'},inplace=True)
The_Mosaic_Company.rename(columns={'Close': 'The_Mosaic_Company_Close'},inplace=True)

Newmont_Mining_Corporation = Newmont_Mining_Corporation [['Date','Open','Close']]
Newmont_Mining_Corporation.rename(columns={'Open': 'Newmont_Mining_Corporation_Open'},inplace=True)
Newmont_Mining_Corporation.rename(columns={'Close': 'Newmont_Mining_Corporation_Close'},inplace=True)

Nucor = Nucor [['Date','Open','Close']]
Nucor.rename(columns={'Open': 'Nucor_Open'},inplace=True)
Nucor.rename(columns={'Close': 'Nucor_Close'},inplace=True)

Packaging_Corporation_of_America = Packaging_Corporation_of_America [['Date','Open','Close']]
Packaging_Corporation_of_America.rename(columns={'Open': 'Packaging_Corporation_of_America_Open'},inplace=True)
Packaging_Corporation_of_America.rename(columns={'Close': 'Packaging_Corporation_of_America_Close'},inplace=True)



# In[66]:


PPG_Industries = quandl.get("WIKI/PPG",start_date="2006-10-20", end_date="2013-10-20") #Materials
Sealed_Air = quandl.get("WIKI/SEE",start_date="2006-10-20", end_date="2013-10-20")
Sherwin_Williams = quandl.get("WIKI/SHW",start_date="2006-10-20", end_date="2013-10-20")
Vulcan_Materials = quandl.get("WIKI/VMC",start_date="2006-10-20", end_date="2013-10-20")
WestRock = quandl.get("WIKI/WRK",start_date="2006-10-20", end_date="2013-10-20")



#indexing
PPG_Industries.reset_index(inplace=True)
Sealed_Air.reset_index(inplace=True)
Sherwin_Williams.reset_index(inplace=True)
Vulcan_Materials.reset_index(inplace=True)
WestRock.reset_index(inplace=True)

#Removing/Renaming columns
PPG_Industries = PPG_Industries [['Date','Open','Close']]
PPG_Industries.rename(columns={'Open': 'PPG_Industries_Open'},inplace=True)
PPG_Industries.rename(columns={'Close': 'PPG_Industries_Close'},inplace=True)

Sealed_Air = Sealed_Air [['Date','Open','Close']]
Sealed_Air.rename(columns={'Open': 'Sealed_Air_Open'},inplace=True)
Sealed_Air.rename(columns={'Close': 'Sealed_Air_Close'},inplace=True)

Sherwin_Williams = Sherwin_Williams [['Date','Open','Close']]
Sherwin_Williams.rename(columns={'Open': 'Sherwin_Williams_Open'},inplace=True)
Sherwin_Williams.rename(columns={'Close': 'Sherwin_Williams_Close'},inplace=True)

Vulcan_Materials = Vulcan_Materials [['Date','Open','Close']]
Vulcan_Materials.rename(columns={'Open': 'Vulcan_Materials_Open'},inplace=True)
Vulcan_Materials.rename(columns={'Close': 'Vulcan_Materials_Close'},inplace=True)

WestRock = WestRock [['Date','Open','Close']]
WestRock.rename(columns={'Open': 'WestRock_Open'},inplace=True)
WestRock.rename(columns={'Close': 'WestRock_Close'},inplace=True)



# In[67]:


Air_Products_Chemicals.shape
Albemarle_Corp.shape
Avery_Dennison.shape
Ball.shape
Celanese.shape
CF_Industries_Holdings.shape
#DowDuPont.shape
Eastman_Chemical.shape
FMC_Corporation.shape
Ecolab.shape

Freeport_McMoRan.shape
International_Paper.shape
#Intl_Flavors_Fragrances.shape
#Linde.shape
LyondellBasell.shape
Martin_Marietta_Materials.shape
The_Mosaic_Company.shape
Newmont_Mining_Corporation.shape
Nucor.shape
Packaging_Corporation_of_America.shape

PPG_Industries.shape
Sealed_Air.shape
Sherwin_Williams.shape
Vulcan_Materials.shape
WestRock.shape



# In[68]:


#Materials = pd.concat([Air_Products_Chemicals[['Date','Air_Products_Chemicals_Open','Air_Products_Chemicals_Close']], Albemarle_Corp[['Albemarle_Corp_Open','Albemarle_Corp_Close']], Avery_Dennison[['Avery_Dennison_Open', 'Avery_Dennison_Close']], Ball[['Ball_Open','Ball_Close']],Celanese[['Celanese_Open','Celanese_Close']],CF_Industries_Holdings[['CF_Industries_Holdings_Open','CF_Industries_Holdings_Close']],DowDuPont[['DowDuPont_Open','DowDuPont_Close']],Eastman_Chemical[['Eastman_Chemical_Open','Eastman_Chemical_Close']],Ecolab[['Ecolab_Open','Ecolab_Close']],FMC_Corporation[['FMC_Corporation_Open','FMC_Corporation_Close']],Freeport_McMoRan[['Freeport_McMoRan_Open','Freeport_McMoRan_Close']],International_Paper[['International_Paper_Open','International_Paper_Close']],LyondellBasell[['LyondellBasell_Open','LyondellBasell_Close']],Martin_Marietta_Materials[['Martin_Marietta_Materials_Open','Martin_Marietta_Materials_Close']],The_Mosaic_Company[['The_Mosaic_Company_Open','The_Mosaic_Company_Close']],Newmont_Mining_Corporation[['Newmont_Mining_Corporation_Open','Newmont_Mining_Corporation_Close']], Nucor[['Nucor_Open','Nucor_Close']],Packaging_Corporation_of_America[['Packaging_Corporation_of_America_Open','Packaging_Corporation_of_America_Close']],PPG_Industries[['PPG_Industries_Open','PPG_Industries_Close']],Sealed_Air[['Sealed_Air_Open','Sealed_Air_Close']],Sherwin_Williams[['Sherwin_Williams_Open','Sherwin_Williams_Close']],Vulcan_Materials[['Vulcan_Materials_Open','Vulcan_Materials_Close']], WestRock[['WestRock_Open','WestRock_Close']]],axis=1)

Materials = (Air_Products_Chemicals).merge(Albemarle_Corp).merge(Avery_Dennison).merge(Ball).merge(Celanese).merge(CF_Industries_Holdings).merge(Eastman_Chemical).merge(Ecolab).merge(FMC_Corporation).merge(Freeport_McMoRan).merge(International_Paper).merge(LyondellBasell).merge(Martin_Marietta_Materials).merge(The_Mosaic_Company).merge(Newmont_Mining_Corporation).merge(Nucor).merge(Packaging_Corporation_of_America).merge(Sealed_Air).merge(Sherwin_Williams).merge(Vulcan_Materials)
Materials

#DowDuPont
#PPG_Industries
#WestRock

#Materials.isnull().sum()


# In[69]:


Alexandria_Real_Estate_Equities = quandl.get("WIKI/ARE",start_date="2006-10-20", end_date="2013-10-20") #Real Estate
American_Tower_Corp = quandl.get("WIKI/AMT",start_date="2006-10-20", end_date="2013-10-20")
Apartment_Investment_Management = quandl.get("WIKI/AIV",start_date="2006-10-20", end_date="2013-10-20")
AvalonBay_Communities = quandl.get("WIKI/AVB",start_date="2006-10-20", end_date="2013-10-20")
Boston_Properties = quandl.get("WIKI/BXP",start_date="2006-10-20", end_date="2013-10-20")
CBRE  = quandl.get("WIKI/CBRE",start_date="2006-10-20", end_date="2013-10-20")
Crown_Castle_International = quandl.get("WIKI/CCI",start_date="2006-10-20", end_date="2013-10-20")
Digital_Realty_Trust = quandl.get("WIKI/DLR",start_date="2006-10-20", end_date="2013-10-20")
Duke_Realty = quandl.get("WIKI/DRE",start_date="2006-10-20", end_date="2013-10-20")
Equinix = quandl.get("WIKI/EQIX",start_date="2006-10-20", end_date="2013-10-20")




#indexing
Alexandria_Real_Estate_Equities.reset_index(inplace=True)
American_Tower_Corp.reset_index(inplace=True)
AvalonBay_Communities.reset_index(inplace=True)
Apartment_Investment_Management.reset_index(inplace=True)
Boston_Properties.reset_index(inplace=True)
CBRE.reset_index(inplace=True)
Crown_Castle_International.reset_index(inplace=True)
Digital_Realty_Trust.reset_index(inplace=True)
Duke_Realty.reset_index(inplace=True)
Equinix.reset_index(inplace=True)


#Removing/Renaming columns
Alexandria_Real_Estate_Equities = Alexandria_Real_Estate_Equities [['Date','Open','Close']]
Alexandria_Real_Estate_Equities.rename(columns={'Open': 'Alexandria_Real_Estate_Equities_Open'},inplace=True)
Alexandria_Real_Estate_Equities.rename(columns={'Close': 'Alexandria_Real_Estate_Equities_Close'},inplace=True)

American_Tower_Corp = American_Tower_Corp [['Date','Open','Close']]
American_Tower_Corp.rename(columns={'Open': 'American_Tower_Corp_Open'},inplace=True)
American_Tower_Corp.rename(columns={'Close': 'American_Tower_Corp_Close'},inplace=True)

AvalonBay_Communities = AvalonBay_Communities [['Date','Open','Close']]
AvalonBay_Communities.rename(columns={'Open': 'AvalonBay_Communities_Open'},inplace=True)
AvalonBay_Communities.rename(columns={'Close': 'AvalonBay_Communities_Close'},inplace=True)

Boston_Properties = Boston_Properties [['Date','Open','Close']]
Boston_Properties.rename(columns={'Open': 'Boston_Properties_Open'},inplace=True)
Boston_Properties.rename(columns={'Close': 'Boston_Properties_Close'},inplace=True)

Apartment_Investment_Management = Apartment_Investment_Management [['Date','Open','Close']]
Apartment_Investment_Management.rename(columns={'Open': 'Apartment_Investment_Management_Open'},inplace=True)
Apartment_Investment_Management.rename(columns={'Close': 'Apartment_Investment_Management_Close'},inplace=True)

CBRE = CBRE [['Date','Open','Close']]
CBRE.rename(columns={'Open': 'CBRE_Open'},inplace=True)
CBRE.rename(columns={'Close': 'CBRE_Close'},inplace=True)

Crown_Castle_International = Crown_Castle_International [['Date','Open','Close']]
Crown_Castle_International.rename(columns={'Open': 'Crown_Castle_International_Open'},inplace=True)
Crown_Castle_International.rename(columns={'Close': 'Crown_Castle_International_Close'},inplace=True)

Digital_Realty_Trust = Digital_Realty_Trust [['Date','Open','Close']]
Digital_Realty_Trust.rename(columns={'Open': 'Digital_Realty_Trust_Open'},inplace=True)
Digital_Realty_Trust.rename(columns={'Close': 'Digital_Realty_Trust_Close'},inplace=True)

Duke_Realty = Duke_Realty [['Date','Open','Close']]
Duke_Realty.rename(columns={'Open': 'Duke_Realty_Open'},inplace=True)
Duke_Realty.rename(columns={'Close': 'Duke_Realty_Close'},inplace=True)

Equinix = Equinix [['Date','Open','Close']]
Equinix.rename(columns={'Open': 'Equinix_Open'},inplace=True)
Equinix.rename(columns={'Close': 'Equinix_Close'},inplace=True)


# In[70]:


Equity_Residential = quandl.get("WIKI/EQR",start_date="2006-10-20", end_date="2013-10-20") #Real Estate
Essex_Property_Trust  = quandl.get("WIKI/ESS",start_date="2006-10-20", end_date="2013-10-20")
Extra_Space_Storage  = quandl.get("WIKI/EXR",start_date="2006-10-20", end_date="2013-10-20")
Federal_Realty_Investment_Trust = quandl.get("WIKI/FRT",start_date="2006-10-20", end_date="2013-10-20")
HCP = quandl.get("WIKI/HCP",start_date="2006-10-20", end_date="2013-10-20")
Host_Hotels_Resorts = quandl.get("WIKI/HST",start_date="2006-10-20", end_date="2013-10-20")
Iron_Mountain = quandl.get("WIKI/IRM",start_date="2006-10-20", end_date="2013-10-20")
Kimco_Realty = quandl.get("WIKI/KIM",start_date="2006-10-20", end_date="2013-10-20")
Macerich = quandl.get("WIKI/MAC",start_date="2006-10-20", end_date="2013-10-20")
Mid_America_Apartments = quandl.get("WIKI/MAA",start_date="2006-10-20", end_date="2013-10-20")


#indexing
Equity_Residential.reset_index(inplace=True)
Essex_Property_Trust.reset_index(inplace=True)
Extra_Space_Storage.reset_index(inplace=True)
Federal_Realty_Investment_Trust.reset_index(inplace=True)
HCP.reset_index(inplace=True)
Host_Hotels_Resorts.reset_index(inplace=True)
Iron_Mountain.reset_index(inplace=True)
Kimco_Realty.reset_index(inplace=True)
Macerich.reset_index(inplace=True)
Mid_America_Apartments.reset_index(inplace=True)

#Removing/Renaming columns
Equity_Residential = Equity_Residential [['Date','Open','Close']]
Equity_Residential.rename(columns={'Open': 'Equity_Residential_Open'},inplace=True)
Equity_Residential.rename(columns={'Close': 'Equity_Residential_Close'},inplace=True)

Essex_Property_Trust = Essex_Property_Trust [['Date','Open','Close']]
Essex_Property_Trust.rename(columns={'Open': 'Essex_Property_Trust_Open'},inplace=True)
Essex_Property_Trust.rename(columns={'Close': 'Essex_Property_Trust_Close'},inplace=True)

Extra_Space_Storage = Extra_Space_Storage [['Date','Open','Close']]
Extra_Space_Storage.rename(columns={'Open': 'Extra_Space_Storage_Open'},inplace=True)
Extra_Space_Storage.rename(columns={'Close': 'Extra_Space_Storage_Close'},inplace=True)

Federal_Realty_Investment_Trust = Federal_Realty_Investment_Trust [['Date','Open','Close']]
Federal_Realty_Investment_Trust.rename(columns={'Open': 'Federal_Realty_Investment_Trust_Open'},inplace=True)
Federal_Realty_Investment_Trust.rename(columns={'Close': 'Federal_Realty_Investment_Trust_Close'},inplace=True)

HCP = HCP [['Date','Open','Close']]
HCP.rename(columns={'Open': 'HCP_Open'},inplace=True)
HCP.rename(columns={'Close': 'HCP_Close'},inplace=True)

Host_Hotels_Resorts = Host_Hotels_Resorts [['Date','Open','Close']]
Host_Hotels_Resorts.rename(columns={'Open': 'Host_Hotels_Resorts_Open'},inplace=True)
Host_Hotels_Resorts.rename(columns={'Close': 'Host_Hotels_Resorts_Close'},inplace=True)

Iron_Mountain = Iron_Mountain [['Date','Open','Close']]
Iron_Mountain.rename(columns={'Open': 'Iron_Mountain_Open'},inplace=True)
Iron_Mountain.rename(columns={'Close': 'Iron_Mountain_Close'},inplace=True)

Kimco_Realty = Kimco_Realty [['Date','Open','Close']]
Kimco_Realty.rename(columns={'Open': 'Kimco_Realty_Open'},inplace=True)
Kimco_Realty.rename(columns={'Close': 'Kimco_Realty_Close'},inplace=True)

Macerich = Macerich [['Date','Open','Close']]
Macerich.rename(columns={'Open': 'Macerich_Open'},inplace=True)
Macerich.rename(columns={'Close': 'Macerich_Close'},inplace=True)

Mid_America_Apartments = Mid_America_Apartments [['Date','Open','Close']]
Mid_America_Apartments.rename(columns={'Open': 'Mid_America_Apartments_Open'},inplace=True)
Mid_America_Apartments.rename(columns={'Close': 'Mid_America_Apartments_Close'},inplace=True)



# In[71]:


Prologis = quandl.get("WIKI/PLD",start_date="2006-10-20", end_date="2013-10-20") #Real Estate
Public_Storage  = quandl.get("WIKI/PSA",start_date="2006-10-20", end_date="2013-10-20")
Realty_Income_Corporation = quandl.get("WIKI/O",start_date="2006-10-20", end_date="2013-10-20")
Regency_Centers_Corporation  = quandl.get("WIKI/REG",start_date="2006-10-20", end_date="2013-10-20")
SBA_Communications  = quandl.get("WIKI/SBAC",start_date="2006-10-20", end_date="2013-10-20")
Simon_Property_Group = quandl.get("WIKI/SPG",start_date="2006-10-20", end_date="2013-10-20")
SL_Green_Realty  = quandl.get("WIKI/SLG",start_date="2006-10-20", end_date="2013-10-20")
UDR = quandl.get("WIKI/UDR",start_date="2006-10-20", end_date="2013-10-20")
Ventas = quandl.get("WIKI/VTR",start_date="2006-10-20", end_date="2013-10-20")
Vornado_Realty_Trust  = quandl.get("WIKI/VNO",start_date="2006-10-20", end_date="2013-10-20")
Welltower   = quandl.get("WIKI/WELL",start_date="2006-10-20", end_date="2013-10-20") #Real Estate
Weyerhaeuser = quandl.get("WIKI/WY",start_date="2006-10-20", end_date="2013-10-20")



#indexing
UDR.reset_index(inplace=True)
Prologis.reset_index(inplace=True)
Public_Storage.reset_index(inplace=True)
Realty_Income_Corporation.reset_index(inplace=True)
Regency_Centers_Corporation.reset_index(inplace=True)
SBA_Communications.reset_index(inplace=True)
Simon_Property_Group.reset_index(inplace=True)
SL_Green_Realty.reset_index(inplace=True)
Ventas.reset_index(inplace=True)
Vornado_Realty_Trust.reset_index(inplace=True)
Welltower.reset_index(inplace=True)
Weyerhaeuser.reset_index(inplace=True)

#Removing/Renaming columns
UDR = UDR [['Date','Open','Close']]
UDR.rename(columns={'Open': 'UDR_Open'},inplace=True)
UDR.rename(columns={'Close': 'UDR_Close'},inplace=True)

Prologis = Prologis [['Date','Open','Close']]
Prologis.rename(columns={'Open': 'Prologis_Open'},inplace=True)
Prologis.rename(columns={'Close': 'Prologis_Close'},inplace=True)

Public_Storage = Public_Storage [['Date','Open','Close']]
Public_Storage.rename(columns={'Open': 'Public_Storage_Open'},inplace=True)
Public_Storage.rename(columns={'Close': 'Public_Storage_Close'},inplace=True)

Realty_Income_Corporation = Realty_Income_Corporation [['Date','Open','Close']]
Realty_Income_Corporation.rename(columns={'Open': 'Realty_Income_Corporation_Open'},inplace=True)
Realty_Income_Corporation.rename(columns={'Close': 'Realty_Income_Corporation_Close'},inplace=True)

Regency_Centers_Corporation = Regency_Centers_Corporation [['Date','Open','Close']]
Regency_Centers_Corporation.rename(columns={'Open': 'Regency_Centers_Corporation_Open'},inplace=True)
Regency_Centers_Corporation.rename(columns={'Close': 'Regency_Centers_Corporation_Close'},inplace=True)

SBA_Communications = SBA_Communications [['Date','Open','Close']]
SBA_Communications.rename(columns={'Open': 'SBA_Communications_Open'},inplace=True)
SBA_Communications.rename(columns={'Close': 'SBA_Communications_Close'},inplace=True)

Simon_Property_Group = Simon_Property_Group [['Date','Open','Close']]
Simon_Property_Group.rename(columns={'Open': 'Simon_Property_Group_Open'},inplace=True)
Simon_Property_Group.rename(columns={'Close': 'Simon_Property_Group_Close'},inplace=True)

SL_Green_Realty = SL_Green_Realty [['Date','Open','Close']]
SL_Green_Realty.rename(columns={'Open': 'SL_Green_Realty_Open'},inplace=True)
SL_Green_Realty.rename(columns={'Close': 'SL_Green_Realty_Close'},inplace=True)

Ventas = Ventas [['Date','Open','Close']]
Ventas.rename(columns={'Open': 'Ventas_Open'},inplace=True)
Ventas.rename(columns={'Close': 'Ventas_Close'},inplace=True)

Vornado_Realty_Trust = Vornado_Realty_Trust [['Date','Open','Close']]
Vornado_Realty_Trust.rename(columns={'Open': 'Vornado_Realty_Trust_Open'},inplace=True)
Vornado_Realty_Trust.rename(columns={'Close': 'Vornado_Realty_Trust_Close'},inplace=True)


Welltower = Welltower [['Date','Open','Close']]
Welltower.rename(columns={'Open': 'Welltower_Open'},inplace=True)
Welltower.rename(columns={'Close': 'Welltower_Close'},inplace=True)


Weyerhaeuser = Weyerhaeuser [['Date','Open','Close']]
Weyerhaeuser.rename(columns={'Open': 'Weyerhaeuser_Open'},inplace=True)
Weyerhaeuser.rename(columns={'Close': 'Weyerhaeuser_Close'},inplace=True)


# In[72]:


#Real_Estate = pd.concat([Alexandria_Real_Estate_Equities[['Date','Alexandria_Real_Estate_Equities_Open','Alexandria_Real_Estate_Equities_Close']], American_Tower_Corp[['American_Tower_Corp_Open','American_Tower_Corp_Close']], Apartment_Investment_Management[['Apartment_Investment_Management_Open', 'Apartment_Investment_Management_Close']], AvalonBay_Communities[['AvalonBay_Communities_Open','AvalonBay_Communities_Close']],Boston_Properties[['Boston_Properties_Open','Boston_Properties_Close']],CBRE[['CBRE_Open','CBRE_Close']],Crown_Castle_International[['Crown_Castle_International_Open','Crown_Castle_International_Close']],Digital_Realty_Trust[['Digital_Realty_Trust_Open','Digital_Realty_Trust_Close']],Duke_Realty[['Duke_Realty_Open','Duke_Realty_Close']],Equinix[['Equinix_Open','Equinix_Close']],Equity_Residential[['Equity_Residential_Open','Equity_Residential_Close']],Essex_Property_Trust[['Essex_Property_Trust_Open','Essex_Property_Trust_Close']],Extra_Space_Storage[['Extra_Space_Storage_Open','Extra_Space_Storage_Close']],Federal_Realty_Investment_Trust[['Federal_Realty_Investment_Trust_Open','Federal_Realty_Investment_Trust_Close']],HCP[['HCP_Open','HCP_Close']],Host_Hotels_Resorts[['Host_Hotels_Resorts_Open','Host_Hotels_Resorts_Close']], Iron_Mountain[['Iron_Mountain_Open','Iron_Mountain_Close']],Kimco_Realty[['Kimco_Realty_Open','Kimco_Realty_Close']],Macerich[['Macerich_Open','Macerich_Close']],Mid_America_Apartments[['Mid_America_Apartments_Open','Mid_America_Apartments_Close']],Prologis[['Prologis_Open','Prologis_Close']],Public_Storage[['Public_Storage_Open','Public_Storage_Close']], Realty_Income_Corporation[['Realty_Income_Corporation_Open','Realty_Income_Corporation_Close']],SBA_Communications[['SBA_Communications_Open','SBA_Communications_Close']], Simon_Property_Group[['Simon_Property_Group_Open','Simon_Property_Group_Close']], SL_Green_Realty[['SL_Green_Realty_Open', 'SL_Green_Realty_Close']], UDR[['UDR_Open','UDR_Close']],Ventas[['Ventas_Open','Ventas_Close']],Vornado_Realty_Trust[['Vornado_Realty_Trust_Open','Vornado_Realty_Trust_Close']],Welltower[['Welltower_Open','Welltower_Close']],Weyerhaeuser[['Weyerhaeuser_Open','Weyerhaeuser_Close']]],axis=1)

Real_Estate = (American_Tower_Corp).merge(Alexandria_Real_Estate_Equities).merge(Apartment_Investment_Management).merge(AvalonBay_Communities).merge(Boston_Properties).merge(Crown_Castle_International).merge(Digital_Realty_Trust).merge(Duke_Realty).merge(Equinix).merge(Equity_Residential).merge(Essex_Property_Trust).merge(Extra_Space_Storage).merge(Federal_Realty_Investment_Trust).merge(HCP).merge(Host_Hotels_Resorts).merge(Iron_Mountain).merge(Kimco_Realty).merge(Macerich).merge(Mid_America_Apartments).merge(Prologis).merge(Public_Storage).merge(Realty_Income_Corporation).merge(Regency_Centers_Corporation).merge(SBA_Communications).merge(Simon_Property_Group).merge(SL_Green_Realty).merge(UDR).merge(Ventas).merge(Vornado_Realty_Trust).merge(Weyerhaeuser)
Real_Estate
#Real_Estate.isnull().sum()
#CBRE
#Welltower


# In[73]:


AES = quandl.get("WIKI/AES",start_date="2006-10-20", end_date="2013-10-20") #Utilities
Alliant_Energy_Corp = quandl.get("WIKI/LNT",start_date="2006-10-20", end_date="2013-10-20")
Ameren_Corp = quandl.get("WIKI/AEE",start_date="2006-10-20", end_date="2013-10-20")
American_Electric_Power = quandl.get("WIKI/AEP",start_date="2006-10-20", end_date="2013-10-20")
American_Water_Works_Company = quandl.get("WIKI/AWK",start_date="2006-10-20", end_date="2013-10-20")
CenterPoint_Energy = quandl.get("WIKI/CNP",start_date="2006-10-20", end_date="2013-10-20")
CMS_Energy = quandl.get("WIKI/CMS",start_date="2006-10-20", end_date="2013-10-20")
Consolidated_Edison  = quandl.get("WIKI/ED",start_date="2006-10-20", end_date="2013-10-20")
DTE_Energy = quandl.get("WIKI/DTE",start_date="2006-10-20", end_date="2013-10-20")
Duke_Energy = quandl.get("WIKI/DUK",start_date="2006-10-20", end_date="2013-10-20")



#indexing

AES.reset_index(inplace=True)
Alliant_Energy_Corp.reset_index(inplace=True)
Ameren_Corp.reset_index(inplace=True)
American_Electric_Power.reset_index(inplace=True)
American_Water_Works_Company.reset_index(inplace=True)
CenterPoint_Energy.reset_index(inplace=True)
CMS_Energy.reset_index(inplace=True)
Consolidated_Edison.reset_index(inplace=True)
DTE_Energy.reset_index(inplace=True)
Duke_Energy.reset_index(inplace=True)


#Removing/Renaming columns
AES = AES [['Date','Open','Close']]
AES.rename(columns={'Open': 'AES_Open'},inplace=True)
AES.rename(columns={'Close': 'AES_Close'},inplace=True)

Alliant_Energy_Corp = Alliant_Energy_Corp [['Date','Open','Close']]
Alliant_Energy_Corp.rename(columns={'Open': 'Alliant_Energy_Corp_Open'},inplace=True)
Alliant_Energy_Corp.rename(columns={'Close': 'Alliant_Energy_Corp_Close'},inplace=True)

Ameren_Corp = Ameren_Corp [['Date','Open','Close']]
Ameren_Corp.rename(columns={'Open': 'Ameren_Corp_Open'},inplace=True)
Ameren_Corp.rename(columns={'Close': 'Ameren_Corp_Close'},inplace=True)

American_Electric_Power = American_Electric_Power [['Date','Open','Close']]
American_Electric_Power.rename(columns={'Open': 'American_Electric_Power_Open'},inplace=True)
American_Electric_Power.rename(columns={'Close': 'American_Electric_Power_Close'},inplace=True)

American_Water_Works_Company = American_Water_Works_Company [['Date','Open','Close']]
American_Water_Works_Company.rename(columns={'Open': 'American_Water_Works_Company_Open'},inplace=True)
American_Water_Works_Company.rename(columns={'Close': 'American_Water_Works_Company_Close'},inplace=True)

CenterPoint_Energy = CenterPoint_Energy [['Date','Open','Close']]
CenterPoint_Energy.rename(columns={'Open': 'CenterPoint_Energy_Open'},inplace=True)
CenterPoint_Energy.rename(columns={'Close': 'CenterPoint_Energy_Close'},inplace=True)

CMS_Energy = CMS_Energy [['Date','Open','Close']]
CMS_Energy.rename(columns={'Open': 'CMS_Energy_Open'},inplace=True)
CMS_Energy.rename(columns={'Close': 'CMS_Energy_Close'},inplace=True)

Consolidated_Edison = Consolidated_Edison [['Date','Open','Close']]
Consolidated_Edison.rename(columns={'Open': 'Consolidated_Edison_Open'},inplace=True)
Consolidated_Edison.rename(columns={'Close': 'Consolidated_Edison_Close'},inplace=True)

DTE_Energy = DTE_Energy [['Date','Open','Close']]
DTE_Energy.rename(columns={'Open': 'DTE_Energy_Open'},inplace=True)
DTE_Energy.rename(columns={'Close': 'DTE_Energy_Close'},inplace=True)

Duke_Energy = Duke_Energy [['Date','Open','Close']]
Duke_Energy.rename(columns={'Open': 'Duke_Energy_Open'},inplace=True)
Duke_Energy.rename(columns={'Close': 'Duke_Energy_Close'},inplace=True)




# In[74]:


Dominion_Energy = quandl.get("WIKI/D",start_date="2006-10-20", end_date="2013-10-20") #Utilities
Edison_Int = quandl.get("WIKI/EIX",start_date="2006-10-20", end_date="2013-10-20")
Entergy = quandl.get("WIKI/ETR",start_date="2006-10-20", end_date="2013-10-20")
Evergy = quandl.get("WIKI/EVHC",start_date="2006-10-20", end_date="2013-10-20")
Eversource_Energy = quandl.get("WIKI/ES",start_date="2006-10-20", end_date="2013-10-20")
Exelon = quandl.get("WIKI/EXC",start_date="2006-10-20", end_date="2013-10-20")
FirstEnergy = quandl.get("WIKI/FE",start_date="2006-10-20", end_date="2013-10-20")
NextEra_Energy  = quandl.get("WIKI/NEE",start_date="2006-10-20", end_date="2013-10-20")
NiSource = quandl.get("WIKI/NI",start_date="2006-10-20", end_date="2013-10-20")
NRG_Energy  = quandl.get("WIKI/NRG",start_date="2006-10-20", end_date="2013-10-20")


#indexing
Dominion_Energy.reset_index(inplace=True)
Edison_Int.reset_index(inplace=True)
Entergy.reset_index(inplace=True)
Evergy.reset_index(inplace=True)
Eversource_Energy.reset_index(inplace=True)
Exelon.reset_index(inplace=True)
FirstEnergy.reset_index(inplace=True)
NextEra_Energy.reset_index(inplace=True)
NiSource.reset_index(inplace=True)
NRG_Energy.reset_index(inplace=True)

#Removing/Renaming columns
Dominion_Energy = Dominion_Energy [['Date','Open','Close']]
Dominion_Energy.rename(columns={'Open': 'Dominion_Energy_Open'},inplace=True)
Dominion_Energy.rename(columns={'Close': 'Dominion_Energy_Close'},inplace=True)

Edison_Int = Edison_Int [['Date','Open','Close']]
Edison_Int.rename(columns={'Open': 'Edison_Int_Open'},inplace=True)
Edison_Int.rename(columns={'Close': 'Edison_Int_Close'},inplace=True)

Entergy = Entergy [['Date','Open','Close']]
Entergy.rename(columns={'Open': 'Entergy_Open'},inplace=True)
Entergy.rename(columns={'Close': 'Entergy_Close'},inplace=True)

Evergy = Evergy [['Date','Open','Close']]
Evergy.rename(columns={'Open': 'Evergy_Open'},inplace=True)
Evergy.rename(columns={'Close': 'Evergy_Close'},inplace=True)

Eversource_Energy = Eversource_Energy [['Date','Open','Close']]
Eversource_Energy.rename(columns={'Open': 'Eversource_Energy_Open'},inplace=True)
Eversource_Energy.rename(columns={'Close': 'Eversource_Energy_Close'},inplace=True)

Exelon = Exelon [['Date','Open','Close']]
Exelon.rename(columns={'Open': 'Exelon_Open'},inplace=True)
Exelon.rename(columns={'Close': 'Exelon_Close'},inplace=True)

FirstEnergy = FirstEnergy [['Date','Open','Close']]
FirstEnergy.rename(columns={'Open': 'FirstEnergy_Open'},inplace=True)
FirstEnergy.rename(columns={'Close': 'FirstEnergy_Close'},inplace=True)

NextEra_Energy = NextEra_Energy [['Date','Open','Close']]
NextEra_Energy.rename(columns={'Open': 'NextEra_Energy_Open'},inplace=True)
NextEra_Energy.rename(columns={'Close': 'NextEra_Energy_Close'},inplace=True)

NiSource = NiSource [['Date','Open','Close']]
NiSource.rename(columns={'Open': 'NiSource_Open'},inplace=True)
NiSource.rename(columns={'Close': 'NiSource_Close'},inplace=True)

NRG_Energy = NRG_Energy [['Date','Open','Close']]
NRG_Energy.rename(columns={'Open': 'NRG_Energy_Open'},inplace=True)
NRG_Energy.rename(columns={'Close': 'NRG_Energy_Close'},inplace=True)


# In[75]:


Pinnacle_West_Capital = quandl.get("WIKI/PNW",start_date="2006-10-20", end_date="2013-10-20") #Utilities
PPL_Corp = quandl.get("WIKI/PPL",start_date="2006-10-20", end_date="2013-10-20")
Public_Serv_Enterprise = quandl.get("WIKI/PEG",start_date="2006-10-20", end_date="2013-10-20")
Southern = quandl.get("WIKI/SO",start_date="2006-10-20", end_date="2013-10-20")
Sempra_Energy = quandl.get("WIKI/SRE",start_date="2006-10-20", end_date="2013-10-20")
Wec_Energy_Group = quandl.get("WIKI/WEC",start_date="2006-10-20", end_date="2013-10-20")
Xcel_Energy = quandl.get("WIKI/XEL",start_date="2006-10-20", end_date="2013-10-20")

#indexing
Pinnacle_West_Capital.reset_index(inplace=True)
PPL_Corp.reset_index(inplace=True)
Public_Serv_Enterprise.reset_index(inplace=True)
Southern.reset_index(inplace=True)
Sempra_Energy.reset_index(inplace=True)
Wec_Energy_Group.reset_index(inplace=True)
Xcel_Energy.reset_index(inplace=True)

#Removing/Renaming columns
Pinnacle_West_Capital = Pinnacle_West_Capital [['Date','Open','Close']]
Pinnacle_West_Capital.rename(columns={'Open': 'Pinnacle_West_Capital_Open'},inplace=True)
Pinnacle_West_Capital.rename(columns={'Close': 'Pinnacle_West_Capital_Close'},inplace=True)

PPL_Corp = PPL_Corp [['Date','Open','Close']]
PPL_Corp.rename(columns={'Open': 'PPL_Corp_Open'},inplace=True)
PPL_Corp.rename(columns={'Close': 'PPL_Corp_Close'},inplace=True)

Public_Serv_Enterprise = Public_Serv_Enterprise [['Date','Open','Close']]
Public_Serv_Enterprise.rename(columns={'Open': 'Public_Serv_Enterprise_Open'},inplace=True)
Public_Serv_Enterprise.rename(columns={'Close': 'Public_Serv_Enterprise_Close'},inplace=True)

Southern = Southern [['Date','Open','Close']]
Southern.rename(columns={'Open': 'Southern_Open'},inplace=True)
Southern.rename(columns={'Close': 'Southern_Close'},inplace=True)

Sempra_Energy = Sempra_Energy [['Date','Open','Close']]
Sempra_Energy.rename(columns={'Open': 'Sempra_Energy_Open'},inplace=True)
Sempra_Energy.rename(columns={'Close': 'Sempra_Energy_Close'},inplace=True)

Wec_Energy_Group = Wec_Energy_Group [['Date','Open','Close']]
Wec_Energy_Group.rename(columns={'Open': 'Wec_Energy_Group_Open'},inplace=True)
Wec_Energy_Group.rename(columns={'Close': 'Wec_Energy_Group_Close'},inplace=True)

Xcel_Energy = Xcel_Energy [['Date','Open','Close']]
Xcel_Energy.rename(columns={'Open': 'Xcel_Energy_Open'},inplace=True)
Xcel_Energy.rename(columns={'Close': 'Xcel_Energy_Close'},inplace=True)



# In[76]:


#Utilities = pd.concat([AES[['Date','AES_Open','AES_Close']], Alliant_Energy_Corp[['Alliant_Energy_Corp_Open','Alliant_Energy_Corp_Close']], Ameren_Corp[['Ameren_Corp_Open', 'Ameren_Corp_Close']], American_Electric_Power[['American_Electric_Power_Open','American_Electric_Power_Close']],American_Water_Works_Company[['American_Water_Works_Company_Open','American_Water_Works_Company_Close']],CenterPoint_Energy[['CenterPoint_Energy_Open','CenterPoint_Energy_Close']],CMS_Energy[['CMS_Energy_Open','CMS_Energy_Close']],Consolidated_Edison[['Consolidated_Edison_Open','Consolidated_Edison_Close']],DTE_Energy[['DTE_Energy_Open','DTE_Energy_Close']],Duke_Energy[['Duke_Energy_Open','Duke_Energy_Close']],Dominion_Energy[['Dominion_Energy_Open','Dominion_Energy_Close']],Edison_Int[['Edison_Int_Open','Edison_Int_Close']],Entergy[['Entergy_Open','Entergy_Close']],Evergy[['Evergy_Open','Evergy_Close']],Eversource_Energy[['Eversource_Energy_Open','Eversource_Energy_Close']],Exelon[['Exelon_Open','Exelon_Close']], FirstEnergy[['FirstEnergy_Open','FirstEnergy_Close']],NextEra_Energy[['NextEra_Energy_Open','NextEra_Energy_Close']],NiSource[['NiSource_Open','NiSource_Close']],NRG_Energy[['NRG_Energy_Open','NRG_Energy_Close']],Pinnacle_West_Capital[['Pinnacle_West_Capital_Open','Pinnacle_West_Capital_Close']],PPL_Corp[['PPL_Corp_Open','PPL_Corp_Close']], Public_Serv_Enterprise[['Public_Serv_Enterprise_Open','Public_Serv_Enterprise_Close']],Southern[['Southern_Open','Southern_Close']], Sempra_Energy[['Sempra_Energy_Open','Sempra_Energy_Close']], Wec_Energy_Group[['Wec_Energy_Group_Open', 'Wec_Energy_Group_Close']], Xcel_Energy[['Xcel_Energy_Open','Xcel_Energy_Close']]],axis=1)

Utilities =  (AES).merge(Alliant_Energy_Corp).merge(Ameren_Corp).merge(American_Electric_Power).merge(American_Water_Works_Company).merge(CMS_Energy).merge(Consolidated_Edison).merge(DTE_Energy).merge(Duke_Energy).merge(Dominion_Energy).merge(Edison_Int).merge(Entergy).merge(Evergy).merge(Eversource_Energy).merge(Exelon).merge(FirstEnergy).merge(NextEra_Energy).merge(NiSource).merge(NRG_Energy).merge(Pinnacle_West_Capital).merge(PPL_Corp).merge(Public_Serv_Enterprise).merge(Southern).merge(Sempra_Energy).merge(Wec_Energy_Group).merge(Xcel_Energy)


Utilities


#Utilities.isnull().sum()


# In[79]:


SP500= Communication_Services.merge(Consumer_Discretionary).merge(Consumer_Staples).merge(Energy).merge(Financials).merge(HealthCare).merge(Industrials).merge(Information_Technology).merge(Materials).merge(Real_Estate).merge(Utilities)

SP500

SP500.to_csv('SP500.csv')


# In[86]:


SP500 = read_csv('SP500.csv')
SP500 = SP500.drop(['Unnamed: 0'], axis=1)
SP500


# In[31]:





# In[32]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for Alphabet Inc Class A
SMA_Alphabet_Inc_Class_C = calculate_SMA(Alphabet_Inc_Class_C.Alphabet_Inc_Class_C_Close)

Alphabet_Inc_Class_C.Alphabet_Inc_Class_C_Close[:365].plot(title='Alphabet Inc Class C',label='Alphabet Inc Class C', ax=axes[0])

SMA_Alphabet_Inc_Class_C[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for Alphabet Inc Class C
upper_band_Alphabet_Inc_Class_C, lower_band_Alphabet_Inc_Class_C = calculate_BB(Alphabet_Inc_Class_C.Alphabet_Inc_Class_C_Close)

upper_band_Alphabet_Inc_Class_C[:365].plot(label='upper band', ax=axes[0])
lower_band_Alphabet_Inc_Class_C[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for Alphabet Inc Class C
DIF_Alphabet_Inc_Class_C, MACD_Alphabet_Inc_Class_C = calculate_MACD(Alphabet_Inc_Class_C.Alphabet_Inc_Class_C_Close)

DIF_Alphabet_Inc_Class_C[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Alphabet_Inc_Class_C[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for Alphabet Inc Class C
RSI_Alphabet_Inc_Class_C = calculate_RSI(Alphabet_Inc_Class_C.Alphabet_Inc_Class_C_Close)
RSI_Alphabet_Inc_Class_C[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for Alphabet Inc Class C
STDEV_Alphabet_Inc_Class_C= calculate_stdev(Alphabet_Inc_Class_C.Alphabet_Inc_Class_C_Close)
STDEV_Alphabet_Inc_Class_C[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Alphabet_Inc_Class_C=Alphabet_Inc_Class_C.Alphabet_Inc_Class_C_Open - Alphabet_Inc_Class_C.Alphabet_Inc_Class_C_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[34]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for Alphabet Inc Class A
SMA_Alphabet_Inc_Class_A = calculate_SMA(Alphabet_Inc_Class_A.Alphabet_Inc_Class_A_Close)

Alphabet_Inc_Class_A.Alphabet_Inc_Class_A_Close[:365].plot(title='Alphabet Inc Class A',label='Alphabet Inc Class A', ax=axes[0])

SMA_Alphabet_Inc_Class_C[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for Alphabet Inc Class A
upper_band_Alphabet_Inc_Class_A, lower_band_Alphabet_Inc_Class_A = calculate_BB(Alphabet_Inc_Class_A.Alphabet_Inc_Class_A_Close)

upper_band_Alphabet_Inc_Class_A[:365].plot(label='upper band', ax=axes[0])
lower_band_Alphabet_Inc_Class_A[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for Alphabet Inc Class A
DIF_Alphabet_Inc_Class_A,MACD_Alphabet_Inc_Class_A = calculate_MACD(Alphabet_Inc_Class_A.Alphabet_Inc_Class_A_Close)

DIF_Alphabet_Inc_Class_A[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Alphabet_Inc_Class_A[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for Alphabet Inc Class A
RSI_Alphabet_Inc_Class_A = calculate_RSI(Alphabet_Inc_Class_A.Alphabet_Inc_Class_A_Close)
RSI_Alphabet_Inc_Class_A[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for Alphabet Inc Class A
STDEV_Alphabet_Inc_Class_A= calculate_stdev(Alphabet_Inc_Class_A.Alphabet_Inc_Class_A_Close)
STDEV_Alphabet_Inc_Class_A[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Alphabet_Inc_Class_A=Alphabet_Inc_Class_A.Alphabet_Inc_Class_A_Open - Alphabet_Inc_Class_A.Alphabet_Inc_Class_A_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[35]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for AT&T
SMA_ATT = calculate_SMA(ATT.ATT_Close)

ATT.ATT_Close[:365].plot(title='AT&T',label='AT&T', ax=axes[0])

SMA_ATT[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for AT&T
upper_band_ATT, lower_band_ATT = calculate_BB(ATT.ATT_Close)

upper_band_ATT[:365].plot(label='upper band', ax=axes[0])
lower_band_ATT[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for AT&T
DIF_ATT, MACD_ATT = calculate_MACD(ATT.ATT_Close)

DIF_ATT[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_ATT[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for AT&T
RSI_ATT = calculate_RSI(ATT.ATT_Close)
RSI_ATT[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for AT&T
STDEV_ATT= calculate_stdev(ATT.ATT_Close)
STDEV_ATT[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_ATT=ATT.ATT_Open - ATT.ATT_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[36]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for CBS
SMA_CBS = calculate_SMA(CBS.CBS_Close)

CBS.CBS_Close[:365].plot(title='CBS',label='CBS', ax=axes[0])

SMA_CBS[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for CBS
upper_band_CBS, lower_band_CBS = calculate_BB(CBS.CBS_Close)

upper_band_CBS[:365].plot(label='upper band', ax=axes[0])
lower_band_CBS[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for CBS
DIF_CBS, MACD_CBS = calculate_MACD(CBS.CBS_Close)

DIF_CBS[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_CBS[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for CBS
RSI_CBS = calculate_RSI(CBS.CBS_Close)
RSI_CBS[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for CBS
STDEV_CBS= calculate_stdev(CBS.CBS_Close)
STDEV_CBS[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_CBS=CBS.CBS_Open - CBS.CBS_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[38]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for CENTURYLINK
SMA_CenturyLink = calculate_SMA(CenturyLink.CenturyLink_Close)

CenturyLink.CenturyLink_Close[:365].plot(title='CenturyLink',label='CenturyLink', ax=axes[0])

SMA_CenturyLink[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for CENTURYLINK
upper_band_CenturyLink, lower_band_CenturyLink = calculate_BB(CenturyLink.CenturyLink_Close)

upper_band_CenturyLink[:365].plot(label='upper band', ax=axes[0])
lower_band_CenturyLink[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for CENTURYLINK
DIF_CenturyLink, MACD_CenturyLink = calculate_MACD(CenturyLink.CenturyLink_Close)

DIF_CenturyLink[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_CenturyLink[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for CENTURYLINK
RSI_CenturyLink = calculate_RSI(CenturyLink.CenturyLink_Close)
RSI_CenturyLink[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for CENTURYLINK
STDEV_CenturyLink= calculate_stdev(CenturyLink.CenturyLink_Close)
STDEV_CenturyLink[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_CenturyLink=CenturyLink.CenturyLink_Open - CenturyLink.CenturyLink_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[39]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for CHARTER_COMMUNICATIONS
SMA_Charter_Communications = calculate_SMA(Charter_Communications.Charter_Communications_Close)

Charter_Communications.Charter_Communications_Close[:365].plot(title='Charter_Communications',label='Charter_Communications', ax=axes[0])

SMA_Charter_Communications[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for CHARTER_COMMUNICATIONS
upper_band_Charter_Communications, lower_band_Charter_Communications = calculate_BB(Charter_Communications.Charter_Communications_Close)

upper_band_Charter_Communications[:365].plot(label='upper band', ax=axes[0])
lower_band_Charter_Communications[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for CHARTER_COMMUNICATIONS
DIF_Charter_Communications, MACD_Charter_Communications = calculate_MACD(Charter_Communications.Charter_Communications_Close)

DIF_Charter_Communications[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Charter_Communications[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for CHARTER_COMMUNICATIONS
RSI_Charter_Communications = calculate_RSI(Charter_Communications.Charter_Communications_Close)
RSI_Charter_Communications[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for CHARTER_COMMUNICATIONS
STDEV_Charter_Communications= calculate_stdev(Charter_Communications.Charter_Communications_Close)
STDEV_Charter_Communications[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Charter_Communications=Charter_Communications.Charter_Communications_Open - Charter_Communications.Charter_Communications_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[40]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for COMCAST
SMA_Comcast = calculate_SMA(Comcast.Comcast_Close)

Comcast.Comcast_Close[:365].plot(title='Comcast',label='Comcast', ax=axes[0])

SMA_Comcast[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for COMCAST
upper_band_Comcast, lower_band_Comcast = calculate_BB(Comcast.Comcast_Close)

upper_band_Comcast[:365].plot(label='upper band', ax=axes[0])
lower_band_Comcast[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for COMCAST
DIF_Comcast, MACD_Comcast = calculate_MACD(Comcast.Comcast_Close)

DIF_Comcast[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Comcast[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for COMCAST
RSI_Comcast = calculate_RSI(Comcast.Comcast_Close)
RSI_Comcast[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for COMCAST
STDEV_Comcast= calculate_stdev(Comcast.Comcast_Close)
STDEV_Comcast[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Comcast=Comcast.Comcast_Open - Comcast.Comcast_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')



# In[41]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for DISCOVERY_CLASS_A
SMA_Discovery_Class_A = calculate_SMA(Discovery_Class_A.Discovery_Class_A_Close)

Discovery_Class_A.Discovery_Class_A_Close[:365].plot(title='Discovery_Class_A',label='Discovery_Class_A', ax=axes[0])

SMA_Discovery_Class_A[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for DISCOVERY_CLASS_A
upper_band_Discovery_Class_A, lower_band_Discovery_Class_A = calculate_BB(Discovery_Class_A.Discovery_Class_A_Close)

upper_band_Discovery_Class_A[:365].plot(label='upper band', ax=axes[0])
lower_band_Discovery_Class_A[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for DISCOVERY_CLASS_A
DIF_Discovery_Class_A, MACD_Discovery_Class_A = calculate_MACD(Discovery_Class_A.Discovery_Class_A_Close)

DIF_Discovery_Class_A[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Discovery_Class_A[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for DISCOVERY_CLASS_A
RSI_Discovery_Class_A = calculate_RSI(Discovery_Class_A.Discovery_Class_A_Close)
RSI_Discovery_Class_A[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for DISCOVERY_CLASS_A
STDEV_Discovery_Class_A= calculate_stdev(Discovery_Class_A.Discovery_Class_A_Close)
STDEV_Discovery_Class_A[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Discovery_Class_A=Discovery_Class_A.Discovery_Class_A_Open - Discovery_Class_A.Discovery_Class_A_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[42]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for DISCOVERY_CLASS_C
SMA_Discovery_Class_C = calculate_SMA(Discovery_Class_C.Discovery_Class_C_Close)

Discovery_Class_C.Discovery_Class_C_Close[:365].plot(title='Discovery_Class_C',label='Discovery_Class_C', ax=axes[0])

SMA_Discovery_Class_C[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for DISCOVERY_CLASS_C
upper_band_Discovery_Class_C, lower_band_Discovery_Class_C = calculate_BB(Discovery_Class_C.Discovery_Class_C_Close)

upper_band_Discovery_Class_C[:365].plot(label='upper band', ax=axes[0])
lower_band_Discovery_Class_C[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for DISCOVERY_CLASS_C
DIF_Discovery_Class_C, MACD_Discovery_Class_C = calculate_MACD(Discovery_Class_C.Discovery_Class_C_Close)

DIF_Discovery_Class_C[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Discovery_Class_C[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for DISCOVERY_CLASS_C
RSI_Discovery_Class_C = calculate_RSI(Discovery_Class_C.Discovery_Class_C_Close)
RSI_Discovery_Class_C[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for DISCOVERY_CLASS_C
STDEV_Discovery_Class_C= calculate_stdev(Discovery_Class_C.Discovery_Class_C_Close)
STDEV_Discovery_Class_C[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Discovery_Class_C=Discovery_Class_C.Discovery_Class_C_Open - Discovery_Class_C.Discovery_Class_C_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[43]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for DISH
SMA_Dish = calculate_SMA(Dish.Dish_Close)

Dish.Dish_Close[:365].plot(title='Dish',label='Dish', ax=axes[0])

SMA_Dish[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for DISH
upper_band_Dish, lower_band_Dish = calculate_BB(Dish.Dish_Close)

upper_band_Dish[:365].plot(label='upper band', ax=axes[0])
lower_band_Dish[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for DISH
DIF_Dish, MACD_Dish = calculate_MACD(Dish.Dish_Close)

DIF_Dish[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Dish[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for DISH
RSI_Dish = calculate_RSI(Dish.Dish_Close)
RSI_Dish[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for DISH
STDEV_Dish= calculate_stdev(Dish.Dish_Close)
STDEV_Dish[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Dish=Dish.Dish_Open - Dish.Dish_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[44]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for ELECTRONIC_ARTS
SMA_Electronic_Arts = calculate_SMA(Electronic_Arts.Electronic_Arts_Close)

Electronic_Arts.Electronic_Arts_Close[:365].plot(title='Electronic_Arts',label='Electronic_Arts', ax=axes[0])

SMA_Electronic_Arts[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for ELECTRONIC_ARTS
upper_band_Electronic_Arts, lower_band_Electronic_Arts = calculate_BB(Electronic_Arts.Electronic_Arts_Close)

upper_band_Electronic_Arts[:365].plot(label='upper band', ax=axes[0])
lower_band_Electronic_Arts[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for ELECTRONIC_ARTS
DIF_Electronic_Arts, MACD_Electronic_Arts = calculate_MACD(Electronic_Arts.Electronic_Arts_Close)

DIF_Electronic_Arts[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Electronic_Arts[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for ELECTRONIC_ARTS
RSI_Electronic_Arts = calculate_RSI(Electronic_Arts.Electronic_Arts_Close)
RSI_Electronic_Arts[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for ELECTRONIC_ARTS
STDEV_Electronic_Arts= calculate_stdev(Electronic_Arts.Electronic_Arts_Close)
STDEV_Electronic_Arts[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Electronic_Arts=Electronic_Arts.Electronic_Arts_Open - Electronic_Arts.Electronic_Arts_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[45]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for FACEBOOK
SMA_Facebook = calculate_SMA(Facebook.Facebook_Close)

Facebook.Facebook_Close[:365].plot(title='Facebook',label='Facebook', ax=axes[0])

SMA_Facebook[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for FACEBOOK
upper_band_Facebook, lower_band_Facebook = calculate_BB(Facebook.Facebook_Close)

upper_band_Facebook[:365].plot(label='upper band', ax=axes[0])
lower_band_Facebook[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for FACEBOOK
DIF_Facebook, MACD_Facebook = calculate_MACD(Facebook.Facebook_Close)

DIF_Facebook[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Facebook[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for FACEBOOK
RSI_Facebook = calculate_RSI(Facebook.Facebook_Close)
RSI_Facebook[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for FACEBOOK
STDEV_Facebook= calculate_stdev(Facebook.Facebook_Close)
STDEV_Facebook[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Facebook=Facebook.Facebook_Open - Facebook.Facebook_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')



# In[46]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for INTERPUBLIC_GROUP
SMA_Interpublic_Group = calculate_SMA(Interpublic_Group.Interpublic_Group_Close)

Interpublic_Group.Interpublic_Group_Close[:365].plot(title='Interpublic_Group',label='Interpublic_Group', ax=axes[0])

SMA_Interpublic_Group[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for INTERPUBLIC_GROUP
upper_band_Interpublic_Group, lower_band_Interpublic_Group = calculate_BB(Interpublic_Group.Interpublic_Group_Close)

upper_band_Interpublic_Group[:365].plot(label='upper band', ax=axes[0])
lower_band_Interpublic_Group[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for INTERPUBLIC_GROUP
DIF_Interpublic_Group, MACD_Interpublic_Group = calculate_MACD(Interpublic_Group.Interpublic_Group_Close)

DIF_Interpublic_Group[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Interpublic_Group[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for INTERPUBLIC_GROUP
RSI_Interpublic_Group = calculate_RSI(Interpublic_Group.Interpublic_Group_Close)
RSI_Interpublic_Group[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for INTERPUBLIC_GROUP
STDEV_Interpublic_Group= calculate_stdev(Interpublic_Group.Interpublic_Group_Close)
STDEV_Interpublic_Group[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Interpublic_Group=Interpublic_Group.Interpublic_Group_Open - Interpublic_Group.Interpublic_Group_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[47]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for NETFLIX
SMA_Netflix = calculate_SMA(Netflix.Netflix_Close)

Netflix.Netflix_Close[:365].plot(title='Netflix',label='Netflix', ax=axes[0])

SMA_Netflix[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for NETFLIX
upper_band_Netflix, lower_band_Netflix = calculate_BB(Netflix.Netflix_Close)

upper_band_Netflix[:365].plot(label='upper band', ax=axes[0])
lower_band_Netflix[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for NETFLIX
DIF_Netflix, MACD_Netflix = calculate_MACD(Netflix.Netflix_Close)

DIF_Netflix[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Netflix[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for NETFLIX
RSI_Netflix = calculate_RSI(Netflix.Netflix_Close)
RSI_Netflix[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for NETFLIX
STDEV_Netflix= calculate_stdev(Netflix.Netflix_Close)
STDEV_Netflix[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Netflix=Netflix.Netflix_Open - Netflix.Netflix_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')



# In[48]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for NEWS_CORP_CLASS_A
SMA_News_Corp_Class_A = calculate_SMA(News_Corp_Class_A.News_Corp_Class_A_Close)

News_Corp_Class_A.News_Corp_Class_A_Close[:365].plot(title='News_Corp_Class_A',label='News_Corp_Class_A', ax=axes[0])

SMA_News_Corp_Class_A[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for NEWS_CORP_CLASS_A
upper_band_News_Corp_Class_A, lower_band_News_Corp_Class_A = calculate_BB(News_Corp_Class_A.News_Corp_Class_A_Close)

upper_band_News_Corp_Class_A[:365].plot(label='upper band', ax=axes[0])
lower_band_News_Corp_Class_A[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for NEWS_CORP_CLASS_A
DIF_News_Corp_Class_A, MACD_News_Corp_Class_A = calculate_MACD(News_Corp_Class_A.News_Corp_Class_A_Close)

DIF_News_Corp_Class_A[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_News_Corp_Class_A[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for NEWS_CORP_CLASS_A
RSI_News_Corp_Class_A = calculate_RSI(News_Corp_Class_A.News_Corp_Class_A_Close)
RSI_News_Corp_Class_A[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for NEWS_CORP_CLASS_A
STDEV_News_Corp_Class_A= calculate_stdev(News_Corp_Class_A.News_Corp_Class_A_Close)
STDEV_News_Corp_Class_A[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_News_Corp_Class_A=News_Corp_Class_A.News_Corp_Class_A_Open - News_Corp_Class_A.News_Corp_Class_A_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')



# In[50]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for NEWS_CORP_CLASS_B
SMA_News_Corp_Class_B = calculate_SMA(News_Corp_Class_B.News_Corp_Class_B_Close)

News_Corp_Class_B.News_Corp_Class_B_Close[:365].plot(title='News_Corp_Class_B',label='News_Corp_Class_B', ax=axes[0])

SMA_News_Corp_Class_B[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for NEWS_CORP_CLASS_B
upper_band_News_Corp_Class_B, lower_band_News_Corp_Class_B = calculate_BB(News_Corp_Class_B.News_Corp_Class_B_Close)

upper_band_News_Corp_Class_B[:365].plot(label='upper band', ax=axes[0])
lower_band_News_Corp_Class_B[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for NEWS_CORP_CLASS_B
DIF_News_Corp_Class_B, MACD_News_Corp_Class_B = calculate_MACD(News_Corp_Class_B.News_Corp_Class_B_Close)

DIF_News_Corp_Class_B[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_News_Corp_Class_B[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for NEWS_CORP_CLASS_B
RSI_News_Corp_Class_B = calculate_RSI(News_Corp_Class_B.News_Corp_Class_B_Close)
RSI_News_Corp_Class_B[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for NEWS_CORP_CLASS_B
STDEV_News_Corp_Class_B= calculate_stdev(News_Corp_Class_B.News_Corp_Class_B_Close)
STDEV_News_Corp_Class_B[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_News_Corp_Class_B=News_Corp_Class_B.News_Corp_Class_B_Open - News_Corp_Class_B.News_Corp_Class_B_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')



# In[51]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for OMNICOM_GROUP
SMA_Omnicom_Group = calculate_SMA(Omnicom_Group.Omnicom_Group_Close)

Omnicom_Group.Omnicom_Group_Close[:365].plot(title='Omnicom_Group',label='Omnicom_Group', ax=axes[0])

SMA_Omnicom_Group[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for OMNICOM_GROUP
upper_band_Omnicom_Group, lower_band_Omnicom_Group = calculate_BB(Omnicom_Group.Omnicom_Group_Close)

upper_band_Omnicom_Group[:365].plot(label='upper band', ax=axes[0])
lower_band_Omnicom_Group[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for OMNICOM_GROUP
DIF_Omnicom_Group, MACD_Omnicom_Group = calculate_MACD(Omnicom_Group.Omnicom_Group_Close)

DIF_Omnicom_Group[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Omnicom_Group[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for OMNICOM_GROUP
RSI_Omnicom_Group = calculate_RSI(Omnicom_Group.Omnicom_Group_Close)
RSI_Omnicom_Group[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for OMNICOM_GROUP
STDEV_Omnicom_Group= calculate_stdev(Omnicom_Group.Omnicom_Group_Close)
STDEV_Omnicom_Group[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Omnicom_Group=Omnicom_Group.Omnicom_Group_Open - Omnicom_Group.Omnicom_Group_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')



# In[52]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for TAKE_TWO_INTERACTIVE
SMA_Take_Two_Interactive = calculate_SMA(Take_Two_Interactive.Take_Two_Interactive_Close)

Take_Two_Interactive.Take_Two_Interactive_Close[:365].plot(title='Take_Two_Interactive',label='Take_Two_Interactive', ax=axes[0])

SMA_Take_Two_Interactive[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for TAKE_TWO_INTERACTIVE
upper_band_Take_Two_Interactive, lower_band_Take_Two_Interactive = calculate_BB(Take_Two_Interactive.Take_Two_Interactive_Close)

upper_band_Take_Two_Interactive[:365].plot(label='upper band', ax=axes[0])
lower_band_Take_Two_Interactive[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for TAKE_TWO_INTERACTIVE
DIF_Take_Two_Interactive, MACD_Take_Two_Interactive = calculate_MACD(Take_Two_Interactive.Take_Two_Interactive_Close)

DIF_Take_Two_Interactive[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Take_Two_Interactive[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for TAKE_TWO_INTERACTIVE
RSI_Take_Two_Interactive = calculate_RSI(Take_Two_Interactive.Take_Two_Interactive_Close)
RSI_Take_Two_Interactive[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for TAKE_TWO_INTERACTIVE
STDEV_Take_Two_Interactive= calculate_stdev(Take_Two_Interactive.Take_Two_Interactive_Close)
STDEV_Take_Two_Interactive[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Take_Two_Interactive=Take_Two_Interactive.Take_Two_Interactive_Open - Take_Two_Interactive.Take_Two_Interactive_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[53]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for TWITTER
SMA_Twitter = calculate_SMA(Twitter.Twitter_Close)

Twitter.Twitter_Close[:365].plot(title='Twitter',label='Twitter', ax=axes[0])

SMA_Twitter[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for TWITTER
upper_band_Twitter, lower_band_Twitter = calculate_BB(Twitter.Twitter_Close)

upper_band_Twitter[:365].plot(label='upper band', ax=axes[0])
lower_band_Twitter[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for TWITTER
DIF_Twitter, MACD_Twitter = calculate_MACD(Twitter.Twitter_Close)

DIF_Twitter[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Twitter[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for TWITTER
RSI_Twitter = calculate_RSI(Twitter.Twitter_Close)
RSI_Twitter[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for TWITTER
STDEV_Twitter= calculate_stdev(Twitter.Twitter_Close)
STDEV_Twitter[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Twitter=Twitter.Twitter_Open - Twitter.Twitter_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')



# In[54]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for TRIPADVISOR
SMA_TripAdvisor = calculate_SMA(TripAdvisor.TripAdvisor_Close)

TripAdvisor.TripAdvisor_Close[:365].plot(title='TripAdvisor',label='TripAdvisor', ax=axes[0])

SMA_TripAdvisor[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for TRIPADVISOR
upper_band_TripAdvisor, lower_band_TripAdvisor = calculate_BB(TripAdvisor.TripAdvisor_Close)

upper_band_TripAdvisor[:365].plot(label='upper band', ax=axes[0])
lower_band_TripAdvisor[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for TRIPADVISOR
DIF_TripAdvisor, MACD_TripAdvisor = calculate_MACD(TripAdvisor.TripAdvisor_Close)

DIF_TripAdvisor[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_TripAdvisor[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for TRIPADVISOR
RSI_TripAdvisor = calculate_RSI(TripAdvisor.TripAdvisor_Close)
RSI_TripAdvisor[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for TRIPADVISOR
STDEV_TripAdvisor= calculate_stdev(TripAdvisor.TripAdvisor_Close)
STDEV_TripAdvisor[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_TripAdvisor=TripAdvisor.TripAdvisor_Open - TripAdvisor.TripAdvisor_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[55]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for TWENTYFIRST_CENTURY_FOX_A
SMA_TwentyFirst_Century_Fox_A = calculate_SMA(TwentyFirst_Century_Fox_A.TwentyFirst_Century_Fox_A_Close)

TwentyFirst_Century_Fox_A.TwentyFirst_Century_Fox_A_Close[:365].plot(title='TwentyFirst_Century_Fox_A',label='TwentyFirst_Century_Fox_A', ax=axes[0])

SMA_TwentyFirst_Century_Fox_A[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for TWENTYFIRST_CENTURY_FOX_A
upper_band_TwentyFirst_Century_Fox_A, lower_band_TwentyFirst_Century_Fox_A = calculate_BB(TwentyFirst_Century_Fox_A.TwentyFirst_Century_Fox_A_Close)

upper_band_TwentyFirst_Century_Fox_A[:365].plot(label='upper band', ax=axes[0])
lower_band_TwentyFirst_Century_Fox_A[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for TWENTYFIRST_CENTURY_FOX_A
DIF_TwentyFirst_Century_Fox_A, MACD_TwentyFirst_Century_Fox_A = calculate_MACD(TwentyFirst_Century_Fox_A.TwentyFirst_Century_Fox_A_Close)

DIF_TwentyFirst_Century_Fox_A[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_TwentyFirst_Century_Fox_A[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for TWENTYFIRST_CENTURY_FOX_A
RSI_TwentyFirst_Century_Fox_A = calculate_RSI(TwentyFirst_Century_Fox_A.TwentyFirst_Century_Fox_A_Close)
RSI_TwentyFirst_Century_Fox_A[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for TWENTYFIRST_CENTURY_FOX_A
STDEV_TwentyFirst_Century_Fox_A= calculate_stdev(TwentyFirst_Century_Fox_A.TwentyFirst_Century_Fox_A_Close)
STDEV_TwentyFirst_Century_Fox_A[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_TwentyFirst_Century_Fox_A=TwentyFirst_Century_Fox_A.TwentyFirst_Century_Fox_A_Open - TwentyFirst_Century_Fox_A.TwentyFirst_Century_Fox_A_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')



# In[56]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for TWENTYFIRST_CENTURY_FOX_B
SMA_TwentyFirst_Century_Fox_B = calculate_SMA(TwentyFirst_Century_Fox_B.TwentyFirst_Century_Fox_B_Close)

TwentyFirst_Century_Fox_B.TwentyFirst_Century_Fox_B_Close[:365].plot(title='TwentyFirst_Century_Fox_B',label='TwentyFirst_Century_Fox_B', ax=axes[0])

SMA_TwentyFirst_Century_Fox_B[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for TWENTYFIRST_CENTURY_FOX_B
upper_band_TwentyFirst_Century_Fox_B, lower_band_TwentyFirst_Century_Fox_B = calculate_BB(TwentyFirst_Century_Fox_B.TwentyFirst_Century_Fox_B_Close)

upper_band_TwentyFirst_Century_Fox_B[:365].plot(label='upper band', ax=axes[0])
lower_band_TwentyFirst_Century_Fox_B[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for TWENTYFIRST_CENTURY_FOX_B
DIF_TwentyFirst_Century_Fox_B, MACD_TwentyFirst_Century_Fox_B = calculate_MACD(TwentyFirst_Century_Fox_B.TwentyFirst_Century_Fox_B_Close)

DIF_TwentyFirst_Century_Fox_B[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_TwentyFirst_Century_Fox_B[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for TWENTYFIRST_CENTURY_FOX_B
RSI_TwentyFirst_Century_Fox_B = calculate_RSI(TwentyFirst_Century_Fox_B.TwentyFirst_Century_Fox_B_Close)
RSI_TwentyFirst_Century_Fox_B[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for TWENTYFIRST_CENTURY_FOX_B
STDEV_TwentyFirst_Century_Fox_B= calculate_stdev(TwentyFirst_Century_Fox_B.TwentyFirst_Century_Fox_B_Close)
STDEV_TwentyFirst_Century_Fox_B[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_TwentyFirst_Century_Fox_B=TwentyFirst_Century_Fox_B.TwentyFirst_Century_Fox_B_Open - TwentyFirst_Century_Fox_B.TwentyFirst_Century_Fox_B_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[57]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for VERIZON_COMMUNICATIONS
SMA_Verizon_Communications = calculate_SMA(Verizon_Communications.Verizon_Communications_Close)

Verizon_Communications.Verizon_Communications_Close[:365].plot(title='Verizon_Communications',label='Verizon_Communications', ax=axes[0])

SMA_Verizon_Communications[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for VERIZON_COMMUNICATIONS
upper_band_Verizon_Communications, lower_band_Verizon_Communications = calculate_BB(Verizon_Communications.Verizon_Communications_Close)

upper_band_Verizon_Communications[:365].plot(label='upper band', ax=axes[0])
lower_band_Verizon_Communications[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for VERIZON_COMMUNICATIONS
DIF_Verizon_Communications, MACD_Verizon_Communications = calculate_MACD(Verizon_Communications.Verizon_Communications_Close)

DIF_Verizon_Communications[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Verizon_Communications[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for VERIZON_COMMUNICATIONS
RSI_Verizon_Communications = calculate_RSI(Verizon_Communications.Verizon_Communications_Close)
RSI_Verizon_Communications[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for VERIZON_COMMUNICATIONS
STDEV_Verizon_Communications= calculate_stdev(Verizon_Communications.Verizon_Communications_Close)
STDEV_Verizon_Communications[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Verizon_Communications=Verizon_Communications.Verizon_Communications_Open - Verizon_Communications.Verizon_Communications_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')


# In[58]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for VIACOM
SMA_Viacom = calculate_SMA(Viacom.Viacom_Close)

Viacom.Viacom_Close[:365].plot(title='Viacom',label='Viacom', ax=axes[0])

SMA_Viacom[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for VIACOM
upper_band_Viacom, lower_band_Viacom = calculate_BB(Viacom.Viacom_Close)

upper_band_Viacom[:365].plot(label='upper band', ax=axes[0])
lower_band_Viacom[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for VIACOM
DIF_Viacom, MACD_Viacom = calculate_MACD(Viacom.Viacom_Close)

DIF_Viacom[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Viacom[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for VIACOM
RSI_Viacom = calculate_RSI(Viacom.Viacom_Close)
RSI_Viacom[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for VIACOM
STDEV_Viacom= calculate_stdev(Viacom.Viacom_Close)
STDEV_Viacom[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Viacom=Viacom.Viacom_Open - Viacom.Viacom_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')



# In[59]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for THE_WALT_DISNEY
SMA_The_Walt_Disney = calculate_SMA(The_Walt_Disney.The_Walt_Disney_Close)

The_Walt_Disney.The_Walt_Disney_Close[:365].plot(title='The_Walt_Disney',label='The_Walt_Disney', ax=axes[0])

SMA_The_Walt_Disney[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for THE_WALT_DISNEY
upper_band_The_Walt_Disney, lower_band_The_Walt_Disney = calculate_BB(The_Walt_Disney.The_Walt_Disney_Close)

upper_band_The_Walt_Disney[:365].plot(label='upper band', ax=axes[0])
lower_band_The_Walt_Disney[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for THE_WALT_DISNEY
DIF_The_Walt_Disney, MACD_The_Walt_Disney = calculate_MACD(The_Walt_Disney.The_Walt_Disney_Close)

DIF_The_Walt_Disney[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_The_Walt_Disney[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for THE_WALT_DISNEY
RSI_The_Walt_Disney = calculate_RSI(The_Walt_Disney.The_Walt_Disney_Close)
RSI_The_Walt_Disney[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for THE_WALT_DISNEY
STDEV_The_Walt_Disney= calculate_stdev(The_Walt_Disney.The_Walt_Disney_Close)
STDEV_The_Walt_Disney[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_The_Walt_Disney=The_Walt_Disney.The_Walt_Disney_Open - The_Walt_Disney.The_Walt_Disney_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')



# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for ADVANCE_AUTO_PARTS
SMA_Advance_Auto_Parts = calculate_SMA(Advance_Auto_Parts.Advance_Auto_Parts_Close)

Advance_Auto_Parts.Advance_Auto_Parts_Close[:365].plot(title='Advance_Auto_Parts',label='Advance_Auto_Parts', ax=axes[0])

SMA_Advance_Auto_Parts[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for ADVANCE_AUTO_PARTS
upper_band_Advance_Auto_Parts, lower_band_Advance_Auto_Parts = calculate_BB(Advance_Auto_Parts.Advance_Auto_Parts_Close)

upper_band_Advance_Auto_Parts[:365].plot(label='upper band', ax=axes[0])
lower_band_Advance_Auto_Parts[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for ADVANCE_AUTO_PARTS
DIF_Advance_Auto_Parts, MACD_Advance_Auto_Parts = calculate_MACD(Advance_Auto_Parts.Advance_Auto_Parts_Close)

DIF_Advance_Auto_Parts[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Advance_Auto_Parts[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for ADVANCE_AUTO_PARTS
RSI_Advance_Auto_Parts = calculate_RSI(Advance_Auto_Parts.Advance_Auto_Parts_Close)
RSI_Advance_Auto_Parts[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for ADVANCE_AUTO_PARTS
STDEV_Advance_Auto_Parts= calculate_stdev(Advance_Auto_Parts.Advance_Auto_Parts_Close)
STDEV_Advance_Auto_Parts[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Advance_Auto_Parts=Advance_Auto_Parts.Advance_Auto_Parts_Open - Advance_Auto_Parts.Advance_Auto_Parts_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')




# In[2]:


Fix_Matel = quandl.get('PERTH/LONMETALS', start_date="2006-10-20", end_date="2013-10-20")

Fix_Matel

#indexing
Fix_Matel.reset_index(inplace=True) ##Communication Services


#Data Management
Gold = Fix_Matel [['Date','Gold AM Fix','Gold PM Fix']]
Gold.rename(columns={'Gold AM Fix': 'Gold_Open'},inplace=True)
Gold.rename(columns={'Gold PM Fix': 'Gold_Close'},inplace=True)

Platinum = Fix_Matel [['Date','Platinum AM Fix','Platinum PM Fix']]
Platinum.rename(columns={'Platinum AM Fix': 'Platinum_Open'},inplace=True)
Platinum.rename(columns={'Platinum PM Fix': 'Platinum_Close'},inplace=True)

Palladium = Fix_Matel [['Date','Palladium AM Fix','Palladium PM Fix']]
Palladium.rename(columns={'Palladium AM Fix': 'Palladium_Open'},inplace=True)
Palladium.rename(columns={'Palladium PM Fix': 'Palladium_Close'},inplace=True)


# In[3]:


Commodities= (Gold).merge(Platinum).merge(Palladium)
Commodities.to_csv('Commodities.csv')


# In[7]:





# In[11]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

# Calculate Simple Moving Average for GOLD
SMA_Gold = calculate_SMA(Gold.Gold_Close)

Gold.Gold_Close[:365].plot(title='Gold',label='Gold', ax=axes[0])

SMA_Gold[:365].plot(label="SMA",ax=axes[0])


# Calculate Bollinger Bands for GOLD
upper_band_Gold, lower_band_Gold = calculate_BB(Gold.Gold_Close)

upper_band_Gold[:365].plot(label='upper band', ax=axes[0])
lower_band_Gold[:365].plot(label='lower band', ax=axes[0])


# Calculate MACD for GOLD
DIF_Gold, MACD_Gold = calculate_MACD(Gold.Gold_Close)

DIF_Gold[:365].plot(title='DIF and MACD',label='DIF', ax=axes[1])
MACD_Gold[:365].plot(label='MACD', ax=axes[1])

# Calculate RSI for GOLD
RSI_Gold = calculate_RSI(Gold.Gold_Close)
RSI_Gold[:365].plot(title='RSI',label='RSI', ax=axes[2])

# Calculating Standard deviation for GOLD
STDEV_Gold= calculate_stdev(Gold.Gold_Close)
STDEV_Gold[:365].plot(title='STDEV',label='STDEV', ax=axes[3])

Open_Close_Gold=Gold.Gold_Open - Gold.Gold_Close

#High_Low=df_final.High-df_final.Low

axes[0].set_ylabel('Price')
axes[1].set_ylabel('Price')
axes[2].set_ylabel('Price')
axes[3].set_ylabel('Price')



axes[0].legend(loc='lower left')
axes[1].legend(loc='lower left')
axes[2].legend(loc='lower left')
axes[3].legend(loc='lower left')






#Gold
SP500_tech_indicator['SMA_Gold'] = SMA_Gold
SP500_tech_indicator['upper_band_Gold'] = upper_band_Gold
SP500_tech_indicator['lower_band_Gold'] = lower_band_Gold
SP500_tech_indicator['DIF_Gold'] = DIF_Gold
SP500_tech_indicator['MACD_Gold'] = MACD_Gold
SP500_tech_indicator['RSI_Gold'] = RSI_Gold
SP500_tech_indicator['STDEV_Gold'] = STDEV_Gold
SP500_tech_indicator['Open_Close_Gold']=Open_Close_Gold


# In[9]:


#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
from functools import reduce



Activision_Blizzard['Open'].plot(label='Activision Blizzard',figsize=(24,12),title='Open Price')
Alphabet_Inc_Class_C['Open'].plot(label='Alphabet Inc Class C')
Alphabet_Inc_Class_A['Open'].plot(label='Alphabet Inc Class A')
ATT['Open'].plot(label='AT&T')
CBS['Open'].plot(label='CBS')
CenturyLink['Open'].plot(label='CenturyL ink')
Charter_Communications['Open'].plot(label='Charter Communications')
Comcast['Open'].plot(label='Comcast')
Discovery_Class_A['Open'].plot(label='Discovery Class A ')
Charter_Communications['Open'].plot(label='Charter Communications')
plt.legend()







# In[10]:


Activision_Blizzard['Volume'].argmax()


# In[40]:


#Activision_Blizzard = Activision_Blizzard[['Date','Close']]
ATT


# In[11]:


Activision_Blizzard['MA50'] = Activision_Blizzard['Open'].rolling(50).mean()
Activision_Blizzard['MA200'] = Activision_Blizzard['Open'].rolling(200).mean()
Activision_Blizzard[['Open','MA50','MA200']].plot(label='Activision Blizzard',figsize=(16,8))

Alphabet_Inc_Class_C['MA50'] = Alphabet_Inc_Class_C['Open'].rolling(50).mean()
Alphabet_Inc_Class_C['MA200'] = Alphabet_Inc_Class_C['Open'].rolling(200).mean()
Alphabet_Inc_Class_C[['Open','MA50','MA200']].plot(label='Alphabet Inc Class C',figsize=(16,8))

Alphabet_Inc_Class_A['MA50'] = Alphabet_Inc_Class_A['Open'].rolling(50).mean()
Alphabet_Inc_Class_A['MA200'] = Alphabet_Inc_Class_A['Open'].rolling(200).mean()
Alphabet_Inc_Class_A[['Open','MA50','MA200']].plot(label='Alphabet Inc Class A',figsize=(16,8))

ATT['MA50'] = ATT['Open'].rolling(50).mean()
ATT['MA200'] = ATT['Open'].rolling(200).mean()
ATT[['Open','MA50','MA200']].plot(label='AT&T',figsize=(16,8))

CBS['MA50'] = CBS['Open'].rolling(50).mean()
CBS['MA200'] = CBS['Open'].rolling(200).mean()
CBS[['Open','MA50','MA200']].plot(label='CBS',figsize=(16,8))

CenturyLink['MA50'] = CenturyLink['Open'].rolling(50).mean()
CenturyLink['MA200'] = CenturyLink['Open'].rolling(200).mean()
CenturyLink[['Open','MA50','MA200']].plot(label='Century Link',figsize=(16,8))

Charter_Communications['MA50'] = Charter_Communications['Open'].rolling(50).mean()
Charter_Communications['MA200'] = Charter_Communications['Open'].rolling(200).mean()
Charter_Communications[['Open','MA50','MA200']].plot(label='Charter Communications',figsize=(16,8))

Comcast['MA50'] = Comcast['Open'].rolling(50).mean()
Comcast['MA200'] = Comcast['Open'].rolling(200).mean()
Comcast[['Open','MA50','MA200']].plot(label='Comcast',figsize=(16,8))



Discovery_Class_A['MA50'] = Discovery_Class_A['Open'].rolling(50).mean()
Discovery_Class_A['MA200'] = Discovery_Class_A['Open'].rolling(200).mean()
Discovery_Class_A[['Open','MA50','MA200']].plot(label='Discovery Class A',figsize=(16,8))

Discovery_Class_C['MA50'] = Discovery_Class_C['Open'].rolling(50).mean()
Discovery_Class_C['MA200'] = Discovery_Class_C['Open'].rolling(200).mean()
Discovery_Class_C[['Open','MA50','MA200']].plot(label='Discovery Class C',figsize=(16,8))


# In[79]:




