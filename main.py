import camelot
import pandas
import matplotlib
import datetime
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# parse the time table in our columns to have a working data format as an intermediary
def parseTimeTableToDF(filename,timetable):

    # Working dataframe for timetable
    df = pandas.DataFrame(columns=["weekday","daynumber","start_time","end_time","type","calculated_amount","glz_saldo"])
    for k,row in timetable.iterrows():
        # Parse info from table into our working dataframe
        row_object = {
            "source": filename.split("/")[-1], # get the last filename with .pdf ending
            "weeknum": None,
            "weekday": row[0],
            "year" : None,
            "daynumber": row[1],
            "start_time": row[2],
            "end_time" : row[3],
            "type" : row[4],
            "calculated_amount": row[5],#typecast for later calculation
            "glz_saldo": row[6] #typecast for later calculation
        }
        # We are not interested in Weekends for our timetable
        if row_object["weekday"] not in ["Sa","So"]:
            df_dictionary = pandas.DataFrame([row_object])
            df = pandas.concat([df,df_dictionary],ignore_index=True)
    return df

# Parse all calulated times into integers
def parseTimeTableAmount(amount):
    # we only try to convert if we have a string from the dataframe
    if isinstance(amount, str) and amount != "":
        amount = amount.replace(",",".")
        if '-' in amount:
            amount = amount.replace("-","")
            return -float(amount)
        else:
            return float(amount)
    else:
        return None


def fill_special_days(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Replaces 'SPECIAL_DAY' in the 'daynumber' column with the previous non-'SPECIAL_DAY' value + 1.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    prev_value = "01"
    for i,row in df.iterrows():
        if row["daynumber"] == 'SPECIAL_DAY':
            df.at[i,"daynumber"] = int(prev_value) + 1
            prev_value = int(prev_value) + 1 # set the previous value to our calculated stuff
        else:
            # Cast daynumber into integer
            df.at[i,"daynumber"] = int(row["daynumber"])
            # Set previous value for next stepp
            prev_value = row["daynumber"]

    return df

# def fixCalculatedAmountForFZGZ(entry):
#     if entry != None and entry[0] != None and entry[1] != None:
#         # If we have a FZGZ entry, the time is reversed (only if we have a valid time)
#         if "FZ-" in entry[0]:
#             entry[1] = -entry[1]
#     return entry

def fixCalculatedAmountForFZGZ(row):
    amount = str(row["calculated_amount"]).replace(",", ".")  # Ensure decimal format
    try:
        value = float(amount) if amount not in ["None", "nan"] else None
        if pandas.notna(value) and "FZ-" in str(row["type"]):
            return -value  # Make negative if "FZ-" is in type
        return value
    except ValueError:
        return None  # Return None for invalid entries

def getCurrentDatesFromFilename(file_datum):
    # Calculate the current week of the year from the filename
    file_year = int(file_datum[:4])
    file_month = int(file_datum[4:6])
    file_day = int(file_datum[6:])
    # Calculate iso-calender datum with these infos from the filename
    return file_year,file_month,file_day

# Use global information from the file to get the week_number
def calculateWeekNumFromWeekday(entry):
    source_name = entry.iloc[2].strip(".pdf") # second entry is source filename
    date_infos = getCurrentDatesFromFilename(source_name)
    day = int(entry.iloc[0])
    entry.iloc[1] = datetime.date(date_infos[0], date_infos[1],day).isocalendar().week
    return entry

def convert24hoursToTimeObject(obj):
    if not None:
        # Converts the format HH:MM to a datetime object
        return datetime.datetime.strptime(obj, r"%H:%M")
    else:
        return None

def fill_double_entrys(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    If we have a row without a weekday, daynumber it mostly corresponds to the row above it
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    for i,row in df.iterrows():
        if row["weekday"] == None and row["daynumber"] == None and row["start_time"] != None: # check if both entrys are empty
            # Check if previous entrys are full
            if df.loc[i-1]['weekday'] != None and df.loc[i-1]["daynumber"] != None:

                # We then merge the backfill the empty entry with information from the field and fix the time
                # df.at[i,"weekday"] = df.iloc[i-1]['weekday']
                df.at[i,"weekday"] = df.loc[i-1]['weekday']
                df.at[i,"daynumber"] = df.loc[i-1]['daynumber']
    return df

def parsePDFFromFile(filename):
    # Sane Defaults for PDF Documents
    timetable_area = ['40,620,400,300']
    # table_areas accepts strings of the form x1,y1,x2,y2 where (x1, y1) -> top-left and (x2, y2) -> bottom-right in PDF coordinate space.
    timetable_columm_settings = ['50,68,95,122,220,255,292,325,360']

    # Starting table parsing
    tables = camelot.read_pdf(filename,flavor="stream",split_text=True, columns=timetable_columm_settings,table_regions =timetable_area )
    return tables[0].df

def generalDataFrameCleanup(timetable):
    # Search for index with keyword "Beginn"
    idx = timetable.index[timetable["start_time"].eq("Beginn")].min()
    timetable = timetable.iloc[idx+1:] # Drop all indexes before "Beginn" to cleanup table

    # Drop all indexes after "Anspru" because this indicates the end of the table
    idx = timetable.index[timetable["type"].str.contains("Anspru")].min()
    timetable = timetable.iloc[:idx-2] # Drop all indexes behind found index

    # replace field that's entirely space (or empty) with NaN
    timetable.replace(r'^\s*$', None, regex=True,inplace=True)

    # Convert all string times to time objects so we can work with time
    timetable['start_time'] = pandas.to_datetime(timetable['start_time'], format='%H:%M')

    timetable['end_time'] = pandas.to_datetime(timetable['end_time'], format='%H:%M')

    # Fix empty rows that are created because we have two different time types on the same day
    timetable = fill_double_entrys(timetable)

    # If we do not have a number in field "daynumber, remove the
    timetable["daynumber"] = timetable["daynumber"].replace(r'^[a-zA-Z].*', 'SPECIAL_DAY', regex=True)
    # All weekdays which are not valid days of the week will be removed
    timetable["weekday"] = timetable["weekday"].apply(lambda x: "SPECIAL_DAY" if x not in ["","Mo","Di","Mi","Do","Fr","Sa","So"] else x)

    # We fill in the daynumber of the specialdays by using the daynumber in the row before +1
    timetable = fill_special_days(timetable)

    # Calculate the week number of the year based on the daynum in the table
    timetable[['daynumber','weeknum',"source"]] = timetable[['daynumber','weeknum','source']].apply(calculateWeekNumFromWeekday,axis=1)

    # Set the year for all weeks
    date_infos = getCurrentDatesFromFilename(timetable['source'].iloc[0].strip(".pdf"))
    timetable["year"] = date_infos[0]

    # Fill NaN Columns of Type of Work with Büro
    timetable = timetable.assign(type=timetable['type'].fillna("Buero"))

    # Parse GLZ and calculated numbers into integers
    timetable['calculated_amount'] = timetable['calculated_amount'].apply(parseTimeTableAmount)
    timetable['glz_saldo'] = timetable['glz_saldo'].apply(parseTimeTableAmount)

    # FIx positive/negative direction of integer for FZGZ
    timetable["calculated_amount"] = timetable.apply(fixCalculatedAmountForFZGZ, axis=1)  # Apply row-wise

    return timetable

def performeDataAnalysis(yearly_timetable):
    # Generalize dataframes
    yearly_timetable_without_fzgz = yearly_timetable[yearly_timetable['type'] != ("FZ-Ausgl. GLZ/AZV")]

    # Summarize the amount of work did in each week!

    st.subheader("Let's take a look at your work! ")
    st.text("This is how much you worked in each week. That's a lot")
    hours_worked_each_week = yearly_timetable.groupby("weeknum")[['calculated_amount']].sum()
    st.bar_chart(hours_worked_each_week)


    hourly_rate_in_eur = st.slider(
        "Your Hourly rate in €?", value=40
    )
    weekly_work_hours_baseline = st.slider(
        "How many hours do you work in a week?", value=16
    )

    st.subheader("Your work in numbers:")
    st.text(f"Have you ever wonderd how many netflix shows you could have watched at work? How many basejumps could have been bought if all your money went to them (after tax)? Did you work your required {weekly_work_hours_baseline} hours?")
    total_col1, total_col2, total_col3 = st.columns(3)


    # Total Hours Worked
    total_hours_worked = yearly_timetable["calculated_amount"].sum()
    netflix_epsiode_in_hours = 0.75 * 10 # 45min * 10 episodes

    total_col1.metric(
        label="Work hours in Netflix shows",
        value=round(total_hours_worked / netflix_epsiode_in_hours,1)
    )

    # Based on your hourly rate, you could have bought x amount of Fallschirmsprünge from your worked hours

    # hourly_rate_in_eur = 40
    cost_of_basejump = 400 # From jochen schweizer
    tax_rate_in_percent = 30 # we assume the goverment takes a lot of our money

    amount_of_basejumps = ((total_hours_worked * hourly_rate_in_eur) * ((100-tax_rate_in_percent)/100)) / cost_of_basejump
    # st.text(f"You could have bought {round(amount_of_basejumps,2)} basejumps after tax with the hours you have worked")

    total_col2.metric(
        label="Bought Basejumps",
        value=round(amount_of_basejumps,2) 
    )


    # You gave X % percent this month (based on a weekly_work_hours_baseline work week)
    amount_of_weeks_worked = len(yearly_timetable["weeknum"].unique())
    percent_given = yearly_timetable["calculated_amount"].sum() / (weekly_work_hours_baseline * amount_of_weeks_worked)

    # print(f"Based on a {weekly_work_hours_baseline} hour work-week, you have worked {percent_given*100}% of that time in the analysed timespan!")

    total_col3.metric(
        label="Percentage of expected work hours",
        value=str(round(percent_given*100,1))+"%"
    )


    # Overtime Champion: How many extra hours did you give to the office this year? Are you eligible for an overtime hall of fame? 🏆
    st.divider()
    st.subheader("Are you the overtime champion?")
    overtime_worked = yearly_timetable["calculated_amount"].sum() - (weekly_work_hours_baseline * amount_of_weeks_worked)
    if overtime_worked > 0:
        st.text(f"You gave a total of {round(overtime_worked,1)} hours of overtime to the office")
    else:
        st.text(f'No overtime for you. You valued your personal life more than work life!')


    # Most active day and least active day
    def calculateTimeDeltaForTimes(row):
        return row["end_time"] - row["start_time"]

    def days_hours_minutes(td):
        return td.days, td.seconds//3600, (td.seconds//60)%60

    # Create a filtered list of timedeltas without NaT values
    timedeltas = filter(lambda x: str(x) != 'NaT', list(yearly_timetable.apply(calculateTimeDeltaForTimes,axis=1)))
    timedeltas = list(timedeltas)


    timedelta_col1, timedelta_col2 = st.columns(2)

    timedelta_col1.metric(
        label="Least active day",
        value=f"{days_hours_minutes(min(timedeltas))[1]}h and {days_hours_minutes(max(timedeltas))[2]}min",
        delta="-Chill out",
        border=True
    )

    timedelta_col2.metric(
        label="Most active day",
        value=f"{days_hours_minutes(max(timedeltas))[1]}h and {days_hours_minutes(max(timedeltas))[2]}min",
        delta="+Hustler",
        border=True
    )

    # Work-Life Balance Score: Did your time tracking scream “workaholic” or “master of balance”? ⚖️
    # https://www.destatis.de/EN/Themes/Labour/Labour-Market/Quality-Employment/Dimension3/3_1_WeeklyHoursWorked.html
    germany_average_worked_hours_per_week = 16 # are you over 40hours in most weeks?

    # Calculate the hours per week of the data
    hours_worked_each_week = yearly_timetable.groupby("weeknum")[['calculated_amount']].sum()
    overworked_hours = hours_worked_each_week[hours_worked_each_week > germany_average_worked_hours_per_week].count()

    if overworked_hours["calculated_amount"] > 3:
        st.text(f'🤠 Workaholic! Out of {len(hours_worked_each_week)} weeks, you did overtime in {overworked_hours["calculated_amount"]} weeks')
    else:
        st.text(f'🧘 Master of balance! Out of {len(hours_worked_each_week)} weeks, you did overtime in {overworked_hours["calculated_amount"]} weeks')

    # Most Productive Week: If your weeks were competing in the Work Olympics, which one would take home the gold medal? 🥇
    index_of_week_with_max_hours = hours_worked_each_week['calculated_amount'].idxmax()

    if not hours_worked_each_week.empty:
        index_of_week_with_max_hours = hours_worked_each_week['calculated_amount'].idxmax()
        if pandas.notna(index_of_week_with_max_hours):  # Ensure it's not NaN
            week = hours_worked_each_week.loc[index_of_week_with_max_hours]  # Use .loc instead of .iloc
            st.write(f'🥇 Most Productive Week: If your weeks were competing in the Work Olympics, week {index_of_week_with_max_hours} would have won with {week['calculated_amount']} hours')
        else:
            print("All your weeks are the same. No highlights here :-(")

    # "Peak Rage Quit Hour" 😡 At what time were you most likely to log off in frustration? Should we start a support group for the 3:57 PM Ragers?
    yearly_timetable['end_time'].describe()

    df_leaving_times = pandas.DataFrame()
    df_leaving_times['hour'] = yearly_timetable_without_fzgz['end_time'].dt.hour
    df_leaving_times['minute'] = yearly_timetable_without_fzgz['end_time'].dt.minute

    # Group by hour and minute
    common_times = df_leaving_times.groupby(['hour', 'minute']).size().reset_index(name='count')

    # Find the most common log-off time
    most_common = common_times.sort_values(by='count', ascending=False)
    logoff_time = f'{round(most_common.iloc[0].hour):02}:{round(most_common.iloc[0].minute):02}'


    hours_start_end_col1,hours_start_end_col2 = st.columns(2)
    st.text(f"At {logoff_time} you were most likely to log off in frustration? Should we start a support group for the {logoff_time} PM Ragers?")
    hours_start_end_col2.metric(
        label='Peak Rage Quit Hour😡',
        value=logoff_time,
        border=True
    )

    # Early bird or long sleeper? When did you usally start work?
    df_start_time = pandas.DataFrame()
    df_start_time['hour'] = yearly_timetable_without_fzgz['start_time'].dt.hour
    df_start_time['minute'] = yearly_timetable_without_fzgz['start_time'].dt.minute

    # Group by hour and minute
    common_times_staring = df_start_time.groupby(['hour', 'minute']).size().reset_index(name='count')
    most_common_start = common_times_staring.sort_values(by='count', ascending=False)
    logon_time = f'{round(most_common_start.iloc[0].hour):02}:{round(most_common_start.iloc[0].minute):02}'

    hours_start_end_col1.metric(
        label='⏲️ Avg wakeup time',
        value=logon_time,
        border=True
    )


    # "Your Work Theme Song"
    # Based on your productivity patterns, should your anthem be:
    st.divider()
    st.subheader("Your Work Theme Song:")
    st.text("Based on your overtime patterns it should be")

    if overworked_hours["calculated_amount"] > 5:
        # Taking care of buisness 
        components.iframe("https://open.spotify.com/embed/album/0cvlJoDkIHaxmOw9dUMhOk?utm_source=generator")
    elif overworked_hours["calculated_amount"] > 3:
        # Work - Rihanna ft. Drake"
        components.iframe("https://open.spotify.com/embed/album/4UlGauD7ROb3YbVOFMgW5u?utm_source=generator")
    else:
        #  "9 to 5 - Dolly Parton"
        components.iframe("https://open.spotify.com/embed/track/2CCPoiBpNX9on3SIu78RJx?utm_source=generator")


    # 🏝️ Vacation MVP:
    # You took 4 days off in one go—are you a master of strategic leave planning, or did you just really need a break?

    df_vaction_calc = pandas.DataFrame()
    df_vaction_calc["is_urlaub"] = yearly_timetable["type"] == "Urlaub"

    # Identify consecutive sequences using cumsum on changes
    df_vaction_calc["group"] = (df_vaction_calc["is_urlaub"] != df_vaction_calc["is_urlaub"].shift()).cumsum()

    urlaub_groups = (
        df_vaction_calc[df_vaction_calc["is_urlaub"]]
        .groupby("group", group_keys=False)["is_urlaub"]
        .apply(lambda x: (x.index[0], x.index[-1]))
    )

    # Find the longest sequence
    longest_sequence = max(urlaub_groups, key=lambda x: x[1] - x[0], default=None)
    if longest_sequence != None:
        sequence_length = longest_sequence[1] - longest_sequence[0] + 1
        st.text(f"🏝️ Vacation MVP: You took {sequence_length} days off in one go—are you a master of strategic leave planning, or did you just really need a break?")
    else:
        st.text(f"🏝️ No vacation days logged in the data. You sure you don't wanna go to the beach? ")


    # This was your favorite time of time entry (e.g type filtering)
    favorite_type_count = yearly_timetable['type'].value_counts()
    st.bar_chart(favorite_type_count)

    # Define holiday types
    holiday_types = ["Urlaub", "FZ-Ausgl. GLZ/AZV"]

    # Filter for Friday and Monday rows where the user was on holiday.
    fridays = yearly_timetable[(yearly_timetable["weekday"] == "Fr") & (yearly_timetable["type"].isin(holiday_types))].copy()
    mondays = yearly_timetable[(yearly_timetable["weekday"] == "Mo") & (yearly_timetable["type"].isin(holiday_types))].copy()

    # For Monday rows, calculate the previous week number.
    mondays["prev_weeknum"] = mondays["weeknum"] - 1

    # Merge: match a Friday with a Monday where Monday's previous week equals Friday's weeknum and the year is the same.
    long_weekends = fridays.merge(mondays, left_on=["weeknum", "year"], right_on=["prev_weeknum", "year"], suffixes=("_fr", "_mo"))


    weekends_glz_col1, weekends_glz_col2 = st.columns(2)

    if not long_weekends.empty:
        weekends_glz_col1.metric(
            value=str(len(long_weekends)),
            label="Long weekends!",
            border=True
        )
    else:
        weekends_glz_col1.metric(
            value="0",
            label="Long weekends!",
            border=True
        )

    # 📊 "GLZ_Saldo: 0.00"
    # You maintained a perfect balance—were with overtime in the calender week x you a time management wizard, or did you just make sure HR wouldn't notice?

    hours_week_upper_lower_tolerance = 1

    hours_week_glz_balance = (hours_worked_each_week < weekly_work_hours_baseline+hours_week_upper_lower_tolerance) & (hours_worked_each_week > weekly_work_hours_baseline-hours_week_upper_lower_tolerance)

    # Check if there is any True in the column
    if hours_week_glz_balance['calculated_amount'].any():
        result = hours_week_glz_balance.loc[hours_week_glz_balance['calculated_amount']].iloc[0]
    else:
        result = hours_week_glz_balance.iloc[0]

    if result['calculated_amount'] == True:
        st.text(f"📊GLZ_Saldo: 0.00. You maintained a near perfect balance—were with overtime in calender week {result.name}. You are a time management wizard, or did you just make sure HR wouldn't notice?")
    else:
        st.text(f"📊GLZ_Saldo: 0.00. You never achieved a week without overtime? Try to spend some time with your family")

    # Count occurrences of each unique combination
    combination_counts = yearly_timetable.groupby(["start_time", "end_time", "type"]).size().reset_index(name="count")

    # Identify rarest combinations (outliers)
    outliers = combination_counts[combination_counts["count"] == 1]  # Appears only once

    st.text("According to our advanced machine learning algorithms: These entrys look the most like Arbeitszeitbetrug!")
    st.dataframe(outliers.head(3))

    st.subheader("Your raw data!")
    st.text("Take a look and see if we got anything wrong. You can even download it as a .csv for your own analytics in excel")
    st.dataframe(yearly_timetable_without_fzgz)



# Streamlit setup
st.set_page_config(
    page_title="Timewatcher 3000",
    page_icon=":shark:"
)
st.title("Timewatcher")


uploaded_files = st.file_uploader(
    "Choose a SAP Zeitnachweis file", accept_multiple_files=True
)

yearly_timetable = pandas.DataFrame()

for doc in uploaded_files:
    filename = doc.name
    camelot_df = parsePDFFromFile(doc)
    timetable = parseTimeTableToDF(filename,camelot_df)
    timetable = generalDataFrameCleanup(timetable)

    yearly_timetable = pandas.concat([yearly_timetable,timetable],ignore_index=True)

if len(uploaded_files) != 0: # Dataset is empty e.g. we have not uploaded anything yet 
    st.text("Loading and parsing of data started!")
    performeDataAnalysis(yearly_timetable)
