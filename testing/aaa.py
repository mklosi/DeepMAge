from datetime import timedelta, datetime

roise0r_td_strs = [
    "17:06:12.519679",
    "16:33:06.926266",
    "16:23:35.040880",
    "16:25:28.239276",
    "16:19:45.533524",
    "17:06:31.793268",
    "16:28:27.468085",
    "17:07:22.677105",
]

mklosi_td_strs = [
    "5:42:48.890267",
    "5:49:00.338472",
    "7:14:39.061912",
    "4:58:29.227970",
    "4:38:42.146232",
    "4:54:36.224949",
    "5:43:35.217828",
    "5:43:23.430778",
]

tds = []
for td_str in roise0r_td_strs:
    dt = datetime.strptime(td_str, "%H:%M:%S.%f")
    td = timedelta(
        hours=dt.hour, minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond
    )
    tds.append(td)

total = sum(tds, timedelta())

ave_td = total / len(tds)
print(ave_td)



fdjkfdjkd = 1
