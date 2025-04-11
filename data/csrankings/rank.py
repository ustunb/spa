import pandas as pd
import numpy as np

# Input data as a string
data = """
item_name,AI-ML-data-mining,AI-NLP,AI-Vision,AI-Web-Retrieval,AI-just-artificial-intelligence
Arizona State University,58.0,25.0,34.0,28.0,4.0
Auburn University,82.0,96.0,117.0,120.0,139.0
Augusta University,115.0,,,,139.0
Binghamton University,84.0,81.0,68.0,65.0,49.0
Boise State University,137.0,83.0,,72.0,149.0
Boston College,90.0,77.0,85.0,,132.0
Boston University,16.0,65.0,14.0,23.0,90.0
Brandeis University,87.0,53.0,80.0,77.0,78.0
Brigham Young University,105.0,69.0,100.0,77.0,76.0
Brown University,43.0,31.0,31.0,35.0,43.0
CUNY,107.0,66.0,91.0,99.0,87.0
California Institute of Technology,55.0,,42.0,86.0,121.0
Carnegie Mellon University,1.0,2.0,1.0,5.0,1.0
Case Western Reserve University,98.0,75.0,,77.0,89.0
Clemson University,137.0,,78.0,,114.0
Cleveland State University,146.0,,95.0,,130.0
College of William and Mary,101.0,120.0,103.0,91.0,125.0
Colorado School of Mines,107.0,,87.0,86.0,72.0
Colorado State University,137.0,,76.0,,130.0
Columbia University,12.0,15.0,13.0,24.0,25.0
Cornell University,6.0,3.0,9.0,2.0,8.0
Dartmouth College,88.0,69.0,117.0,86.0,117.0
DePaul University,125.0,96.0,,,112.0
Drexel University,96.0,87.0,87.0,,83.0
Duke University,17.0,38.0,35.0,52.0,30.0
Emory University,75.0,50.0,113.0,11.0,70.0
Florida Atlantic University,,,,,148.0
Florida International University,135.0,81.0,87.0,,123.0
Florida State University,95.0,103.0,117.0,120.0,98.0
George Mason University,61.0,49.0,48.0,42.0,33.0
George Washington University,,46.0,76.0,95.0,123.0
Georgetown University,117.0,47.0,,56.0,114.0
Georgia Institute of Technology,23.0,20.0,15.0,12.0,32.0
Georgia State University,100.0,113.0,112.0,,
Harvard University,26.0,67.0,57.0,45.0,2.0
IUPUI,85.0,,127.0,,139.0
Illinois Institute of Technology,107.0,88.0,73.0,57.0,95.0
Indiana University,51.0,88.0,64.0,95.0,84.0
Iowa State University,68.0,83.0,105.0,65.0,63.0
Johns Hopkins University,33.0,4.0,7.0,,49.0
Kansas State University,131.0,113.0,,99.0,125.0
Kent State University,101.0,,,120.0,132.0
Lehigh University,93.0,95.0,100.0,55.0,67.0
Louisiana State University,,,,111.0,149.0
Massachusetts Institute of Technology,4.0,18.0,6.0,50.0,15.0
Michigan State University,50.0,68.0,16.0,34.0,34.0
Michigan Technological University,,103.0,,,118.0
Mississippi State University,,120.0,,,112.0
Missouri S&T,117.0,,,,139.0
Missouri University of Technology,,,,,149.0
Montana State University,143.0,,,,
NJIT,85.0,,105.0,61.0,80.0
New Mexico State University,,,,,86.0
New York University,13.0,10.0,45.0,21.0,56.0
North Carolina State University,113.0,76.0,,48.0,45.0
Northeastern University,29.0,38.0,27.0,13.0,9.0
Northern Arizona University,143.0,,,99.0,
Northwestern University,25.0,61.0,79.0,41.0,54.0
Nova Southeastern University,137.0,,113.0,,
OHSU,,77.0,113.0,,
Ohio State University,35.0,37.0,64.0,25.0,62.0
Ohio University,,,,,139.0
Oklahoma State University,,,100.0,,
Old Dominion University,,113.0,105.0,91.0,104.0
Oregon State University,48.0,23.0,21.0,,36.0
Pennsylvania State University,41.0,25.0,32.0,10.0,19.0
Portland State University,,103.0,46.0,,159.0
Princeton University,10.0,36.0,28.0,42.0,74.0
Purdue University,21.0,47.0,68.0,7.0,22.0
Rensselaer Polytechnic Institute,61.0,103.0,,83.0,28.0
Rice University,31.0,72.0,44.0,30.0,20.0
Rochester Institute of Technology,76.0,58.0,54.0,64.0,59.0
Rutgers University,18.0,35.0,23.0,3.0,7.0
Simmons University,,,,,127.0
Stanford University,3.0,7.0,4.0,20.0,24.0
Stevens Institute of Technology,68.0,88.0,30.0,99.0,78.0
Stony Brook University,63.0,29.0,8.0,15.0,51.0
Syracuse University,79.0,,91.0,71.0,73.0
TTI Chicago,36.0,24.0,37.0,,93.0
Temple University,65.0,51.0,57.0,61.0,44.0
Texas A&M University,39.0,45.0,54.0,16.0,54.0
Texas State University,125.0,,122.0,,159.0
Texas Tech University,,,,,102.0
The University of Alabama,,91.0,,91.0,
Towson University,135.0,,,,139.0
Tufts University,74.0,91.0,,99.0,67.0
Tulane University,98.0,96.0,61.0,111.0,27.0
UCCS,137.0,103.0,68.0,99.0,122.0
UNC - Charlotte,120.0,91.0,80.0,99.0,119.0
UNC - Greensboro,131.0,,,,132.0
Univ. of California - Berkeley,2.0,27.0,2.0,95.0,30.0
Univ. of California - Davis,57.0,74.0,53.0,48.0,61.0
Univ. of California - Irvine,28.0,42.0,39.0,47.0,41.0
Univ. of California - Los Angeles,7.0,14.0,37.0,19.0,3.0
Univ. of California - Merced,65.0,103.0,12.0,,109.0
Univ. of California - Riverside,48.0,71.0,36.0,45.0,90.0
Univ. of California - San Diego,8.0,10.0,3.0,22.0,35.0
Univ. of California - Santa Barbara,34.0,17.0,40.0,13.0,46.0
Univ. of California - Santa Cruz,40.0,44.0,50.0,18.0,42.0
Univ. of Illinois at Urbana-Champaign,5.0,12.0,5.0,1.0,13.0
Univ. of Maryland - Baltimore County,101.0,72.0,105.0,111.0,76.0
University at Albany - SUNY,125.0,,,86.0,138.0
University at Buffalo,30.0,64.0,20.0,42.0,23.0
University of Alabama - Birmingham,124.0,120.0,122.0,99.0,149.0
University of Arizona,56.0,33.0,85.0,,139.0
University of Arkansas,90.0,103.0,,77.0,71.0
University of Central Florida,59.0,59.0,10.0,67.0,46.0
University of Chicago,44.0,59.0,56.0,25.0,114.0
University of Cincinnati,137.0,,105.0,,
University of Colorado - Denver,,,,99.0,
University of Colorado Boulder,54.0,28.0,98.0,50.0,56.0
University of Connecticut,83.0,85.0,117.0,37.0,104.0
University of Delaware,77.0,,62.0,39.0,102.0
University of Florida,71.0,91.0,51.0,91.0,94.0
University of Georgia,78.0,113.0,91.0,67.0,56.0
University of Hawaii at Manoa,120.0,,,,
University of Houston,125.0,57.0,64.0,57.0,104.0
University of Illinois at Chicago,32.0,21.0,46.0,8.0,18.0
University of Iowa,47.0,80.0,80.0,83.0,95.0
University of Kansas,94.0,103.0,122.0,,129.0
University of Kentucky,112.0,85.0,95.0,95.0,67.0
University of Louisiana - Lafayette,92.0,,,77.0,88.0
University of Maryland - College Park,20.0,6.0,11.0,39.0,12.0
University of Massachusetts Amherst,15.0,16.0,22.0,9.0,6.0
University of Massachusetts Boston,107.0,,105.0,,139.0
University of Massachusetts Lowell,105.0,31.0,,99.0,85.0
University of Memphis,96.0,103.0,122.0,111.0,109.0
University of Miami,,,,,149.0
University of Michigan,24.0,8.0,29.0,4.0,21.0
University of Michigan-Dearborn,131.0,100.0,,,155.0
University of Minnesota,45.0,62.0,49.0,37.0,48.0
University of Missouri,,,74.0,,155.0
University of Missouri - Kansas City,,,,,132.0
University of Nebraska,113.0,113.0,,,65.0
University of Nebraska - Omaha,143.0,,,,109.0
University of Nevada,115.0,,113.0,,119.0
University of New Hampshire,81.0,113.0,,52.0,66.0
University of New Mexico,,,,111.0,
University of New Orleans,,,,,155.0
University of North Carolina,46.0,9.0,41.0,72.0,40.0
University of North Texas,101.0,77.0,122.0,,104.0
University of Notre Dame,60.0,38.0,87.0,31.0,39.0
University of Oklahoma,,113.0,,,155.0
University of Oregon,79.0,41.0,71.0,77.0,60.0
University of Pennsylvania,14.0,5.0,19.0,61.0,29.0
University of Pittsburgh,37.0,33.0,32.0,32.0,5.0
University of Rochester,64.0,43.0,23.0,67.0,26.0
University of South Carolina,120.0,96.0,60.0,83.0,90.0
University of South Florida,,120.0,91.0,99.0,
University of Southern California,22.0,22.0,25.0,33.0,14.0
University of Southern Mississippi,,,,,104.0
University of Tennessee,131.0,,72.0,72.0,132.0
University of Texas - El Paso,,,117.0,,100.0
University of Texas at Arlington,73.0,120.0,63.0,57.0,82.0
University of Texas at Austin,19.0,13.0,26.0,72.0,16.0
University of Texas at Dallas,42.0,19.0,59.0,52.0,11.0
University of Texas at San Antonio,146.0,120.0,,111.0,132.0
University of Tulsa,,100.0,,99.0,
University of Utah,53.0,30.0,80.0,35.0,80.0
University of Vermont,,,,111.0,149.0
University of Virginia,38.0,54.0,80.0,6.0,17.0
University of Washington,9.0,1.0,18.0,27.0,38.0
University of Wisconsin - Madison,11.0,55.0,16.0,57.0,36.0
Utah State University,125.0,,,,
Vanderbilt University,107.0,100.0,74.0,111.0,97.0
Virginia Commonwealth University,125.0,103.0,,,127.0
Virginia Tech,70.0,55.0,103.0,28.0,51.0
Washington State University,72.0,120.0,95.0,111.0,53.0
Washington University in St. Louis,52.0,,51.0,72.0,10.0
Wayne State University,117.0,,67.0,67.0,98.0
West Virginia University,120.0,,43.0,,101.0
Western Michigan University,,120.0,,,
Wichita State University,88.0,,,,139.0
Worcester Polytechnic Institute,67.0,63.0,99.0,16.0,74.0
Yale University,27.0,51.0,111.0,86.0,64.0
"""

# Convert the string data to a DataFrame
df = pd.read_csv("/home/kadekool/tiered-rankings/data/csrankings/csrankings_rankings.csv")

# Convert all numeric columns to integers and handle missing values
for column in df.columns[1:]:
    df[column] = pd.to_numeric(df[column], errors='coerce')
    min_val = df[column].max()
    df[column] = df[column].fillna(min_val + 1).astype(int)

# Display the DataFrame to the user
df.to_csv("/home/kadekool/tiered-rankings/data/csrankings/csrankings.csv", index=False)
