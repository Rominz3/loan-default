import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

training_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Training Data.csv')
test_data=pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Test Data.csv')
data = pd.read_csv(r'F:\job\Loan Default Prediction Analysis\combined Data.csv')


def categorize_profession_group(Profession):
    engineering_jobs = ['Computer_hardware_engineer', 'Industrial_Engineer', 'Mechanical_engineer',
                        'Chemical_engineer', 'Biomedical_Engineer', 'Design_Engineer', 'Civil_engineer','Petroleum_Engineer','Engineer','Biomedical Engineer','Chemical engineer','Civil engineer','Computer hardware engineer','Design Engineer','Industrial Engineer','Mechanical engineer','Petroleum Engineer']

    it_jobs = ['Web_designer', 'Software_Developer', 'Computer_operator', 'Technology_specialist','Computer operator','Software Developer','Technology specialist','Web designer']

    healthcare_jobs = ['Physician', 'Psychologist', 'Dentist', 'Surgeon']

    management_jobs = ['Hotel_Manager', 'Consultant','Hotel Manager']

    arts_and_design_jobs = ['Fashion_Designer', 'Graphic_Designer', 'Artist', 'Designer', 'Architect', 'Comedian','Fashion Designer','Graphic Designer']

    research_jobs = ['Statistician', 'Microbiologist', 'Scientist', 'Analyst','Geologist']

    government_jobs = ['Lawyer', 'Politician', 'Civil_servant', 'Official','Magistrate','Civil servant']

    military_and_security_jobs = ['Air_traffic_controller', 'Army_officer', 'Police_officer', 'Aviator','Air traffic controller','Army officer','Police officer']

    customer_service_and_hospitality_jobs = ['Flight_attendant', 'Technician', 'Surveyor', 'Chef', 'Librarian','Flight attendant']

    Financial_economic_jobs =['Financial_Analyst','Economist', 'Chartered_Accountant','Chartered Accountant','Financial Analyst']



    if Profession in engineering_jobs:
        return 'Engineering'
    elif Profession in it_jobs:
        return 'IT'
    elif Profession in healthcare_jobs:
        return 'Healthcare'
    elif Profession in management_jobs:
        return 'Management'
    elif Profession in arts_and_design_jobs:
        return 'Arts_and_Design'
    elif Profession in research_jobs:
        return 'Research'
    elif Profession in government_jobs:
        return 'Government'
    elif Profession in military_and_security_jobs:
        return 'Military_and_Security'
    elif Profession in customer_service_and_hospitality_jobs:
        return 'Customer_Service_and_Hospitality'
    elif Profession in Financial_economic_jobs:
        return 'Financial_economic_jobs'
    else:
        return 'Other'


data['profession_group'] = data['Profession'].apply(categorize_profession_group)

data = data.drop('Profession', axis=1)

data.to_csv(r'F:\job\Loan Default Prediction Analysis\Data.csv',index=False)


newData= pd.read_csv(r'F:\job\Loan Default Prediction Analysis\Data.csv')

group_counts = newData['profession_group'].value_counts()

plt.figure(figsize=(15, 12))
sns.countplot(x='profession_group',hue='Risk_Flag', data=newData, palette='viridis')
plt.title('Number of Items in Each Profession Group')
plt.xlabel('Profession Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.subplots_adjust(top=0.9,bottom=0.3)
plt.show()


plt.figure(figsize=(15, 8))
plt.pie(group_counts, labels=group_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis'))
plt.title('Distribution of Items Across Profession Groups')
plt.show()