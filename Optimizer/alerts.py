'''
Set of functions to handle local optimization failure, 
and send email alerts.
'''
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from string import Template

### Set of globals to be set relative to workflow ###
USERS_FILE = './operators.txt'
EMAIL_LOGIN = ''
EMAIL_PASSWORD = ''
SMTP_PORT = 999
SMTP_ADDR = ''
### Set of globals to be set relative to workflow ###

def get_contacts(fname):
    names = []
    emails = []
    with open(fname, 'r') as f:
        for line in f:
            cols = line.split(sep=',')
            names.append(cols[0])
            emails.append(cols[1])
            
    return zip(names, emails) 


def send_alert(trace):

    contacts = get_contacts(USERS_FILE)
    
    # set up SMTP server
    s = smtplib.SMTP(host=SMTP_ADDR, port=SMTP_PORT)
    s.starttls()
    s.login(EMAIL_LOGIN, EMAIL_PASSWORD)

    template  = Template("Dear ${PERSON_NAME},\n\nThe KUKA workflow has encountered and error "
                         "at the algorithm level. Not to point fingers, but it's probably PMM's fault. The "
                         "traceback follows:\n${TRACEBACK} \n\nBest regards,\nYour friendly neighborhood optimizer.")


    for name, email in contacts:
        msg = MIMEMultipart()
        
        body = template.substitute(PERSON_NAME=name.title(), TRACEBACK=trace)
        
        msg['From'] = EMAIL_LOGIN
        mgs['To'] = email
        msg['Subject']= 'KUKA workflow optimization failure!'
        
        msg.attach(MIMEText(body, 'plain'))
        
        s.send_message(msg)
        del msg
        
    s.quit()
    
if __name__ == '__main__':
    tb = "Nothing to see here, move along...."
    send_alert(tb)