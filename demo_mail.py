import smtplib

# Setup the SMTP server
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()

# Use the 16-character app password instead of your Gmail password
server.login('fire.alert.notification.7679@gmail.com', 'empr vcwv oupw ntnd')

# Send the email
server.sendmail(
    'fire.alert.notification.7679@gmail.com', 
    'atulrahul704@gmail.com', 
    'Subject: Fire Alert Notification\n\nfire alert fire alert .'
)

print('Mail sent successfully!')
server.quit()
