from config import *

import smtplib

def send_email_notification(message = "default training notification email"):
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(config.notif_email_address, config.notif_email_password)
    s.sendmail(config.notif_email_address, config.notif_targets, message)

def maybe_send_email_notification(message = "default training notification email"):
    if not config.send_email:
        return
    send_email_notification(message)


if __name__ != "__main__":
    pass

print("Testing Email notification")
send_email_notification("email notification test")
print("mail sent")